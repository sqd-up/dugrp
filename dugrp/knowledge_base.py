"""
DUGRP Innovation I：HNSW 知识库
================================
论文 Section 4.1.3

"All N historical embeddings {z_i}^N_{i=1} are indexed via HNSW [25]
 (implemented in FAISS [26]). The multi-layer graph structure admits
 O(log N) cosine similarity queries, enabling real-time retrieval at
 N=10^5 within the 12ms latency budget."

Table 4（HNSW 延迟与召回率）：
  N=10^4: Exact=42.1ms, HNSW=0.21ms, Speedup=52×, Recall@10=0.997
  N=10^5: Exact=421ms,  HNSW=2.13ms, Speedup=198×, Recall@10=0.994
  N=10^6: Exact=4183ms, HNSW=4.71ms, Speedup=888×, Recall@10=0.989

实现选择：
  优先使用 faiss.IndexHNSWFlat（论文明确指定 FAISS）。
  若 faiss 不可用，自动降级为 hnswlib（API 兼容）。
  若两者均不可用，降级为 numpy 精确搜索（仅限小规模测试）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入后端
_BACKEND = "numpy"  # 降级后端
try:
    import faiss
    _BACKEND = "faiss"
    logger.info("[KnowledgeBase] 使用 FAISS-HNSW 后端")
except ImportError:
    try:
        import hnswlib
        _BACKEND = "hnswlib"
        logger.info("[KnowledgeBase] 使用 hnswlib 后端（faiss 未安装）")
    except ImportError:
        logger.warning(
            "[KnowledgeBase] faiss 和 hnswlib 均未安装，降级为 numpy 精确搜索。"
            "安装：pip install faiss-cpu  或  pip install hnswlib"
        )


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """
    检索结果。

    sequences  : (K, T_context, D) — 检索到的历史上下文序列
    futures    : (K, T_future,  D) — 对应的历史未来序列
    scores     : (K,)              — 余弦相似度分数 ∈ [-1, 1]
    indices    : (K,)              — 在知识库中的索引
    """
    sequences: np.ndarray   # (K, T_context, D)
    futures:   np.ndarray   # (K, T_future,  D)
    scores:    np.ndarray   # (K,)
    indices:   np.ndarray   # (K,)


# ---------------------------------------------------------------------------
# 知识库主类
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """
    HNSW 知识库。

    存储三元组 (z_i, X_i, Y_i)：
      z_i ∈ S^{m-1}：对比学习嵌入（用于 HNSW 检索）
      X_i ∈ R^{L×D}：历史上下文序列
      Y_i ∈ R^{H×D}：对应未来序列

    参数（论文 Section 5.1.4 / Table 4）
    -----
    embed_dim         : m=128
    hnsw_M            : HNSW 每层邻居数（32，平衡性能与内存）
    hnsw_ef_construction: 构建搜索宽度（200）
    hnsw_ef_search    : 查询搜索宽度（50，Recall@10=0.997）
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hnsw_M: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ):
        self.embed_dim = embed_dim
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # 序列存储（Python list，允许不同长度）
        self._contexts: List[np.ndarray] = []   # 每项 (T_context, D)
        self._futures:  List[np.ndarray] = []   # 每项 (T_future, D)
        self._embeddings: List[np.ndarray] = [] # 每项 (m,)

        # HNSW 索引（懒初始化，添加数据后构建）
        self._index = None
        self._index_is_stale = False  # 标记：有新数据尚未重建索引

    # -----------------------------------------------------------------------
    # 属性
    # -----------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._contexts)

    # -----------------------------------------------------------------------
    # 添加数据
    # -----------------------------------------------------------------------
    def add(
        self,
        context: np.ndarray,    # (T_context, D)
        future: np.ndarray,     # (T_future, D)
        embedding: np.ndarray,  # (m,)
    ):
        """
        向知识库添加一条记录。
        添加后标记索引为 stale，下次检索时自动重建。
        """
        assert embedding.shape == (self.embed_dim,), (
            f"embedding 维度错误：期望 ({self.embed_dim},)，实际 {embedding.shape}"
        )
        self._contexts.append(context.astype(np.float32))
        self._futures.append(future.astype(np.float32))
        # 保持 L2 归一化（HNSW 余弦距离 = 内积距离，需提前归一化）
        norm = np.linalg.norm(embedding) + 1e-8
        self._embeddings.append((embedding / norm).astype(np.float32))
        self._index_is_stale = True

    def add_batch(
        self,
        contexts:   np.ndarray,   # (N, T_context, D)
        futures:    np.ndarray,   # (N, T_future, D)
        embeddings: np.ndarray,   # (N, m)
    ):
        """批量添加，效率更高（构建索引仅一次）。"""
        N = len(contexts)
        for i in range(N):
            norm = np.linalg.norm(embeddings[i]) + 1e-8
            self._contexts.append(contexts[i].astype(np.float32))
            self._futures.append(futures[i].astype(np.float32))
            self._embeddings.append((embeddings[i] / norm).astype(np.float32))
        self._index_is_stale = True

    # -----------------------------------------------------------------------
    # 重建 HNSW 索引
    # -----------------------------------------------------------------------
    def _rebuild_index(self):
        """
        用当前所有嵌入重建 HNSW 索引。
        仅在 _index_is_stale=True 时调用（懒重建）。
        """
        N = self.size
        if N == 0:
            return

        embs = np.stack(self._embeddings, axis=0).astype(np.float32)  # (N, m)

        if _BACKEND == "faiss":
            self._index = faiss.IndexHNSWFlat(self.embed_dim, self.hnsw_M)
            self._index.hnsw.efConstruction = self.hnsw_ef_construction
            self._index.hnsw.efSearch = self.hnsw_ef_search
            # IndexHNSWFlat 使用 L2 距离；对 L2 归一化向量，L2 ↔ 余弦距离
            self._index.add(embs)

        elif _BACKEND == "hnswlib":
            self._index = hnswlib.Index(space="cosine", dim=self.embed_dim)
            self._index.init_index(
                max_elements=max(N, 1024),
                ef_construction=self.hnsw_ef_construction,
                M=self.hnsw_M,
            )
            self._index.set_ef(self.hnsw_ef_search)
            self._index.add_items(embs, np.arange(N))

        else:
            # numpy 精确搜索（无索引结构，O(N)）
            self._index = embs  # 直接存矩阵

        self._index_is_stale = False

    # -----------------------------------------------------------------------
    # 检索（Algorithm 2 line 3）
    # KB_K ← I.SEARCH(z_t, K) ▷ O(log N)
    # -----------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,  # (m,) L2 归一化
        K: int,
    ) -> RetrievalResult:
        """
        检索最相似的 K 条历史记录。

        参数
        ----
        query_embedding : 查询嵌入，L2 归一化，(m,)
        K               : 检索数量，由 K*(τ) 动态决定

        返回
        ----
        RetrievalResult（sequences, futures, scores, indices）
        """
        if self.size == 0:
            raise RuntimeError("知识库为空，请先调用 add() 或 add_batch()")

        # 懒重建索引
        if self._index_is_stale or self._index is None:
            self._rebuild_index()

        K = min(K, self.size)
        query = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query) + 1e-8
        query = query / norm

        if _BACKEND == "faiss":
            query_2d = query.reshape(1, -1)
            # FAISS HNSW 返回 L2 距离；对归一化向量 L2² = 2(1 - cosine)
            distances, indices = self._index.search(query_2d, K)
            distances = distances[0]    # (K,)
            indices   = indices[0]      # (K,)
            # 转为余弦相似度：cos = 1 - L2²/2
            scores = 1.0 - distances / 2.0

        elif _BACKEND == "hnswlib":
            indices, distances = self._index.knn_query(query, k=K)
            indices   = indices[0]    # (K,)
            distances = distances[0]  # (K,)，cosine distance = 1 - cosine_sim
            scores = 1.0 - distances

        else:
            # numpy 精确余弦相似度
            sims = self._index @ query   # (N,)
            top_k = np.argsort(sims)[::-1][:K]
            indices = top_k.astype(np.int64)
            scores  = sims[indices]

        # 过滤无效索引（FAISS 可能返回 -1）
        valid = indices >= 0
        indices = indices[valid]
        scores  = scores[valid]

        sequences = np.stack([self._contexts[i] for i in indices], axis=0)  # (K, T_ctx, D)
        futures   = np.stack([self._futures[i]  for i in indices], axis=0)  # (K, T_fut, D)

        return RetrievalResult(
            sequences=sequences,
            futures=futures,
            scores=scores.astype(np.float32),
            indices=indices,
        )

    # -----------------------------------------------------------------------
    # 持久化
    # -----------------------------------------------------------------------
    def save(self, path: str):
        """保存知识库到磁盘（.npz 格式）。"""
        embs = np.stack(self._embeddings) if self._embeddings else np.zeros((0, self.embed_dim))
        np.savez_compressed(
            path,
            contexts=np.array(self._contexts, dtype=object),
            futures=np.array(self._futures, dtype=object),
            embeddings=embs,
            metadata=np.array([self.embed_dim, self.hnsw_M,
                                self.hnsw_ef_construction, self.hnsw_ef_search]),
        )
        logger.info(f"[KnowledgeBase] 已保存 {self.size} 条记录至 {path}.npz")

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        """从磁盘加载知识库。"""
        data = np.load(f"{path}.npz" if not path.endswith(".npz") else path,
                       allow_pickle=True)
        meta = data["metadata"].tolist()
        kb = cls(
            embed_dim=int(meta[0]),
            hnsw_M=int(meta[1]),
            hnsw_ef_construction=int(meta[2]),
            hnsw_ef_search=int(meta[3]),
        )
        contexts   = data["contexts"]
        futures    = data["futures"]
        embeddings = data["embeddings"]
        for i in range(len(embeddings)):
            kb._contexts.append(contexts[i])
            kb._futures.append(futures[i])
            kb._embeddings.append(embeddings[i])
        kb._index_is_stale = True
        logger.info(f"[KnowledgeBase] 已加载 {kb.size} 条记录")
        return kb
