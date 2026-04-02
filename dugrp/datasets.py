"""
DUGRP 数据集工具
================
论文 Section 5.1.1 (Datasets)

支持的数据集：
  1. UnderwaterTele-ROV（主数据集，非公开）
     - 真实 BlueROV2 水下机器人遥操作，6-DOF
     - 12小时，70/10/20 划分
     - 按论文 Table 1 三种时延场景注入时延

  2. ETTh1（公开，用于零样本泛化验证）
     - 电力变压器温度，48传感器
     - 使用 D3 时延配置注入（delay-injected）
     - Section 5.1.1："ETTh1 (delay-injected): M4-Weekly format, H=48"

时延注入（Table 1）：
  D1: τ = 200 + N(0, 5²)         ms，均值 200ms
  D2: τ = 200 + 15·sin(2πt/60)  ms，周期正弦
  D3: τ = 300 + 20·sin(2πt/120) + 10·sin(2πt/30)  ms，复合正弦

时延注入实现：
  对时间序列 x[t]，时延 τ(t) 意味着观测值为 x[t - τ(t)/Δt_step]。
  即：通过索引偏移模拟"只能看到过去 τ ms 之前的数据"。
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 时延配置（Table 1）
# ---------------------------------------------------------------------------
@dataclass
class DelayProfile:
    """Table 1 中的时延场景配置。"""
    name: str
    mean_delay_ms: float        # 均值时延（ms）

    def sample(self, t: float, dt_ms: float = 1.0) -> float:
        """在时刻 t (ms) 采样当前时延值 τ(t)。"""
        raise NotImplementedError


class DelayProfileD1(DelayProfile):
    """D1：200 + N(0, 5²) ms，均值 200ms（模拟稳定水下信道）"""
    def __init__(self):
        super().__init__(name="D1", mean_delay_ms=200.0)
        self._std = 5.0

    def sample(self, t: float, dt_ms: float = 1.0) -> float:
        return float(np.clip(200.0 + np.random.normal(0, self._std), 50.0, 400.0))


class DelayProfileD2(DelayProfile):
    """D2：200 + 15·sin(2πt/60) ms，周期 60s（模拟周期性网络抖动）"""
    def __init__(self):
        super().__init__(name="D2", mean_delay_ms=200.0)

    def sample(self, t: float, dt_ms: float = 1.0) -> float:
        t_s = t / 1000.0   # 转换为秒
        return float(200.0 + 15.0 * np.sin(2 * np.pi * t_s / 60.0))


class DelayProfileD3(DelayProfile):
    """D3：300 + 20·sin(2πt/120) + 10·sin(2πt/30) ms，均值 300ms（复合时延，最恶劣）"""
    def __init__(self):
        super().__init__(name="D3", mean_delay_ms=300.0)

    def sample(self, t: float, dt_ms: float = 1.0) -> float:
        t_s = t / 1000.0   # 转换为秒
        return float(
            300.0
            + 20.0 * np.sin(2 * np.pi * t_s / 120.0)
            + 10.0 * np.sin(2 * np.pi * t_s / 30.0)
        )


# 预定义实例（论文 Table 1）
DELAY_PROFILES: Dict[str, DelayProfile] = {
    "D1": DelayProfileD1(),
    "D2": DelayProfileD2(),
    "D3": DelayProfileD3(),
    "no_delay": DelayProfileD1(),  # τ≈0，用 D1 最小值
}


# ---------------------------------------------------------------------------
# 时延注入工具函数
# ---------------------------------------------------------------------------
def inject_delay(
    sequence: np.ndarray,     # (T, D) — 原始时间序列
    delay_profile: DelayProfile,
    dt_ms: float = 100.0,     # 采样间隔 (ms)，BlueROV2 @ 10Hz → dt=100ms
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将时延注入时间序列，模拟遥操作中控制器只能看到延迟数据。

    方法：
      对每个时刻 t，采样 τ(t)，计算延迟步数 d = round(τ(t) / dt_ms)，
      观测值 = sequence[max(0, t-d)]（边界用第一个值填充）。

    参数
    ----
    sequence      : (T, D) — 原始序列（视为真值）
    delay_profile : 时延配置（D1/D2/D3）
    dt_ms         : 采样间隔 (ms)

    返回
    ----
    delayed_sequence : (T, D) — 时延后可观测序列
    delay_array      : (T,)   — 每步时延值 (ms)
    """
    T, D = sequence.shape
    delayed = np.zeros_like(sequence)
    delays  = np.zeros(T, dtype=np.float32)

    for t in range(T):
        t_ms = t * dt_ms
        tau = delay_profile.sample(t_ms, dt_ms)
        delays[t] = tau
        d = int(round(tau / dt_ms))
        src_t = max(0, t - d)
        delayed[t] = sequence[src_t]

    return delayed.astype(np.float32), delays


# ---------------------------------------------------------------------------
# ETTh1 数据集加载器（公开数据，用于论文 Table 2 零样本实验）
# ---------------------------------------------------------------------------
class ETTh1Dataset:
    """
    ETTh1 数据集（电力变压器温度，Section 5.1.1）。

    "ETTh1 (delay-injected): Electricity transformer temperature
     dataset [28] with synthetic D3-profile delay injection, to assess
     generalization beyond robotics. H=48"

    自动下载 ETTh1.csv（若本地不存在），并支持三种时延场景注入。
    """

    # ETTh1 原始 CSV 下载地址（来自官方 TimesNet/PatchTST 仓库）
    _CSV_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-data/ETTh1.csv"
    _LOCAL_PATH = "data/ETTh1.csv"

    # ETTh1 特征列（7列：HUFL, HULL, MUFL, MULL, LUFL, LULL, OT）
    _FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    def __init__(
        self,
        data_dir: str = "data",
        context_length: int = 512,
        prediction_length: int = 48,    # H=48（论文 Section 5.1.1）
        delay_profile_name: str = "D3",
        dt_ms: float = 3600000.0,       # ETTh1 @ 1小时间隔 = 3600s = 3600000ms
        normalize: bool = True,
    ):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.delay_profile = DELAY_PROFILES[delay_profile_name]
        self.dt_ms = dt_ms
        self.normalize = normalize

        csv_path = os.path.join(data_dir, "ETTh1.csv")
        data = self._load_csv(csv_path, data_dir)
        # data: (T, 7)

        # 归一化（Z-score，按训练集统计量）
        T = len(data)
        train_end = int(T * 0.7)
        if normalize:
            self.mean = data[:train_end].mean(axis=0)
            self.std  = data[:train_end].std(axis=0) + 1e-8
            data = (data - self.mean) / self.std

        # 注入时延
        self.data_clean, self.delays = data, np.zeros(len(data))
        self.data_delayed, self.delays = inject_delay(data, self.delay_profile, dt_ms)

        # 划分：70/10/20（Section 5.1.1）
        self.train_end = train_end
        self.val_end   = int(T * 0.80)
        self.T         = T

    def _load_csv(self, csv_path: str, data_dir: str) -> np.ndarray:
        """加载 ETTh1.csv，若不存在则自动下载。"""
        if not os.path.exists(csv_path):
            logger.info(f"[ETTh1] 下载数据集到 {csv_path}...")
            os.makedirs(data_dir, exist_ok=True)
            try:
                urllib.request.urlretrieve(self._CSV_URL, csv_path)
                logger.info("[ETTh1] 下载完成。")
            except Exception as e:
                raise RuntimeError(
                    f"无法下载 ETTh1.csv：{e}\n"
                    f"请手动下载到 {csv_path}"
                )

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            data = df[self._FEATURE_COLS].values.astype(np.float32)
        except ImportError:
            # 无 pandas：用 numpy 读取（跳过表头）
            data = np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                                 usecols=range(1, 8), dtype=np.float32)
        return data

    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (train, val, test) 三个完整延迟序列。"""
        train = self.data_delayed[:self.train_end]
        val   = self.data_delayed[self.train_end:self.val_end]
        test  = self.data_delayed[self.val_end:]
        return train, val, test

    def get_windows(
        self,
        split: str = "test",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取滑动窗口 (context, future, delay)。

        参数
        ----
        split : "train" / "val" / "test"

        返回
        ----
        contexts : (N, L, D)
        futures  : (N, H, D)
        delays   : (N,) — 每个窗口末尾的时延 (ms)
        """
        L, H = self.context_length, self.prediction_length

        if split == "train":
            seq    = self.data_delayed[:self.train_end]
            delays = self.delays[:self.train_end]
        elif split == "val":
            seq    = self.data_delayed[self.train_end:self.val_end]
            delays = self.delays[self.train_end:self.val_end]
        else:
            seq    = self.data_delayed[self.val_end:]
            delays = self.delays[self.val_end:]

        contexts_list, futures_list, tau_list = [], [], []
        for t in range(L, len(seq) - H + 1):
            contexts_list.append(seq[t - L: t])
            futures_list.append(seq[t: t + H])
            tau_list.append(delays[t])

        return (
            np.stack(contexts_list).astype(np.float32),
            np.stack(futures_list).astype(np.float32),
            np.array(tau_list, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# 合成数据集（用于单元测试和快速验证，无需真实数据）
# ---------------------------------------------------------------------------
def make_synthetic_rov_data(
    n_sequences: int = 100,
    T: int = 1024,
    D: int = 6,
    dt_ms: float = 100.0,
    delay_profile_name: str = "D3",
    noise_std: float = 0.05,
    seed: int = 42,
    context_length: int = 512,     # <--- 已修改：接受动态传入参数
    prediction_length: int = 24,   # <--- 已修改：接受动态传入参数
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成合成的 6-DOF 水下机器人遥操作数据（用于代码测试）。

    模拟 BlueROV2 运动：6个控制通道（surge, sway, heave, roll, pitch, yaw）
    用低频正弦叠加 + 随机游走近似真实遥操作轨迹。

    返回
    ----
    contexts : (N, L, D)  — 时延上下文（已注入时延）
    futures  : (N, H, D)  — 真值未来序列
    delays   : (N,)       — 对应时延 (ms)
    """
    rng = np.random.RandomState(seed)
    profile = DELAY_PROFILES[delay_profile_name]

    # 生成基础 6-DOF 轨迹
    freqs = [0.1, 0.15, 0.08, 0.05, 0.12, 0.07]  # Hz（每个 DOF 不同频率）
    t = np.arange(T) * dt_ms / 1000.0              # 秒

    sequences = []
    for _ in range(n_sequences):
        seq = np.zeros((T, D), dtype=np.float32)
        for d in range(D):
            amp   = rng.uniform(0.3, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            drift = np.cumsum(rng.randn(T) * 0.001)  # 随机游走
            seq[:, d] = (
                amp * np.sin(2 * np.pi * freqs[d] * t + phase)
                + drift
                + rng.randn(T) * noise_std
            )
        sequences.append(seq)

    all_sequences = np.stack(sequences)  # (N, T, D)

    # 提取窗口并注入时延
    L, H = context_length, prediction_length  # <--- 已修改：使用传入参数
    
    ctx_list, fut_list, tau_list = [], [], []
    for seq in all_sequences:
        delayed, delays_arr = inject_delay(seq, profile, dt_ms)
        for t_idx in range(L, T - H + 1, 10):  # 步长10避免重叠过多
            ctx_list.append(delayed[t_idx - L: t_idx])
            fut_list.append(seq[t_idx: t_idx + H])   # 未来序列用真值
            tau_list.append(delays_arr[t_idx])

    return (
        np.stack(ctx_list).astype(np.float32),
        np.stack(fut_list).astype(np.float32),
        np.array(tau_list, dtype=np.float32),
    )