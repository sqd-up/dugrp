# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def process_rov_csv(csv_path: str, output_path: str, seq_length: int = 1000):
    print(f"正在读取 CSV 文件: {csv_path} ...")
    df = pd.read_csv(csv_path)

    try:
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        qx = df['qx'].values
        qy = df['qy'].values
        qz = df['qz'].values
        qw = df['qw'].values
    except KeyError as e:
        print(f"找不到列名: {e}。请打开 CSV 文件核对确切的表头名称，并修改代码！")
        return

    print("正在将四元数转换为 Roll, Pitch, Yaw (弧度制) ...")
    quats = np.stack([qx, qy, qz, qw], axis=1)
    rot = R.from_quat(quats)
    euler = rot.as_euler('xyz', degrees=False)
    
    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]

    trajectory = np.stack([x, y, z, roll, pitch, yaw], axis=1).astype(np.float32)
    total_steps = len(trajectory)
    print(f"成功提取单条长轨迹，总长度: {total_steps} 步")

    N = total_steps // seq_length
    if N == 0:
        print(f"数据总长度 ({total_steps}) 小于设定的序列长度 ({seq_length})，无法切分！")
        return

    sequences = trajectory[:N * seq_length].reshape(N, seq_length, 6)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, sequences)
    print(f"\n处理完成！")
    print(f"数据已保存至: {output_path}")
    print(f"最终数据形状: {sequences.shape} (N={N}条序列, T={seq_length}步长, D=6维特征)")

if __name__ == "__main__":
    INPUT_CSV = "odom-12-01-2024-run1.csv"
    OUTPUT_NPY = "data/rov_data.npy"
    process_rov_csv(INPUT_CSV, OUTPUT_NPY)
