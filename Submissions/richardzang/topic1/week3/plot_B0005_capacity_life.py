#!/usr/bin/env python3
"""
plot_B0005_capacity_life.py

从指定目录读取每个放电循环的 CSV（每个 CSV 包含 time 与 current 列），计算每次放电的电池容量并绘制寿命曲线。

容量计算公式（已由用户给出并在下方验证适用性）：
    capacity_Ah = np.trapz(np.abs(Current_A), Time_s) / 3600

分析与注意事项：
- 积分公式对非均匀采样（非等步长时间）是有效的，因为 np.trapz 可以处理不等间隔的 x（Time_s）。
- 公式的单位要求：Time_s 单位为秒，Current_A 单位为安培。积分得到库仑/秒乘以秒 => 库仑，除以 3600 得到 安时(Ah)。
- 因为你的 CSV 中电流为负值（放电通常为负），脚本对电流取绝对值 np.abs(current) 再积分。
- 在运行时脚本会做一个简单检查：如果 Time 的范围非常小（例如全部 < 1），则提示可能单位不是秒，需要用户确认/转换。

用法：
    python plot_B0005_capacity_life.py
或指定目录/输出：
    python plot_B0005_capacity_life.py --indir Submissions/richardzang/topic1/week3/B0005_170cycles_Discharge --out ./Submissions/richardzang/topic1/week3/B0005_capacity_life.png

依赖：numpy, matplotlib, pandas (可选)
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    import pandas as pd
except Exception:
    pd = None


def read_csv_file(path: Path):
    """Read CSV and return (time_s, current_A) numpy arrays.
    Tries pandas first, falls back to numpy.genfromtxt.
    Accepts columns named 'time_s' and 'current_A', or uses first two columns.
    """
    if pd is not None:
        try:
            df = pd.read_csv(path)
            cols = [c.lower() for c in df.columns]
            if 'time_s' in cols and 'current_a' in cols:
                t = df.iloc[:, cols.index('time_s')].to_numpy(dtype=float)
                i = df.iloc[:, cols.index('current_a')].to_numpy(dtype=float)
                return t, i
            # fallback: try to find columns containing 'time' and 'current'
            time_col = None
            current_col = None
            for c in df.columns:
                lc = c.lower()
                if time_col is None and 'time' in lc:
                    time_col = c
                if current_col is None and ('current' in lc or 'i' == lc):
                    current_col = c
            if time_col is not None and current_col is not None:
                t = df[time_col].to_numpy(dtype=float)
                i = df[current_col].to_numpy(dtype=float)
                return t, i
            # final fallback: use first two numeric columns
            numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if len(numeric_cols) >= 2:
                t = df[numeric_cols[0]].to_numpy(dtype=float)
                i = df[numeric_cols[1]].to_numpy(dtype=float)
                return t, i
        except Exception:
            pass

    # numpy fallback
    try:
        data = np.genfromtxt(str(path), delimiter=',', names=True)
        if data.size == 0:
            return None, None
        names = data.dtype.names
        if names is not None:
            lname = [n.lower() for n in names]
            if 'time_s' in lname and 'current_a' in lname:
                t = data['time_s']
                i = data['current_a']
                return t.astype(float), i.astype(float)
            # try to find time/current
            time_idx = None
            current_idx = None
            for j, n in enumerate(lname):
                if time_idx is None and 'time' in n:
                    time_idx = j
                if current_idx is None and 'current' in n:
                    current_idx = j
            if time_idx is not None and current_idx is not None:
                t = data[names[time_idx]]
                i = data[names[current_idx]]
                return t.astype(float), i.astype(float)
    except Exception:
        # try simple load without header
        try:
            arr = np.loadtxt(str(path), delimiter=',')
            if arr.ndim == 1 and arr.size >= 2:
                t = arr[:, 0]
                i = arr[:, 1]
                return t.astype(float), i.astype(float)
        except Exception:
            return None, None

    return None, None


def compute_capacity_Ah(t: np.ndarray, i: np.ndarray):
    """Compute capacity in Ah using trapezoidal integration of |I| over Time (seconds).
    Returns capacity in Ah.
    """
    # Ensure 1D float arrays
    t = np.asarray(t).ravel().astype(float)
    i = np.asarray(i).ravel().astype(float)
    if t.size == 0 or i.size == 0:
        return None
    # Ensure lengths match
    n = min(t.size, i.size)
    t = t[:n]
    i = i[:n]
    # If time is not strictly increasing, sort by time
    if not np.all(np.diff(t) >= 0):
        order = np.argsort(t)
        t = t[order]
        i = i[order]

    # Basic sanity: check time units plausibility
    t_span = t.max() - t.min()
    if t_span <= 0:
        return None
    # warn if time likely not in seconds (e.g., max < 1 meaning maybe hours?)
    if t.max() < 1:
        print('Warning: max(Time) < 1 — verify Time units are seconds. If Time is in hours/minutes, rescale to seconds before integration.')

    # integrate absolute current over time and convert coulombs -> Ah
    coulomb = np.trapz(np.abs(i), t)
    capacity_Ah = coulomb / 3600.0
    return capacity_Ah


def main():
    p = argparse.ArgumentParser(description='Compute capacity per discharge CSV and plot capacity vs cycle count')
    p.add_argument('--indir', type=str, default='Submissions/richardzang/topic1/week3/B0005_170cycles_Discharge')
    p.add_argument('--out', type=str, default='Submissions/richardzang/topic1/week3/B0005_capacity_life.png')
    args = p.parse_args()

    indir = Path(args.indir)
    outpng = Path(args.out)

    if not indir.exists() or not indir.is_dir():
        print('Input directory does not exist:', indir)
        sys.exit(1)

    files = sorted(indir.glob('*.csv'))
    if not files:
        print('No CSV files found in', indir)
        sys.exit(1)

    capacities = []
    labels = []
    for f in files:
        t, i = read_csv_file(f)
        if t is None or i is None:
            print('Skipping (cannot read or missing cols):', f.name)
            continue
        cap = compute_capacity_Ah(t, i)
        if cap is None:
            print('Skipping (bad data):', f.name)
            continue
        capacities.append(cap)
        labels.append(f.name)

    if not capacities:
        print('No capacities computed')
        sys.exit(1)

    # Plot
    x = np.arange(1, len(capacities) + 1)
    y = np.array(capacities)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('Discharge cycle count')
    plt.ylabel('Capacity (Ah)')
    plt.title('B0005 discharge capacity life curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outpng.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outpng), dpi=200)
    print(f'Saved capacity life plot to: {outpng} (points={len(y)})')

    # also print first few capacities
    for i, c in enumerate(y[:10], start=1):
        print(f'{i}: {c:.6f} Ah')


if __name__ == '__main__':
    main()
