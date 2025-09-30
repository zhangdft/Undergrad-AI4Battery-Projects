#!/usr/bin/env python3
"""
extract_B0005_discharges.py

从 B0005.mat 中提取每次 discharge 循环的 Time 和 Current_measured 数据，
将每次循环保存为单独的 CSV 文件，保存在指定输出目录下。

用法：
    python extract_B0005_discharges.py
或：
    python extract_B0005_discharges.py --matpath Submissions/richardzang/topic1/week3/B0005.mat --outdir Submissions/richardzang/topic1/week3/B0005_170cycles_Discharge

依赖：numpy, scipy, pandas (可选)
"""

from pathlib import Path
import argparse
import numpy as np
import sys

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy import io as sio
except Exception:
    sio = None


def unwrap(x):
    """Unwrap common scipy loadmat containers (0-dim or single-element arrays)."""
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        if x.size == 1:
            return x.flatten()[0]
    return x


def load_mat(mat_path: Path):
    if sio is None:
        raise RuntimeError('scipy is required to load .mat files (install scipy)')
    return sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)


def extract_and_save(mat_path: Path, outdir: Path):
    if not mat_path.exists():
        print('MAT file not found:', mat_path)
        return

    print('Loading', mat_path)
    mat = load_mat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print('Top-level keys:', keys)

    if 'B0005' not in mat:
        print('B0005 key not found in mat file')
        return

    entry = unwrap(mat['B0005'])
    cycles = unwrap(getattr(entry, 'cycle', None))
    if cycles is None:
        print('No cycles found in B0005')
        return

    outdir.mkdir(parents=True, exist_ok=True)

    total = len(cycles)
    print(f'Found {total} cycles; scanning for discharge cycles...')

    saved = 0
    for idx, c in enumerate(cycles):
        c = unwrap(c)
        ctype = getattr(c, 'type', None)
        if ctype is None:
            continue
        if str(ctype).lower() != 'discharge':
            continue

        data = unwrap(getattr(c, 'data', None))
        if data is None:
            print(f'cycle {idx}: no data field, skipping')
            continue

        # time candidates
        time = getattr(data, 'Time', None)
        if time is None:
            time = getattr(data, 'time', None)

        # current candidates
        current = getattr(data, 'Current_measured', None)
        if current is None:
            current = getattr(data, 'Current', None)
        if current is None:
            current = getattr(data, 'Current_discharge', None)

        if time is None or current is None:
            print(f'cycle {idx} (discharge): missing time or current, skipping')
            continue

        t = np.asarray(time).ravel()
        cur = np.asarray(current).ravel()
        n = min(t.size, cur.size)
        t = t[:n]
        cur = cur[:n]

        # filename: B0005_discharge_001.csv ... (1-based count of discharge files)
        saved += 1
        fname = f'B0005_discharge_{saved:03d}.csv'
        out_path = outdir / fname

        try:
            if pd is not None:
                df = pd.DataFrame({'time_s': t, 'current_A': cur})
                df.to_csv(out_path, index=False)
            else:
                header = 'time_s,current_A'
                data_out = np.column_stack([t, cur])
                np.savetxt(str(out_path), data_out, delimiter=',', header=header, comments='')
            if saved % 10 == 0 or saved <= 3:
                print(f'Wrote {out_path} (n={n})')
        except Exception as e:
            print(f'Failed to write {out_path}:', e)

    print(f'Done. Saved {saved} discharge cycle files to: {outdir}')


def main():
    p = argparse.ArgumentParser(description='Extract discharge cycles time & current from B0005.mat')
    p.add_argument('--matpath', type=str, default='Submissions/richardzang/topic1/week3/B0005.mat')
    p.add_argument('--outdir', type=str, default='Submissions/richardzang/topic1/week3/B0005_170cycles_Discharge')
    args = p.parse_args()

    mat_path = Path(args.matpath)
    outdir = Path(args.outdir)

    try:
        extract_and_save(mat_path, outdir)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
