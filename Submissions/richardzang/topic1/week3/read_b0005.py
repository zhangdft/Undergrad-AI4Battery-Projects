"""
read_b0005.py

简单脚本：读取并打印 B0005.mat 的结构信息与样例数据。

用法：
    python read_b0005.py
或指定文件：
    python read_b0005.py --matpath path/to/B0005.mat

依赖：numpy, scipy, h5py (可选), pathlib

输出：顶层 keys，B0005.cycle 长度，type 分布，以及前几个 cycle 的 time/voltage 简要样例。
"""

import argparse
from pathlib import Path
import pprint
import numpy as np

try:
    from scipy import io as sio
except Exception:
    sio = None

try:
    import h5py
except Exception:
    h5py = None


def unwrap(x):
    """Unwrap common scipy loadmat containers (0-dim or single-element arrays)."""
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        if x.size == 1:
            return x.flatten()[0]
    return x


def try_scipy_load(mat_path: Path):
    if sio is None:
        raise RuntimeError('scipy not available')
    return sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)


def try_h5py_load(mat_path: Path):
    if h5py is None:
        raise RuntimeError('h5py not available')
    f = h5py.File(str(mat_path), 'r')
    # convert h5py groups to a simple dict with info only (no deep conversion)
    info = {k: {'type': type(f[k]).__name__, 'shape': getattr(f[k], 'shape', None)} for k in f.keys()}
    f.close()
    return {'__h5py_summary__': info}


def inspect_b0005(mat):
    # mat may be dict-like from scipy.loadmat
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print('Top-level keys:', keys)

    if 'B0005' not in mat:
        print('No B0005 key found in MAT file.')
        return

    entry = unwrap(mat['B0005'])
    print('\nType of B0005 object:', type(entry))

    # try to access cycle
    cycles = getattr(entry, 'cycle', None)
    cycles = unwrap(cycles)
    if cycles is None:
        print('No cycles found inside B0005')
        return

    print('Number of cycles:', len(cycles))

    # collect types
    types = []
    for c in cycles:
        c = unwrap(c)
        t = getattr(c, 'type', None)
        types.append(str(t))

    from collections import Counter
    cnt = Counter(types)
    print('\nCycle type counts:')
    for t, n in cnt.items():
        print(f'  {t}: {n}')

    # show first few cycle details
    n_show = min(8, len(cycles))
    print(f'\nShowing first {n_show} cycles samples:')
    for i in range(n_show):
        c = unwrap(cycles[i])
        ctype = getattr(c, 'type', None)
        print(f'\nCycle {i}: type={ctype}')
        data = getattr(c, 'data', None)
        if data is None:
            print('  no data field')
            continue
        data = unwrap(data)
        # list candidate fields
        fields = []
        for fld in ['Time', 'time', 'Voltage_measured', 'Voltage', 'Voltage_charge', 'Current_measured', 'Temperature_measured']:
            val = getattr(data, fld, None)
            if val is not None:
                arr = np.asarray(val)
                fields.append((fld, arr.shape))
        print('  data fields (name, shape):', fields)
        # show small sample of time/voltage if present
        # avoid using `or` with numpy arrays (truth value ambiguous).
        time = getattr(data, 'Time', None)
        if time is None:
            time = getattr(data, 'time', None)

        volt = getattr(data, 'Voltage_measured', None)
        if volt is None:
            volt = getattr(data, 'Voltage', None)
        if volt is None:
            volt = getattr(data, 'Voltage_charge', None)

        if time is not None and volt is not None:
            t = np.asarray(time).ravel()
            v = np.asarray(volt).ravel()
            n = min(5, t.size, v.size)
            print('  sample time:', t[:n])
            print('  sample voltage:', v[:n])
        else:
            print('  time/voltage not both present')


def main():
    p = argparse.ArgumentParser(description='Read and inspect B0005.mat')
    p.add_argument('--matpath', type=str, default='Submissions/richardzang/topic1/week2/B0005.mat')
    args = p.parse_args()

    mat_path = Path(args.matpath)
    if not mat_path.exists():
        print('MAT file not found:', mat_path)
        return

    print('Attempting to load with scipy...')
    try:
        mat = try_scipy_load(mat_path)
        print('Loaded with scipy')
        inspect_b0005(mat)
        return
    except Exception as e:
        print('scipy load failed:', e)

    print('Attempting to load with h5py...')
    try:
        summary = try_h5py_load(mat_path)
        print('h5py summary:')
        pprint.pprint(summary)
    except Exception as e:
        print('h5py load failed:', e)


if __name__ == '__main__':
    main()
