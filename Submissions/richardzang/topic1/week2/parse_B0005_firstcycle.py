import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path


def unwrap(x):
	"""
	处理 MATLAB 加载结果里的封装对象：
	- 如果是 0-dim 的 numpy ndarray，返回内部元素（item）
	- 如果是单元素数组（size==1），返回该元素
	- 否则原样返回
	这样可以把 scipy.io.loadmat 返回的 mat_struct / 单元素 ndarray 解开，方便后续访问属性。
	"""
	if isinstance(x, np.ndarray):
		if x.shape == ():
			return x.item()
		if x.size == 1:
			return x.flatten()[0]
	return x


def extract_first_charge_discharge(mat_file: Path, out_dir: Path):
	"""
	解析指定的 .mat 文件（期望包含 B0005 这个结构），并找到首次出现的 charge 和 discharge 循环：
	- 读取 B0005.cycle（一个数组，每个元素是一个循环对象）
	- 遍历 cycle，寻找 type 包含 'charge' 或 'discharge' 的第一个循环
	- 从每个循环的 data 中提取 Time 与 Voltage（支持多种字段名），写入 CSV

	输出文件名（写入到 out_dir）：
	- B0005_first_charge.csv
	- B0005_first_discharge.csv
	"""
	# 用 scipy 读取 mat 文件
	mat = scipy.io.loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
	if 'B0005' not in mat:
		raise RuntimeError('B0005 key not found in mat file')

	# 取得 B0005 对象并解包
	entry = unwrap(mat['B0005'])
	# 取得 cycles（通常是 ndarray，内部为 mat_struct）
	cycles = getattr(entry, 'cycle', None)
	cycles = unwrap(cycles)
	if cycles is None:
		raise RuntimeError('No cycles found')

	first_charge = None
	first_discharge = None

	# 遍历每个 cycle，按 type 判断是 charge 还是 discharge
	for i, c in enumerate(cycles):
		c = unwrap(c)  # 解包 mat_struct
		ctype = getattr(c, 'type', None)
		ctype = str(ctype).lower() if ctype is not None else 'unknown'

		# 获取 data 字段（里面通常含 Time, Voltage_measured 等）
		data = getattr(c, 'data', None)
		if data is None:
			continue
		data = unwrap(data)

		# 支持多种 Time 字段名：Time 或 time
		time = getattr(data, 'Time', None)
		if time is None:
			time = getattr(data, 'time', None)

		# 支持多种电压字段名：Voltage_measured, Voltage, Voltage_charge
		volt = getattr(data, 'Voltage_measured', None)
		if volt is None:
			volt = getattr(data, 'Voltage', None)
		if volt is None:
			volt = getattr(data, 'Voltage_charge', None)

		# 如果没找到 time 或 volt，则跳过该 cycle
		if time is None or volt is None:
			continue

		# 转为一维 numpy 数组，便于写入 DataFrame
		time = np.asarray(time).ravel()
		volt = np.asarray(volt).ravel()

		# 记录首个 charge / discharge
		if first_charge is None and 'charge' in ctype:
			first_charge = (i, time, volt)
			print('Found first charge at cycle', i)
		if first_discharge is None and 'discharge' in ctype:
			first_discharge = (i, time, volt)
			print('Found first discharge at cycle', i)

		if first_charge is not None and first_discharge is not None:
			break

	# 把找到的数据写入 CSV
	if first_charge is not None:
		i, t, v = first_charge
		df = pd.DataFrame({'Time(s)': t, 'Voltage(V)': v})
		out = out_dir / 'B0005_first_charge.csv'
		df.to_csv(out, index=False)
		print('Wrote', out)
	else:
		print('No charge cycle found')

	if first_discharge is not None:
		i, t, v = first_discharge
		df = pd.DataFrame({'Time(s)': t, 'Voltage(V)': v})
		out = out_dir / 'B0005_first_discharge.csv'
		df.to_csv(out, index=False)
		print('Wrote', out)
	else:
		print('No discharge cycle found')


if __name__ == '__main__':
	# 脚本被直接运行时的入口：定位当前脚本目录并读取同目录下的 B0005.mat
	base = Path(__file__).resolve().parent
	mat_path = base / 'B0005.mat'
	print('Loading', mat_path)
	extract_first_charge_discharge(mat_path, base)
	print('Done')