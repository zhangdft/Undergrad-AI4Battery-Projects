# week3 — B0005 数据处理

本目录包含针对 `B0005.mat` 的读取、分解（提取放电循环）以及电池容量寿命绘图的脚本与输出。

目录结构（本说明针对当前 repo 截止到此脚本的状态）：

- `B0005.mat` — 原始 MATLAB 数据文件。
- `extract_B0005_discharges.py` — 从 `.mat` 中提取每次 `discharge` 循环的 `Time` 与 `Current_measured`，并将每次放电保存为独立 CSV 文件（默认保存到 `B0005_170cycles_Discharge`）。
- `B0005_170cycles_Discharge/` — 存放由 `extract_B0005_discharges.py` 生成的放电 CSV 文件（每个文件为一次放电循环）。
- `plot_B0005_capacity_life.py` — 读取 `B0005_170cycles_Discharge` 中的 CSV 文件，计算每次放电的容量（Ah），并绘制容量-放电次数曲线，输出 PNG 文件（默认 `B0005_capacity_life.png`）以及一个同名的 CSV（包含每个循环的容量）。
- `read_b0005.py` — 轻量的诊断脚本，用于查看 `B0005.mat` 的顶层结构、周期数量及前若干个 cycle 的样例数据。

依赖

- Python 3.8+
- numpy
- scipy (必须，用于读取 `.mat`)
- matplotlib (用于绘图)
- pandas (可选，用于更稳健的 CSV 读写；若未安装脚本会回退到 numpy 方法)

使用示例

1. 提取所有放电循环为 CSV：

```powershell
& <python_executable> extract_B0005_discharges.py
```

2. 计算容量并绘制寿命曲线：

```powershell
& <python_executable> plot_B0005_capacity_life.py
```

3. 查看 `.mat` 文件结构（快速诊断）：

```powershell
& <python_executable> read_b0005.py
```

注意事项

- `plot_B0005_capacity_life.py` 使用的容量积分公式为：

  capacity_Ah = np.trapz(np.abs(Current_A), Time_s) / 3600

  该公式在样本时间不均匀（非等间距）时仍然适用；前提是 `Time` 单位必须为秒，`Current` 单位为安培。

- CSV 中电流值通常为负（放电为负），脚本对电流取绝对值后积分。

- 如果 Time 单位不是秒（例如单位为小时或分钟），需要先在 CSV 中换算为秒，或在脚本运行前对数据做转换。

- 脚本会把按发现顺序编号的放电 CSV 命名为 `B0005_discharge_001.csv` ... 。如果你需要保留原始 cycle 索引，请告知我，我可以修改脚本以将文件名包含原始索引或生成一个索引映射文件（CSV/JSON）。

常见问题

- 为什么看到的容量在 1.x Ah 附近？
  - 这取决于数据里采样的电流与时间区间，当前数据的容量范围看起来在 ~1.8 Ah 左右，为合理值。若需要单位校验或排查异常点，请运行 `read_b0005.py` 查看原始 cycle 的时间与电流样例。

下步建议

- 可生成一个 `capacity_index.csv`，记录每个输出 CSV 对应的原始 cycle 索引与计算得到的容量，便于追溯与绘图标注。
- 如果需要对寿命曲线进行平滑或拟合（例如 LOWESS、移动平均或多项式拟合），我可以在 `plot_B0005_capacity_life.py` 中增加选项并生成平滑曲线。

