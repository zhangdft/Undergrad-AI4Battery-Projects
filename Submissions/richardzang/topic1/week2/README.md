# Week2 数据提取与可视化说明

本说明针对 `Submissions/richardzang/topic1/week2` 文件夹内的数据与脚本撰写，包含依赖、数据来源、如何从 `B0005.mat` 中提取电压/时间数据（CSV），以及如何绘制电压-时间曲线的操作步骤。

## 文件概览

- `B0005.mat`：原始 MATLAB 格式的电池测试数据（包含多个 cycle，每个 cycle 有 type/data 等字段）。
 - `parse_b0005_cycle1.py`：自动化脚本，提取 `B0005.mat` 中首次出现的 charge 与 discharge 循环，输出：
  - `B0005_first_charge.csv`
  - `B0005_first_discharge.csv`
- `parse_b0005_cycle1.py`：与 `B0005.py` 等效的脚本（包含详细中文注释），用于学习与调试。
- 其他生成的 CSV（若曾生成）会以 `B0005_cycle_*.csv`、`B0005_all_*.csv` 或 `B0005_real_*` 为名前缀；若需要干净状态可以删除这些文件后重跑脚本。

## 依赖（建议）

建议使用 Conda 或 virtualenv 创建隔离环境。主要 Python 包：

- Python 3.8+（实际环境使用 3.11）
- numpy
- scipy
- pandas
- matplotlib

安装（在 PowerShell 中示例）：

```powershell
# 使用 conda 创建并激活环境（可选）
conda create -n battery python=3.11 -y
conda activate battery

# 安装依赖
pip install numpy scipy pandas matplotlib
```

或者直接用 pip（若已有合适的 Python 环境）：

```powershell
pip install numpy scipy pandas matplotlib
```

## 数据来源与结构说明

- `B0005.mat` 由 MATLAB 导出，使用 `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)` 可加载为 Python 对象。
- 顶层 `B0005` 对象包含属性 `cycle`，它是一个长度为 616 的数组（示例数据），每个元素是一个循环（mat_struct），其常见属性：
  - `type`：循环类型，如 `charge`、`discharge`、`impedance` 等。
  - `data`：包含具体测量数据的子结构，常见字段名：`Time`、`Voltage_measured`、`Current_measured`、`Voltage_charge` 等。

示例：`B0005.cycle[0].type == 'charge'`，其 `data.Time` 与 `data.Voltage_measured` 为同长度的一维数组。

## 如何提取电压与时间（CSV）

仓库中已经提供了脚本 `parse_B0005_firstcycle.py`，其逻辑简述：

1. 读取 `B0005.mat` 并解包 `B0005` 对象。
2. 遍历 `cycle` 数组，寻找 `type` 字段包含 `charge` 与 `discharge` 的第一个循环。
3. 从该循环的 `data` 中提取 `Time`（或 `time`）与 `Voltage_measured`（或 `Voltage` / `Voltage_charge`）字段。
4. 把数据写为 CSV：`B0005_first_charge.csv` 与 `B0005_first_discharge.csv`。

运行（在 PowerShell，切换到本目录）：

```powershell
# 进入 week2 文件夹
cd e:/works/vscode/vscodeprojects/Undergrad-AI4Battery-Projects/Submissions/richardzang/topic1/week2

# 使用当前 Python 运行脚本（或在已激活的 conda 环境中直接 python）
python parse_b0005_cycle1.py
```

脚本运行成功后，目录下会出现：
- `B0005_first_charge.csv`
- `B0005_first_discharge.csv`

CSV 列名为 `Time(s)`（秒）与 `Voltage(V)`（伏特）。

## 绘图

如需绘制电压-时间图，可以使用 `pandas` + `matplotlib` 读取生成的 CSV 并绘图。本 README 不包含示例代码；如果你需要，我可以为你生成并运行绘图脚本以输出 PNG 图像。

## 常见问题与注意事项

- 路径分隔符：在 Python 字符串中尽量使用正斜杠 `/` 或 pathlib 来避免 Windows 反斜杠转义问题。
- 字段名不统一：有些 cycle 的 data 结构可能使用不同字段名（例如 `Voltage_charge`、`Voltage_measured`），脚本已尝试做若干字段名的兼容判断。
- 有些 cycle 可能缺少 `Time` 或 `Voltage`（脚本会跳过这些 cycle）。
- 如果需要提取所有 cycle 的 CSV（而不是仅第一次），可以使用较早版本的脚本（已在历史修改中生成过 `B0005_cycle_*.csv`），或联系我帮你恢复/修改脚本以按需提取。

## 扩展建议（可选）

- 合并多个 cycle：若要把所有 charge 或 discharge 合并为一个大 CSV（含 cycle 索引），可以在脚本中收集所有符合类型的 cycles 并用 `pd.concat(...)` 导出为 `B0005_all_charge.csv` / `B0005_all_discharge.csv`。
- 添加更多量（如电流、温度）到 CSV：脚本目前写入 Time/Voltage 两列，可以很容易添加 `Current`、`Temperature` 等列（若 data 中存在这些字段）。
- 自动化环境：推荐建立 `requirements.txt` 或 conda 环境导出文件以便重现环境。

---
