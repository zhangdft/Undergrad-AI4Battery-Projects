# Battery Cycle Analysis Tool

A Python tool to analyze battery cycling data from Neware .ndax files. It visualizes:

- Voltage vs. Capacity curves across cycles
- Differential capacity (dQ/dV) analysis
- Discharge capacity fade over cycle life

## Installation

```bash
cd Undergrad-AI4Battery-Projects/Submissions/zhongxiansun
pip install -r requirements.txt
```
## Usage

Place your `.ndax` file (e.g., `FullCell.ndax`) in the project directory and run:
```
python battery_analysis.py
