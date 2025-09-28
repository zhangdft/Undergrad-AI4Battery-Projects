import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据
charge = pd.read_csv('Submissions/richardzang/topic1/week2/B0005_first_charge.csv')
discharge = pd.read_csv('Submissions/richardzang/topic1/week2/B0005_first_discharge.csv')

# 2. 画图
plt.figure(figsize=(8, 4))
plt.plot(charge['Time(s)'], charge['Voltage(V)'], label='Charge', color='green')
plt.plot(discharge['Time(s)'], discharge['Voltage(V)'], label='Discharge', color='red')
plt.title('B0005 Charge vs Discharge Voltage Curve')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()