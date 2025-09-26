import numpy as np
import pandas as pd

Rate1 = np.arange(1, 5, 1)
Rate2 = np.arange(1, 5, 1)
Ton1 = np.arange(1, 10, 1)
Ton2 = np.arange(1, 10, 1)
Toff1 = np.arange(1, 10, 1)
Toff2 = np.arange(1, 10, 1)

param_grid = np.array(np.meshgrid(Rate1, Rate2, Ton1, Ton2, Toff1, Toff2)).T.reshape(-1, 6)


results = pd.DataFrame(param_grid, columns = ['Rate1', 'Rate2', 'Ton1', 'Ton2', 'Toff1', 'Toff2'])
results['D1'] = param_grid[:, 2]/(param_grid[:, 2] + param_grid[:, 4])
results['D2'] = param_grid[:, 3]/(param_grid[:, 3] + param_grid[:, 5])

delta_t =(np.round(10/(results['Rate1'] * results['D1'])
                            + 30/(results['Rate2'] * results['D2']), 4) == 40) 

results['0.5C-det'] = delta_t
filtered_results = results[results['0.5C-det']]
print(f"Number of policies is {filtered_results.shape[0]}")

columns = ['Rate1', 'Rate2', 'Ton1', 'Ton2', 'Toff1', 'Toff2']

with open("policies_raw.csv", "w", newline='') as g:
    filtered_results[columns].to_csv(g, index=False, encoding="utf-8")
with open("policies_all.csv", "w", newline='') as f:
    filtered_results = filtered_results.sample(frac=1, random_state=filtered_results.shape[0])
    filtered_results[columns].to_csv(f, index=False, encoding="utf-8")





