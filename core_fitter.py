import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Users/nmoran/Downloads/ForNick/DATA/PWAVE_LOGGER.csv', low_memory=False)


core_section = df[(df['site'] == 'U1606') & (df['hole'] == 'B') & (df['core'] == 1.0) & (df['section'] == '1')].copy()
core_section['compressional_velocity(m/s)'] = core_section['compressional_velocity(m/s)'].astype(float)
#core_section = core_section[core_section['compressional_velocity(m/s)'] < 5000]

a,b = np.polyfit( core_section["depth(mbsf)"], core_section["compressional_velocity(m/s)"], 1)
core_section['fit'] = a * core_section["depth(mbsf)"] + b

distances = np.abs(a * core_section["depth(mbsf)"] - core_section["compressional_velocity(m/s)"] + b) / np.sqrt(a**2 + 1)

r2 = np.corrcoef(core_section['compressional_velocity(m/s)'], a*core_section['depth(mbsf)'])[0,1] ** 2

print(distances)
print(f'Mean {np.mean(distances)}')
print(f'Median {np.median(distances)}')
print(f'Standard Deviation {np.std(distances)}')
print(f'Max {np.max(distances)}')
print(f'R^2 {r2}')
print(f'Inlier Fraction {np.mean(distances < 0.1)}')


sns.scatterplot(data=core_section, y="compressional_velocity(m/s)", x="depth(mbsf)")
#sns.regplot(data=core_section, y="compressional_velocity(m/s)", x="depth(mbsf)")
sns.lineplot(data=core_section.sort_values("depth(mbsf)"), x="depth(mbsf)", y="fit")
plt.show()
