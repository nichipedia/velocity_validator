import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data_file = '/Users/nmoran/Downloads/merged_velocity_data.csv'

print(f'Reading data from {data_file}')
df = pd.read_csv(data_file, low_memory=False)

print(f'Droping NaNs')
cdf = df.dropna(subset=['compressional_velocity(m/s)', 'depth_m', 'GRA_density(g/cm^3)','magnetic_susceptibility(IU)', 'natural_gamma_ray_total_counts(cps)'])

plt_dir = '/Users/nmoran/Documents/school/velocity_validator/plots'

for leg_id in cdf['leg'].unique():
    leg = cdf[cdf['leg'] == leg_id].copy()
    leg['compressional_velocity(m/s)'] = leg['compressional_velocity(m/s)'].astype(float)
    for site_id in leg['site'].unique():
        site = leg[leg['site'] == site_id]
        for hole_id in site['hole'].unique():
            hole = site[site['hole'] == hole_id]
            for core_id in hole['core'].unique():
                core = hole[hole['core'] == core_id]
                for section_id in core['section'].unique():
                    section = core[core['section'] == section_id]
                    fig, axes = plt.subplots(2, 4, figsize=(20,10))
                    axes = axes.flatten()
                    cols = [('compressional_velocity(m/s)', 'depth_m'), ('GRA_density(g/cm^3)', 'depth_m'),
                            ('magnetic_susceptibility(IU)', 'depth_m'), ('natural_gamma_ray_total_counts(cps)','depth_m'),
                            ('GRA_density(g/cm^3)','magnetic_susceptibility(IU)'), ('GRA_density(g/cm^3)','natural_gamma_ray_total_counts(cps)')]
                    for ax, (x, y) in zip(axes[:6], cols):
                        sns.scatterplot(data=section, x=x, y=y, hue='section', palette='deep', ax=ax)
                        ax.set_title(f'{x} vs {y}')

                    sns.scatterplot(data=hole, x='calcium_carbonate(wt%)', y='depth_m', hue='section', palette='deep', ax=axes[6])
                    axes[6].set_title('calcium_carbonate(wt%) vs depth_m')
                    sns.scatterplot(data=hole, x='thermal_conductivity_mean(W/mK)', y='depth_m', hue='section', palette='deep', ax=axes[7])
                    axes[7].set_title('thermal_conductivity_mean(W/mK) vs depth_m')
                    plt.tight_layout()
                    plot_name = f'{plt_dir}/leg{leg_id}_site{site_id}_hole{hole_id}_core{core_id}_section{section_id}.png'
                    print(f'Saving Plot: {plot_name}')
                    plt.savefig(f'{plot_name}')
                    plt.close()
