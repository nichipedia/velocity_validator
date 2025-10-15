
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Users/nmoran/Downloads/ForNick/DATA/PWAVE_LOGGER.csv', low_memory=False)



for site_id in df['site'].unique():
    site = df[df['site'] == site_id].copy()
    site['compressional_velocity(m/s)'] = site['compressional_velocity(m/s)'].astype(float)
    means = []
    medians = []
    stds = []
    max_vals = []
    inliers = []
    for hole_id in site['hole'].unique():
        hole = site[site['hole'] == hole_id]
        for core_id in hole['core'].unique():
            core = hole[hole['core'] == core_id]
            for section_id in core['section'].unique():
                section = core[core['section'] == section_id]
                a,b = np.polyfit( section["depth(mbsf)"], section["compressional_velocity(m/s)"], 1)
                distances = np.abs(a * section["depth(mbsf)"] - section["compressional_velocity(m/s)"] + b) / np.sqrt(a**2 + 1)
                r2 = np.corrcoef(section['compressional_velocity(m/s)'], a*section['depth(mbsf)'])[0,1] ** 2
                mean = np.mean(distances)
                median = np.median(distances)
                std = np.std(distances)
                max_val = np.max(distances)
                inlier = np.mean(distances<0.1)
                means.append(mean)
                medians.append(medians)
                stds.append(std)
                max_vals.append(max_val)
                inliers.append(inliers)
    pf = pd.DataFrame({
        "Mean": means,
        "Median": medians,
        "Standard Deviation": stds,
        "Max": max_vals,
        "Inlier Fraction": inliers
    })
    sns.pairplot(pf, diag_kind="hist", corner=False)
    plt.show()
