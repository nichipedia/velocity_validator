import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into pandas dataframe
df = pd.read_csv('/Users/nmoran/Downloads/ForNick/DATA/PWAVE_LOGGER.csv', low_memory=False)

# Select a site to make a smaller set of plots, I chose 1606, but this could be iterated over also.
site1606 = df[df['site'] == 'U1606'].copy()
# Make sure the compressional_velocity column is of type float. This is for plotting purposes. There is bad data included in this raw stuff
site1606['compressional_velocity(m/s)'] = site1606['compressional_velocity(m/s)'].astype(float)

# Uncomment the below if you want to chop off some of the really bad data. Sort of a heuristic.
#site1606 = site1606[site1606['compressional_velocity(m/s)'] < 9000]


for hole in site1606['hole'].unique():
    core_hole = site1606[site1606['hole'] == hole].copy()
    for core_id in core_hole['core'].unique():
        core = core_hole[core_hole['core'] == core_id]
        for section_id in core['section'].unique():
            section = core[core['section'] == section_id]
            sns.scatterplot(data=section, x="compressional_velocity(m/s)", y="depth(mbsf)")
            plt.show()

# Same as above, but it is alittle more clear.
#holes = site1606['hole'].unique()
#cores = site1606['core'].unique()
#sections = site1606['section'].unique()
#for hole in holes:
#    for core in cores:
#        for section in sections:
#            data = site1606[(site1606['hole'] == hole) & (site1606['core'] == core) & (site1606['section'] == section)]
#            sns.scatterplot(data=data, y="compressional_velocity(m/s)", x="depth(mbsf)")
#            plt.show()
