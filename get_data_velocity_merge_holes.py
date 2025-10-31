import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import re

def save_custom_csv(df, file_path):
    """
    Save the DataFrame as a CSV file with a custom header where each line is commented out by '#'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing 'latitude', 'longitude', 'depth', and 'density' columns.
    file_path (str): The path where the CSV file will be saved.
    """
    # Custom header lines
    header_lines = [
        "# The columns are latitude longitude depth (mbsf)  and velocity (m/s)\n"
        "# These data are from scientific drilling\n"
    ]
    
    # Open file for writing
    with open(file_path, 'w') as f:
        # Write the header lines with '#' at the beginning
        for line in header_lines:
            f.write(line + '\n')
        
        # Add a blank line after the header
        f.write('\n')
        # Write the column names
        f.write('latitude(dd),longitude(dd),depth(mbsf),velocity(m/s)\n')
        # Write the DataFrame without the column names (header=False) and without the index
        df[['latitude(dd)', 'longitude(dd)', 'depth(mbsf)', 'velocity(m/s)']].to_csv(f, index=False, header=False)
    f.close()
    return

def is_point_in_polygon(lat, lon, polygon):
    """
    Determine if a point (lat, lon) is inside a polygon
    polygon: list of (lon, lat) tuples, defining the vertices of the polygon
    """
    num_vertices = len(polygon)
    inside = False
    
    x, y = lon, lat
    
    # Loop through each edge of the polygon
    for i in range(num_vertices):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % num_vertices]  # Next vertex, wrapping around

        # Check if the point is on the same horizontal line as the edge, and if it's within the edge's x bounds
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside

    return inside

def rolling_average_with_depth(df, density_column, max_depth_diff=50, threshold=2):
    """
    Detect outliers in the specified density column of the DataFrame using a depth-based rolling average method.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        density_column (str): The name of the density column to check for outliers.
        max_depth_diff (float): The maximum depth difference to consider for the rolling average.
        threshold (float): The number of standard deviations to define an outlier.

    Returns:
        pd.Series: A boolean Series indicating the presence of outliers.
    """
    # Sort DataFrame by depth
    df_sorted = df.sort_values(by='depth(mbsf)').reset_index(drop=True)

    # Initialize lists to store rolling mean and std
    rolling_mean = []
    rolling_std = []

    # Calculate rolling mean and std based on depth differences
    for i in range(len(df_sorted)):
        # Get the current depth
        current_depth = df_sorted['depth(mbsf)'].iloc[i]

        # Define the depth window
        depth_window = df_sorted[(df_sorted['depth(mbsf)'] >= current_depth - max_depth_diff) &
                                 (df_sorted['depth(mbsf)'] <= current_depth + max_depth_diff)]
        
        # Calculate mean and std if there are samples in the window
        if len(depth_window) > 0:
            mean = depth_window[density_column].mean()
            std = depth_window[density_column].std()
        else:
            mean = np.nan
            std = np.nan
        
        rolling_mean.append(mean)
        rolling_std.append(std)

    # Convert to Series for easier handling
    rolling_mean = pd.Series(rolling_mean, index=df_sorted.index)
    rolling_std = pd.Series(rolling_std, index=df_sorted.index)

    # Determine upper and lower bounds for outliers
    upper_bound = rolling_mean + (threshold * rolling_std)
    lower_bound = rolling_mean - (threshold * rolling_std)

    # Create a boolean Series indicating where outliers are present
    outliers = (df_sorted[density_column] > upper_bound) | (df_sorted[density_column] < lower_bound)

    return outliers

def plot_velocity_vs_depth(df, basename):
    """
    Create a plot of velocity vs depth, with outliers in red and non-outliers in blue, and save it as a PNG file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        basename (str): The basename for the output PNG file.
    """
    # Extract the first latitude and longitude for the title
    first_latitude = df['latitude(dd)'].iloc[0]
    first_longitude = df['longitude(dd)'].iloc[0]
    title = f"Velocity vs Depth\nLatitude: {first_latitude}, Longitude: {first_longitude}"

    # Create a plot
    plt.figure(figsize=(8, 12))

    # Plot non-outliers (False in blue)
    plt.scatter(df[df['outlier'] == False]['velocity(m/s)'],
                df[df['outlier'] == False]['depth(mbsf)'], 
                color='blue', label='Non-outliers', marker='o')

    plt.plot(df['upper_bound'], df['depth(mbsf)'], linestyle='dashed',color='red')
    plt.plot(df['lower_bound'], df['depth(mbsf)'], linestyle='dashed',color='red')

    # Plot outliers (True in red)
    plt.scatter(df[df['outlier'] == True]['velocity(m/s)'],
                df[df['outlier'] == True]['depth(mbsf)'], 
                color='red', label='Outliers', marker='o')

    # Labeling and formatting
    plt.ylabel('Depth (m)')
    plt.xlabel('Velocity (m/s)')
    plt.title(title)
    plt.grid()
    plt.ylim(0, 1800)  # Add some padding
    plt.xlim(1200, 5000)  # Add some padding

    plt.gca().invert_yaxis()  # Invert y-axis
    # Add a legend
    plt.legend()

    # Define the output file name
    output_filename = f"{basename}.png"

    # Save the plot as a PNG file
    plt.savefig(output_filename)
    plt.close()  # Close the plot to free memory

    print(f"Plot saved as {output_filename}")

def detect_outliers_with_rolling_std(df, window_size=3):
    """
    Detect outliers in the density values using rolling standard deviation.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'depth' and 'density' columns.
        window_size (int): The size of the rolling window for calculating standard deviation.
    
    Returns:
        pd.DataFrame: Updated DataFrame with rolling mean, rolling std, 
                      upper and lower bounds, and outlier flags.
    """
    # Calculate the rolling mean and standard deviation for density
    df['rolling_mean'] = df['velocity(m/s)'].rolling(window=window_size, min_periods=1).mean()
    df['rolling_std'] = df['velocity(m/s)'].rolling(window=window_size, min_periods=1).std()

    # Fill NaN values in rolling std with 0 (to handle initial values)
    df['rolling_std'] = df['rolling_std'].fillna(0)

    # Define the upper and lower bounds (±1 standard deviation)
    df['upper_bound'] = df['rolling_mean'] + df['rolling_std']
    df['lower_bound'] = df['rolling_mean'] - df['rolling_std']

    # Detect outliers: True if density is outside the bounds
    df['outlier'] = (df['velocity(m/s)'] < df['lower_bound']) | (df['velocity(m/s)'] > df['upper_bound'])

    return df

def clean_dataframe(df):
    """
    Detect non-numeric columns, convert numeric values to numeric type,
    and delete rows that contain non-numeric values.

    Parameters:
        df (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame with numeric values only.
    """
    # Detect non-numeric columns
    non_numeric_columns = []
    
    for column in df.columns:
        # Attempt to convert the column to numeric
        if pd.to_numeric(df[column], errors='coerce').isna().any():
            non_numeric_columns.append(column)

    # Convert all numeric columns to numeric type
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows that contain NaN values in any non-numeric columns
    df_cleaned = df.dropna(subset=non_numeric_columns)

    # Return the cleaned DataFrame
    return df_cleaned


# Define the polygon as a list of (longitude, latitude) tuples
polygon_coords = [
    (XX, YY),  
    (XX, YY),  
    (XX, YY),  
    (XX, YY),
    (XX, YY),
    (XX, YY),
    (XX, YY),
    (XX, YY)
] # FILL THESE IN WITH LONGITUDES, LATITUDES 

param = 'PWaveVel_AllSciDrilling'
dsdp = '/DSDP/*' + 'sonic' + '*/*.csv'
# odp = '/ODP/*' + 'Velocity' + '*/*.csv'
# iodp = '/IODP/*' + 'Velocity' + '*/*.csv'
files = []
files.append(glob.glob(dsdp))
# files.append(glob.glob(odp))
# files.append(glob.glob(iodp))
files = [x for xs in files for x in xs]
count = 0
files_you_need = []
cols = []

# Create a defaultdict to hold lists of files grouped by 'Site_XXX'
files_grouped_by_site = defaultdict(list)

# Pattern to match 'Site_XXX' (case-insensitive), allowing for variations in delimiters
site_pattern = re.compile(r'Site\d+', re.IGNORECASE)

# Loop through all files in the directory
for file in files:
    if file.endswith(".csv"):  # Check if it's a CSV file
        # Search for 'Site_XXX' pattern in the filename
        match = site_pattern.search(file)
        if match:
            site_identifier = match.group()  # Extract the matched 'Site_XXX'
            files_grouped_by_site[site_identifier].append(file)

directory = '/concatenate_holes_together_files_velocity_prefilter_prework/'
# Now loop through the grouped files and concatenate them
for site, files in files_grouped_by_site.items():
    # Create a list to store dataframes
    df_list = []
    
    # Read and concatenate each file for the current 'Site_XXX'
    for file in files:
        df = pd.read_csv(file, sep=',', comment='#')
        df_list.append(df)
    
    # Concatenate all DataFrames for this site
    concatenated_df = pd.concat(df_list)
    
    # Save the concatenated DataFrame or process it further
    concatenated_df.to_csv(f'{directory}/concatenated_{site}.csv', index=False)

    print(f'Files for {site} have been concatenated and saved.')

files = glob.glob('/concatenate_holes_together_files_velocity_prefilter_prework/*.csv')
for f in range(len(files)):
    df = pd.read_csv(files[f], sep=',', comment='#')
    latitude = df.filter(like='lat')
    longitude = df.filter(like='lon')
    if len(latitude) > 0:
        lat = latitude.iloc[0].values[0]
        lon = longitude.iloc[0].values[0]
        resultBound = is_point_in_polygon(lat, lon, polygon_coords)
        if resultBound == True:
            print(files[f])
            cols.append(df.columns)
            depths = df.filter(like='depth')
            if len(depths.columns) > 3: # DSDP
                depths = depths['depth(mbsf)']
            elif len(depths.columns) == 1: # ODP
                depths = depths['depth(mbsf)']
            elif len(depths.columns) > 1:
                depths = depths['depth_csf-B(mbsf)']
            else:
                raise('Issue with depths, too many depths declared.')
            # print(depths.name)
            velocity = df.filter(like='velocity')
            if len(velocity.columns) == 1:
                if 'sonic_velocity(m/s)' in velocity.columns:
                    velocity = velocity['sonic_velocity(m/s)'] # DSDP
                elif 'p_wave_velocity(m/s)' in velocity.columns:
                    velocity = velocity['p_wave_velocity(m/s)'] # ODP
                elif 'p_wave_velocity_xy(m/s)' in velocity.columns:
                    velocity = velocity['p_wave_velocity_xy(m/s)'] # IODP
            else:
                raise('Issue with velocity, too many velocity declared.')
            print(velocity.name)
            # Sample DataFrame
            one_dataset = {'longitude(dd)':np.ones_like(depths)*lon,
                           'latitude(dd)':np.ones_like(depths)*lat,
                           'depth(mbsf)': depths.values,
                           'velocity(m/s)': velocity.values}
            one_dataset = pd.DataFrame(one_dataset)
            one_dataset = one_dataset.dropna()
            if len(one_dataset) == 0:
                print("The DataFrame is empty after dropping NaN values: ", files[f])
                continue
            else:
                print("Cleaned DataFrame:")
                one_dataset = one_dataset.sort_values(by='depth(mbsf)')
                one_dataset = clean_dataframe(one_dataset) 
                one_dataset = one_dataset[one_dataset['velocity(m/s)'] >= 1100]
                one_dataset = one_dataset[one_dataset['depth(mbsf)'] >= 0.0]
                if (len(one_dataset) >= 10) & ((np.nanmax(one_dataset['depth(mbsf)']) - np.nanmin(one_dataset['depth(mbsf)'])) >= 50):
                    one_dataset_bounded = detect_outliers_with_rolling_std(one_dataset, window_size=10)
                    if len(np.where(one_dataset_bounded['outlier'] == True)[0])/len(one_dataset_bounded) <= 0.4:
                        # outliers = rolling_average_with_depth(one_dataset, 'density(g/cc)', max_depth_diff=50, threshold=2)
                        
                        
                        saveDir = '/Plots_DepthvsVelocity_Merged/'
                        basename = os.path.splitext(os.path.basename(files[f]))[0]
                        plot_velocity_vs_depth(one_dataset_bounded, saveDir+basename)
                        
                        one_dataset_bounded = one_dataset_bounded[one_dataset_bounded['outlier'] != True]
                        # print(one_dataset)
                        final = one_dataset_bounded[['latitude(dd)','longitude(dd)','depth(mbsf)','velocity(m/s)']]

                        saveDir_Files = '/VELOCITY_Merged_'
                        file_path = saveDir_Files + basename + '.csv'
                        save_custom_csv(final, file_path)
                        files_you_need.append(file_path)
            

# Specify the output file name
output_file = 'velocityFiles_merged_DSDP_for_validation.txt'

# Write the list to a text file
with open(output_file, 'w') as f:
    for item in files_you_need:
        f.write(f'"{item}",')  # Write each item followed by a newline

print('break')