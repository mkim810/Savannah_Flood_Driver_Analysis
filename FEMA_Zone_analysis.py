import pandas as pd
import seaborn as sns

# =============================================================================
# Data Preparation for Seaborn Boxplot
# =============================================================================
# Helper function to create a dataframe for a specific zone
def build_plot_df(data_list, surface_names, zone_name):
    dfs = []
    for data, name in zip(data_list, surface_names):
        # Create a temporary dataframe for each surface type
        temp_df = pd.DataFrame({
            'Importance': data,
            'Surface_Type': name,
            'FEMA_Zone': zone_name
        })
        dfs.append(temp_df)
    return pd.concat(dfs, ignore_index=True)

surface_labels = ['Urban', 'River', 'Ditch', 'Wetland', 'AuxDitch']

# Build DataFrames for both zones and combine them
df_ae = build_plot_df(data_ae_pre, surface_labels, 'Zone AE')
df_x  = build_plot_df(data_x_pre,  surface_labels, 'Zone X')
df_plot = pd.concat([df_ae, df_x], ignore_index=True)

# =============================================================================
# Plotting
# =============================================================================
plt.figure(figsize=(8, 5), dpi=300)

# Create the grouped boxplot
sns.boxplot(
    data=df_plot, 
    x='Surface_Type', 
    y='Importance', 
    hue='FEMA_Zone',
    palette={'Zone AE': 'salmon', 'Zone X': 'lightblue'}, # Match your map colors
    showfliers=False, 
    width=0.6
)

# Formatting
plt.title('Precipitation Importance by FEMA Flood Zone', fontsize=14, fontweight='bold')
plt.xlabel('Surface Type', fontsize=12)
plt.ylabel('Precipitation Importance (Gain)', fontsize=12)
plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=11)
plt.legend(title='FEMA Zone', loc='upper right', fontsize=10)
plt.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 1. Group the Extracted Data into Lists
# =============================================================================
surface_labels = ['Urban', 'River', 'Ditch', 'Wetland', 'AuxDitch']

# Downstream (DN) Data
data_ae_down = [Im_down_ae_urban, Im_down_ae_river, Im_down_ae_Ditch, Im_down_ae_wet, Im_down_ae_aux]
data_x_down  = [Im_down_x_urban, Im_down_x_river, Im_down_x_Ditch, Im_down_x_wet, Im_down_x_aux]

# Precipitation (Pre) Data
data_ae_pre  = [Im_pre_ae_urban, Im_pre_ae_river, Im_pre_ae_Ditch, Im_pre_ae_wet, Im_pre_ae_aux]
data_x_pre   = [Im_pre_x_urban, Im_pre_x_river, Im_pre_x_Ditch, Im_pre_x_wet, Im_pre_x_aux]

# Upstream (UP) Data
data_ae_up   = [Im_up_ae_urban, Im_up_ae_river, Im_up_ae_Ditch, Im_up_ae_wet, Im_up_ae_aux]
data_x_up    = [Im_up_x_urban, Im_up_x_river, Im_up_x_Ditch, Im_up_x_wet, Im_up_x_aux]


# =============================================================================
# 2. Build a Unified DataFrame
# =============================================================================
# Helper function updated to include the Driver Name
def build_plot_df(data_list, surface_names, zone_name, driver_name):
    dfs = []
    for data, name in zip(data_list, surface_names):
        temp_df = pd.DataFrame({
            'Importance': data,
            'Surface_Type': name,
            'FEMA_Zone': zone_name,
            'Driver': driver_name
        })
        dfs.append(temp_df)
    return pd.concat(dfs, ignore_index=True)

# Build DataFrames for each driver and zone
df_down_ae = build_plot_df(data_ae_down, surface_labels, 'Zone AE', 'Downstream')
df_down_x  = build_plot_df(data_x_down,  surface_labels, 'Zone X',  'Downstream')

df_pre_ae  = build_plot_df(data_ae_pre,  surface_labels, 'Zone AE', 'Precipitation')
df_pre_x   = build_plot_df(data_x_pre,   surface_labels, 'Zone X',  'Precipitation')

df_up_ae   = build_plot_df(data_ae_up,   surface_labels, 'Zone AE', 'Upstream')
df_up_x    = build_plot_df(data_x_up,    surface_labels, 'Zone X',  'Upstream')

# Combine EVERYTHING into one master DataFrame
df_plot = pd.concat([df_down_ae, df_down_x, df_pre_ae, df_pre_x, df_up_ae, df_up_x], ignore_index=True)


# =============================================================================
# 3. Plotting the 3-Panel Figure
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(7, 10), dpi=300, sharex=True)
drivers = ['Downstream', 'Precipitation', 'Upstream']
palette_colors = {'Zone AE': 'salmon', 'Zone X': 'lightblue'}

for i, driver in enumerate(drivers):
    ax = axes[i]
    
    # Filter data for the specific driver
    subset = df_plot[df_plot['Driver'] == driver]
    
    # Create the boxplot
    sns.boxplot(
        data=subset, 
        x='Surface_Type', 
        y='Importance', 
        hue='FEMA_Zone',
        palette=palette_colors,
        showfliers=False, 
        width=0.6,
        ax=ax
    )
    
    # Formatting each subplot
    ax.set_title(f'{driver} Importance', fontsize=13, fontweight='bold')
    ax.set_ylabel('Importance (Gain)', fontsize=12)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.tick_params(axis='y', labelsize=11)
    
    # Only keep the x-axis label on the bottom plot
    if i == 2:
        ax.set_xlabel('Surface Type', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
    else:
        ax.set_xlabel('')
    
    # Clean up legends: only put the legend on the top plot to avoid redundancy
    if i == 0:
        ax.legend(title='FEMA Zone', loc='right', fontsize=14, title_fontsize=15)
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.show()
