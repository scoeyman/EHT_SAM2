import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Usage
main_directory = r"C:\Users\scoeyman\Desktop\Jake - Mesh\bigmag thickness 241211\single mag"

# Step 1: Remove TIFF files in a given directory and rename "jpeg_frames" subfolders to "processed"
def remove_tiff_files(directory):
    for root, dirs, files in os.walk(directory):
        
        # Rename any subfolders called "jpeg_frames" to "processed"
        for dir_name in dirs:
            if dir_name == "jpeg_frames":
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, "processed")
                print(f"Renaming folder {old_path} to {new_path}")
                os.rename(old_path, new_path)
        
        # Remove TIFF files in the current directory
        for file in files:
            if file.endswith('.tiff') or file.endswith('.tif'):
                file_path = os.path.join(root, file)
                print(f"Removing: {file_path}")
                os.remove(file_path)

# Step 2: Rename subfolders based on an Excel mapping file
def rename_subfolders(directory, mapping_file):
    if not os.path.exists(mapping_file):
        print(f"Mapping file {mapping_file} not found in {directory}, skipping renaming.")
        return
    # Read the Excel file to get the mappings
    df = pd.read_excel(mapping_file)
    # Create a dictionary from 'Group #' (the folder numbers) to 'ID' (new names)
    mappings = dict(zip(df['Group #'].astype(str), df['ID']))
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            if dir_name.isdigit() and dir_name in mappings:
                old_path = os.path.join(root, dir_name)
                new_name = mappings[dir_name]
                new_path = os.path.join(root, new_name)
                print(f"Renaming {old_path} to {new_path}")
                os.rename(old_path, new_path)

# Step 3: Plot data from an Excel file
def plot_data_consolidated(file_path, thickness):
    if not os.path.exists(file_path):
        print(f"Data file {file_path} not found, skipping plotting.")
        return

    # Read the Excel file
    data = pd.read_excel(file_path)
    # Fill NaN values with 0
    data = data.fillna(0)
    data = data.replace([np.inf, -np.inf], 0)  # Replace inf and -inf with 0
    data['Stress'] = np.nan
    data['Cyclic Strain'] = np.nan
    data['Stiffness'] = np.nan
    data['R2'] = np.nan

    # Assuming `data` is a pandas DataFrame and 'thickness' is already defined
    data['Stress'] = data['Amps'] / (data['Mid Width'] * thickness)

    # Identify cycle start (`0`) and end (`99`) indices
    cycle_starts = data.index[data['Amps'] == 0].tolist()
    cycle_ends = data.index[data['Amps'] == 99].tolist()

    # Ensure cycles are valid
    if len(cycle_starts) != len(cycle_ends) or len(cycle_starts) == 0:
        print(f"Mismatch between cycle starts ({len(cycle_starts)}) and ends ({len(cycle_ends)}). Skipping plotting.")
        return
    
    # Lists to store the results    
    cycle_results = []

    # Create a figure dynamically with 3 subplots per cycle
    rows, cols = len(cycle_starts), 3
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust size for better visibility

    # If only one cycle, axs might not be an array
    if len(cycle_starts) == 1:
        axs = [axs]

    # Iterate over detected cycles and plot
    for i in range(len(cycle_starts)):
        start_idx = cycle_starts[i]
        end_idx = cycle_ends[i]
        cycle_data = data.iloc[start_idx:end_idx - 1]  # Exclude the last two rows for the main plot
        special_data = data.iloc[end_idx - 1:end_idx+1]  # Rows with 98 and 99

        # Replace negative strains with NaN
        cycle_data.loc[cycle_data['Strain 1'] < 0, 'Strain 1'] = np.nan
        special_data.loc[special_data['Strain 1'] < 0, 'Strain 1'] = np.nan

        # Drop NaN values from 'Strain 1' and 'Stress' columns for linear regression
        cycle_data_clean = cycle_data.dropna(subset=['Strain 1', 'Stress'])
        
        # Main Plot: 0 to number before 98
        axs[i][0].plot(range(len(cycle_data)), cycle_data['Strain 1'], 'b*--', label='Strain')
        axs[i][0].set_title(f'Cycle {i + 1} Main')
        axs[i][0].set_xticks(range(len(cycle_data)))  # Use index range for x-ticks
        axs[i][0].set_xticklabels(cycle_data['Amps'].tolist(), rotation=45)  # Use Amps values for labels
        axs[i][0].set_xlim(0, len(cycle_data) - 1)  # Set x-axis limits
        axs[i][0].legend()

        # Special Points Plot: 98 and 99
        axs[i][1].plot(range(len(special_data)), special_data['Strain 1'], 'ro--', label='Strain')
        axs[i][1].set_title(f'Cycle {i + 1} Cyclic')
        axs[i][1].set_xticks(range(len(special_data)))  # Use index range for x-ticks
        axs[i][1].set_xticklabels(special_data['Amps'].tolist())  # Use Amps values for labels
        axs[i][1].set_xlim(0, len(special_data) - 1)  # Set x-axis limits
        axs[i][1].legend()

        # Stress/Strain Plot with Regression Line
        if len(cycle_data_clean) > 1:  # Ensure enough data points remain after dropping NaN 
            x_data = cycle_data_clean['Strain 1'].values.reshape(-1, 1)  # Reshape for the model
            y_data = cycle_data_clean['Stress'].values

            # Create a boolean mask for finite values in both x_data and y_data
            finite_mask = np.isfinite(x_data.flatten()) & np.isfinite(y_data)

            # Apply the mask to both x_data and y_data
            x_data = x_data[finite_mask].reshape(-1, 1)  # Reshape back after filtering
            y_data = y_data[finite_mask]

            # Perform linear regression
            model = LinearRegression()
            model.fit(x_data, y_data)
            y_pred = model.predict(x_data)
            
            # Get the slope (stiffness) and R² value
            slope = model.coef_[0]
            r2 = model.score(x_data, y_data)  # R² value

            # Add regression line to the plot
            axs[i][2].scatter(x_data, y_data, color='green', marker='*', label='Data')
            axs[i][2].plot(x_data, y_pred, 'g--', label=f'R²={r2:.2f}\nStiff={slope:.2f}')
            axs[i][2].set_title(f'Cycle {i + 1}: Stress-Strain Stiffness')
            axs[i][2].set_xlabel('Strain')
            axs[i][2].set_ylabel('Stress')
            axs[i][2].grid(True)
            axs[i][2].legend()

            # Store the results: 98-99 strain, stiffness (slope), and r2 value
            cyclic_strain = special_data['Strain 1'].iloc[1]  # 99th value (second row of special_data)
            cycle_results.append({
                'Cycle': i + 1,
                'Cyclic Strain': cyclic_strain,  # 99 value
                'Stiffness': slope,  # Stiffness is the slope of the regression line
                'R2': r2  # R² value
            })

            # Add values to the original data for the special row (end_idx - 1 for the special data)
            data.loc[end_idx - 1, 'Cycle'] = i + 1
            data.loc[end_idx - 1, 'Cyclic Strain'] = cyclic_strain
            data.loc[end_idx - 1, 'Stiffness'] = slope
            data.loc[end_idx - 1, 'R2'] = r2
        else:
            print(f"Not enough valid data points for cycle {i + 1} to perform regression.")

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(cycle_results)
    print(results_df)

    # Save the updated DataFrame
    data.to_excel(file_path, index=False)
    print(f"Updated data saved to {file_path}")

    # Adjust layout to reduce overlapping
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Add more space between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Ensure everything fits within the figure
    
    # Save the figure
    subfolder_path = os.path.dirname(file_path)
    output_file_path = os.path.join(subfolder_path, 'ss.png')
    plt.savefig(output_file_path, dpi=300)
    print(f"Plot saved as {output_file_path}")
        
            

def plot_regression(ax, x_data, y_data, color, label):
    if len(x_data) > 0:
        x_reshaped = x_data.reshape(-1, 1)
        model = LinearRegression()  
        model.fit(x_reshaped, y_data)
        y_pred = model.predict(x_reshaped)  # output this... 
        r2 = r2_score(y_data, y_pred)
        slope = model.coef_[0]

        # Plot the best-fit line with R² and Stiffness in the label
        ax.plot(x_data, y_pred, f'{color}--', label=f'R²={r2:.2f}\nStiff={slope:.2f}')

        # Create the legend
        legend = ax.legend(loc='upper right', fontsize='small')
        # Return R² and slope values
        return r2, slope, x_data, y_pred  # Return the R² and slope values

# Main function to walk through all subfolders in main directory
def process_subfolders(main_directory):
    for root, dirs, files in os.walk(main_directory):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)

            print(f"\nProcessing subfolder: {subfolder_path}")

            # Remove TIFF files in this subfolder
            remove_tiff_files(subfolder_path)

            # Check for jpeg_to_tiff.txt in the subfolder
            mapping_file = os.path.join(subfolder_path, 'jpeg_to_tiff_mapping.xlsx')
            rename_subfolders(subfolder_path, mapping_file)

            # Check for data_consolidated.csv in the subfolder and plot
            data_consolidated_file = os.path.join(subfolder_path, 'data_consolidated.xlsx')
            plot_data_consolidated(data_consolidated_file, thickness = 1)

# Usage
process_subfolders(main_directory)
