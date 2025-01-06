import os
import pandas as pd
from PIL import Image
import re

# Specify the main directory where the experiment folders are located
main_directory = r"C:\Users\scoeyman\Desktop\Jake - Mesh\bigmag thickness 241211"  # Update this to your main directory path
output_data = []

# Walk through the main directory
for root, dirs, files in os.walk(main_directory):    
    # Check for the presence of the data_consolidated.xlsx file
    for file in files:
        if file == "data_consolidated.xlsx":
            file_path = os.path.join(root, file)

            # Extract the experiment, group, and label from the path
            parts = os.path.normpath(root).split(os.sep)
           
            if len(parts) >= 5:
                # Adjusting indices based on your directory structure
                experiment = parts[-4]  # This gets the "7/14/23..." part
                group = parts[-3]       # This gets the "control" part
                label = parts[-1]       # This gets the label (e.g., "P1Aw1D10")
                # Extract the experiment, group, and label from the path
                parts = os.path.normpath(root).split(os.sep)
            
                if len(parts) >= 5:
                    # Adjusting indices based on your directory structure
                    experiment = parts[-4]  # This gets the "7/14/23..." part
                    group = parts[-3]       # This gets the "control" part
                    label = parts[-1]       # This gets the label (e.g., "P1Aw1D10")
                    # Extract the first digit following 'D' in the label using regex
                    match = re.search(r'D(\d)', label)  # This looks for the first 'D' followed by a single digit
                    if match:
                        day = int(match.group(1))  # This gets the first digit after 'D'
                    else:
                        day = None  # If no match is found, set to None 

                # Read the Excel file and strip column names
                df = pd.read_excel(file_path)
                df.columns = df.columns.str.strip()  # Strip whitespace from column names

                # Check for the required columns
                if 'Stiffness' in df.columns and 'R2' in df.columns:
                    stiff_values = df['Stiffness'].dropna().iloc[:3]
                    stiff1 = stiff_values.iloc[0] if len(stiff_values) > 0 else None
                    stiff2 = stiff_values.iloc[1] if len(stiff_values) > 1 else None
                    stiff3 = stiff_values.iloc[2] if len(stiff_values) > 2 else None
                    
                    r2_values = df['R2'].dropna().iloc[:3]
                    r21 = r2_values.iloc[0] if len(r2_values) > 0 else None
                    r22 = r2_values.iloc[1] if len(r2_values) > 1 else None
                    r23 = r2_values.iloc[2] if len(r2_values) > 2 else None
                    
                    cs_values = df['Cyclic Strain'].dropna().iloc[:3]
                    cs1 = cs_values.iloc[0] if len(cs_values) > 0 else None
                    cs2 = cs_values.iloc[1] if len(cs_values) > 1 else None
                    cs3 = cs_values.iloc[2] if len(cs_values) > 2 else None

                    # Append the collected data to the output list
                    output_data.append({
                        "Experiment": experiment,
                        "Group": group,
                        "Label": label,
                        "Cycle 1 Stiff": stiff1,
                        "Cycle 1 R2": r21,
                        "Cycle 1 Cyclic Strain": cs1,
                        "Cycle 2 Stiff": stiff2,
                        "Cycle 2 R2": r22,
                        "Cycle 2 Cyclic Strain": cs2,
                        "Cycle 3 Stiff": stiff3,
                        "Cycle 3 R2": r23,
                        "Cycle 3 Cyclic Strain": cs3,
                    })
                else:
                    print(f"Not enough rows in {file_path} to get values.")
            else:
                print(f"Required columns not found in {file_path}")
        
        if file == "ss.png":
            file_path = os.path.join(root, file)
            # Set img_fold to main_directory + "plots"
            img_fold = os.path.join(main_directory, "plots")
            # Ensure the plots directory exists
            os.makedirs(img_fold, exist_ok=True)
            if file == "ss.png":
                file_path = os.path.join(root, file)
            # Extract the experiment, group, and label from the path
            parts = os.path.normpath(root).split(os.sep)
            if len(parts) >= 5:
                # Adjusting indices based on directory structure
                experiment = parts[-4]  # This gets the "7/14/23..." part
                group = parts[-3]       # This gets the "control" part
                label = parts[-1]       # This gets the label (e.g., "P1Aw1D10")
                # Construct the new filename
                new_filename = f"{experiment}_{group}_{label}_ss.png"
                # Full path for saving the new image
                save_path = os.path.join(img_fold, new_filename)
                # Load and save the image with the new filename
                with Image.open(file_path) as img:
                    img.save(save_path)
                print(f"Saved {file_path} as {save_path}")
        
# Convert the output data to a DataFrame
output_df = pd.DataFrame(output_data)

# Define the output path for the consolidated results
output_file_path = os.path.join(main_directory, "master.xlsx")  # Save as master.xlsx in the main directory

# Save the results to an Excel file
output_df.to_excel(output_file_path, index=False)

print(f"Data consolidated and saved to {output_file_path}")