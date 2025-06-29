import os
import pandas as pd
from PIL import Image
import re

# Specify the main directory where the experiment folders are located
main_directory = r"C:\Users\scoeyman\Desktop\Jake - Mesh\bigmag thickness 241211"  # Update this to your main directory path
output_data = []

# Walk through the main directory 
for root, dirs, files in os.walk(main_directory):                       # This will go through all subfolders (the 1,2,3,4,5,.... from group #s)
    for file in files:
        if file == "data_consolidated.xlsx":                            # Check for the presence of the data_consolidated.xlsx file
            file_path = os.path.join(root, file)
            parts = os.path.normpath(root).split(os.sep)                # Getting parts of file name to extract the experiment, group, and label from the path
           
            if len(parts) >= 5:
                # Adjust indices based on your directory structure
                # For example... my images are in "C:\Users\scoeyman\Desktop\Jake - Mesh\bigmag thickness 241211\both mag"
                experiment = parts[-4]                  # This gets the "bigmag thickness 241211" part
                group = parts[-3]                       # This gets the "both mag" part
                label = parts[-1]                       # This gets the image name being processed (e.g., "P1Aw1D10")
                match = re.search(r'D(\d)', label)      # Extract the first digit following 'D' in the label using regex
                if match:
                    day = int(match.group(1))           # This gets the first digit after 'D'
                else:
                    day = None                          # If no match is found, set to None 
                    
                df = pd.read_excel(file_path)           # Read the Excel file
                df.columns = df.columns.str.strip()     # Strip whitespace from column names

                # Check for the required columns
                # These are hard coded for three cycles of 0-15 amp testing - 6,7,14,15 are missing because those are the cyclic stretch values (98/99 amp)
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

                    strain_values = df['Strain 1']
                    c1_0_strain = strain_values.iloc[0] if len(strain_values) > 0 else None
                    c1_3_strain = strain_values.iloc[1] if len(strain_values) > 1 else None
                    c1_6_strain = strain_values.iloc[2] if len(strain_values) > 2 else None
                    c1_9_strain = strain_values.iloc[3] if len(strain_values) > 3 else None
                    c1_12_strain = strain_values.iloc[4] if len(strain_values) > 4 else None
                    c1_15_strain = strain_values.iloc[5] if len(strain_values) > 5 else None
                    
                    c2_0_strain = strain_values.iloc[8] if len(strain_values) > 8 else None
                    c2_3_strain = strain_values.iloc[9] if len(strain_values) > 9 else None
                    c2_6_strain = strain_values.iloc[10] if len(strain_values) > 10 else None
                    c2_9_strain = strain_values.iloc[11] if len(strain_values) > 11 else None
                    c2_12_strain = strain_values.iloc[12] if len(strain_values) > 12 else None
                    c2_15_strain = strain_values.iloc[13] if len(strain_values) > 13 else None

                    c3_0_strain = strain_values.iloc[16] if len(strain_values) > 16 else None
                    c3_3_strain = strain_values.iloc[17] if len(strain_values) > 17 else None
                    c3_6_strain = strain_values.iloc[18] if len(strain_values) > 18 else None
                    c3_9_strain = strain_values.iloc[19] if len(strain_values) > 19 else None
                    c3_12_strain = strain_values.iloc[20] if len(strain_values) > 20 else None
                    c3_15_strain = strain_values.iloc[21] if len(strain_values) > 21 else None

                    # Append the collected data to the output list
                    output_data.append({
                        "Experiment": experiment,
                        "Group": group,
                        "Label": label,
                        "Cycle 1, 0 Amp Strain": c1_0_strain,
                        "Cycle 1, 3 Amp Strain": c1_3_strain,
                        "Cycle 1, 6 Amp Strain": c1_6_strain,
                        "Cycle 1, 9 Amp Strain": c1_9_strain,
                        "Cycle 1, 12 Amp Strain": c1_12_strain,
                        "Cycle 1, 15 Amp Strain": c1_15_strain,
                        "Cycle 1, Stiff": stiff1,
                        "Cycle 1, R2": r21,
                        "Cycle 1, Cyclic Strain": cs1,
                        "Cycle 2, 0 Amp Strain": c2_0_strain,
                        "Cycle 2, 3 Amp Strain": c2_3_strain,
                        "Cycle 2, 6 Amp Strain": c2_6_strain,
                        "Cycle 2, 9 Amp Strain": c2_9_strain,
                        "Cycle 2, 12 Amp Strain": c2_12_strain,
                        "Cycle 2, 15 Amp Strain": c2_15_strain,
                        "Cycle 2, Stiff": stiff2,
                        "Cycle 2, R2": r22,
                        "Cycle 2, Cyclic Strain": cs2,
                        "Cycle 3, 0 Amp Strain": c3_0_strain,
                        "Cycle 3, 3 Amp Strain": c3_3_strain,
                        "Cycle 3, 6 Amp Strain": c3_6_strain,
                        "Cycle 3, 9 Amp Strain": c3_9_strain,
                        "Cycle 3, 12 Amp Strain": c3_12_strain,
                        "Cycle 3, 15 Amp Strain": c3_15_strain,
                        "Cycle 3, Stiff": stiff3,
                        "Cycle 3, R2": r23,
                        "Cycle 3, Cyclic Strain": cs3,
                    })
                else:
                    print(f"Not enough rows in {file_path} to get values.")
            else:
                print(f"Required columns not found in {file_path}")
        
        if file == "ss.png":
            file_path = os.path.join(root, file)
            img_fold = os.path.join(main_directory, "plots")        # Set img_fold to main_directory + "plots"
            os.makedirs(img_fold, exist_ok=True)                    # Ensure the plots directory exists
            if file == "ss.png":
                file_path = os.path.join(root, file)
            new_filename = f"{experiment}_{group}_{label}_ss.png"   # Construct the new filename
            save_path = os.path.join(img_fold, new_filename)        # Full path for saving the new image
            # Load and save the image with the new filename
            with Image.open(file_path) as img:
                img.save(save_path)
            print(f"Saved {file_path} as {save_path}")
        
output_df = pd.DataFrame(output_data)                               # Convert the output data to a DataFrame
output_file_path = os.path.join(main_directory, "master.xlsx")      # Save as master.xlsx in the main directory
output_df.to_excel(output_file_path, index=False)                   # Save the results to an Excel file
print(f"Data consolidated and saved to {output_file_path}")