import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

main_directory = r"C:\Users\scoeyman\Desktop\Jake - Mesh\bigmag thickness 241211\both mag"

# Step 2 (Reversed): Rename subfolders back to original names based on an Excel mapping file
def reverse_rename_subfolders(directory, mapping_file):
    if not os.path.exists(mapping_file):
        print(f"Mapping file {mapping_file} not found in {directory}, skipping renaming.")
        return
    # Read the Excel file to get the mappings
    df = pd.read_excel(mapping_file)
    # Create a dictionary from 'ID' (current names) to 'Group #' (original names)
    reverse_mappings = dict(zip(df['ID'], df['Group #'].astype(str)))
    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            if dir_name in reverse_mappings:
                old_path = os.path.join(root, dir_name)
                new_name = reverse_mappings[dir_name]
                new_path = os.path.join(root, new_name)
                print(f"Renaming {old_path} to {new_path}")
                os.rename(old_path, new_path)

# Main function to walk through all subfolders in main directory
def process_subfolders(main_directory):
    for root, dirs, files in os.walk(main_directory):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)

            print(f"\nProcessing subfolder: {subfolder_path}")
            # Check for jpeg_to_tiff.txt in the subfolder
            mapping_file = os.path.join(subfolder_path, 'jpeg_to_tiff_mapping.xlsx')
            reverse_rename_subfolders(subfolder_path, mapping_file)

# Usage
process_subfolders(main_directory)

