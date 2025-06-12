import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from PIL import UnidentifiedImageError
import re
import csv
import cv2
from sam2.build_sam import build_sam2_video_predictor
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
from scipy.ndimage import gaussian_filter1d 
from matplotlib.lines import Line2D

# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# Function for showing predicted mask
def show_mask(mask, ax, obj_id=None, random_color=False):
    cmap = plt.get_cmap("tab10")  # Use a color map for consistent coloring
    color = cmap(obj_id % 10)  # Cycle through the colormap
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

# Function for showing clicked points
def show_points(coords, labels, ax, marker_size=200):
    """Show positive and negative points on the axes."""
    # Convert to numpy array if coords is a list
    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.size == 0:
        return
    # Ensure labels are also a numpy array
    if isinstance(labels, list):
        labels = np.array(labels)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    # Check if pos_points and neg_points are not empty
    if pos_points.size > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if neg_points.size > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# SAM2 initialization
def init_sam2_predictor(checkpoint_path, config_path, device):
    """Initialize SAM2 predictor."""
    return build_sam2_video_predictor(config_path, checkpoint_path, device)

# Function to extract metadata from filename
def get_plate_group_day_well(file_name):
    """Extract Plate, Group, Day, and Well from the filename based on a pattern."""
    # Ensure we only process files with valid image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(file_name)
    if ext.lower() not in image_extensions:
        return None, None, None, None
    # Assume the filename format is P1Aw101x10-00filt -> plate 1, group A, well 1, day 01, cycle 1, amps 0-00
    base_name = os.path.splitext(file_name)[0]
    plate = base_name[0:2] if len(base_name) >= 2 else None  # First 2 characters for Plate
    group = base_name[2] if len(base_name) >= 3 else None    # 3rd character for Group
    day = base_name[5:7] if len(base_name) >= 4 else None      # 6-7th character for Day
    well = base_name[4] if len(base_name) >= 6 else None      # 5th characters for Well
    return plate, group, day, well

# Function to filter files by plate, group, day, and well
def filter_files_by_plate_group_day_well(files, plates=None, groups=None, days=None, wells=None):
    filtered_files = []
    for file in files:
        file_plate, file_group, file_day, file_well = get_plate_group_day_well(file)
        if plates and file_plate not in plates:
            continue
        if groups and file_group not in groups:
            continue
        if days and file_day not in days:
            continue
        if wells and file_well not in wells:
            continue
        filtered_files.append(file)
    return filtered_files

# GUI function to select folder
def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        folder_path.set(folder_selected)
        populate_plate_group_day_well_options(folder_selected)

# GUI function to select results folder
def select_results_folder():
    results_folder = filedialog.askdirectory()
    if results_folder:
        results_folder_var.set(results_folder)

# Function to populate Plate, Group, Day, and Well options
def populate_plate_group_day_well_options(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    plates, groups, days, wells = set(), set(), set(), set()

    for file in files:
        plate, group, day, well = get_plate_group_day_well(file)
        if plate:
            plates.add(plate)
        if group:
            groups.add(group)
        if day:
            days.add(day)
        if well:
            wells.add(well)

    update_checkbuttons(plate_frame, plates, plate_vars)
    update_checkbuttons(group_frame, groups, group_vars)
    update_checkbuttons(day_frame, days, day_vars)
    update_checkbuttons(well_frame, wells, well_vars)

# Function to update Checkbuttons
def update_checkbuttons(frame, items, var_dict):
    for widget in frame.winfo_children():
        widget.destroy()

    for item in sorted(items):
        var = tk.BooleanVar()
        var_dict[item] = var
        tk.Checkbutton(frame, text=item, variable=var).pack(anchor='w')

# Function to select or deselect all Checkbuttons
def select_all(var_dict, select=True):
    for var in var_dict.values():
        var.set(select)

# Function to apply filters and display filtered files
def apply_filters():
    folder = folder_path.get()
    if not folder:
        messagebox.showwarning("Warning", "Please select a folder first.")
        return
    selected_plates = [item for item, var in plate_vars.items() if var.get()]
    selected_groups = [item for item, var in group_vars.items() if var.get()]
    selected_days = [item for item, var in day_vars.items() if var.get()]
    selected_wells = [item for item, var in well_vars.items() if var.get()]
    all_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    filtered_files = filter_files_by_plate_group_day_well(all_files, plates=selected_plates, groups=selected_groups, days=selected_days, wells=selected_wells)
    if filtered_files:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "\n".join(filtered_files))
        global files_to_process
        files_to_process = filtered_files
    else:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "No files match the selected filters.")

# Function to extract imaging cycle and frame
def extract_cycle_and_frame(filename):
    """
    Extract the cycle number (number after 'x') and the combined frame number (e.g., 060, 150).
    Returns a tuple: (cycle_number, frame_number) for sorting purposes.
    """
    # Extract cycle number
    cycle_match = re.search(r"x(\d+)", filename)  # Extracts number after 'x'
    cycle_number = int(cycle_match.group(1)) if cycle_match else None
    
    # Extract frame number (combined padded format)
    frame_match = re.search(r"(\d+)-(\d+)", filename)  # Matches numbers before and after '-'
    if frame_match:
        frame_start = int(frame_match.group(1))  # Number before '-'
        frame_end = int(frame_match.group(2))    # Number after '-'
        # Combine frame_start and frame_end into a single padded frame number
        frame_number = int(f"{frame_start}{frame_end:02d}")  # Ensures frame_end is padded to 2 digits
    else:
        frame_number = None
    
    return (cycle_number, frame_number)

# Function to group image files by prefixes
def group_files_by_main_prefix(files):
    """
    Group files by main prefix up to 'x' (e.g., "P1Aw101").
    """
    grouped_files = {}
    for file in files:
        main_prefix = file.split('x')[0]  # Prefix up to 'x', e.g., "P1Aw101"
        if main_prefix not in grouped_files:
            grouped_files[main_prefix] = []
        grouped_files[main_prefix].append(file)
    return grouped_files

# Function to extract the part of the filename before the first hyphen
def extract_name_before_hyphen(filename):
    return filename.split('x')[0] if '-' in filename else filename

# Process the TIFF files, convert them to JPEG, and organize them by group    - might need to update flipping of image
def process_images():
    """Process the filtered images with SAM2 and save them in group directories with appropriate naming."""
    global files_to_process, device, jpeg_to_tiff_mapping, excel_file_path, cycle
    if not files_to_process:
        messagebox.showwarning("Warning", "No files to process. Please apply filters first.")
        return
    # Select folder for images
    img_dir = folder_path.get()
    if not img_dir:
        messagebox.showwarning("Warning", "Please select an image folder.")
        return
    # Define the name of the JPEG directory
    jpeg_frames = "jpeg_frames"
    jpeg_dir = os.path.join(img_dir, jpeg_frames)
    os.makedirs(jpeg_dir, exist_ok=True)  # Create JPEG frames directory if it doesn't exist
    # Path for the shared Excel file to save results
    excel_file_path = os.path.join(img_dir, 'processed_results.xlsx')
    # Initialize a set to store unique mappings for the CSV
    unique_mappings = set()
    jpeg_to_tiff_mapping = {}  # This will map JPEG filenames to their original TIFF filenames
    # Group files by their prefixes
    grouped_files = group_files_by_main_prefix(files_to_process)
    group_number = 1  # Start with group 1
    for prefix, group_files in grouped_files.items():
        print(f"Processing group: {prefix}, Group Number: {group_number}")
        # Sort files by the numeric part after the hyphen
        sorted_group_files = sorted(group_files, key=extract_cycle_and_frame)
        # Create subdirectory for each group (e.g., "1", "2", etc.)
        group_dir = os.path.join(jpeg_dir, str(group_number))
        os.makedirs(group_dir, exist_ok=True)
        # Loop through files in the directory to remove data excel sheet from previous runs..
        for file in os.listdir(group_dir):
            # Full file path
            file_path = os.path.join(group_dir, file)
            # Check if it's an Excel file
            if file.endswith(".xlsx") or file.endswith(".xls"):
                os.remove(file_path)  # Delete the file
        # Loop through the files within the group
        for idx, file_name in enumerate(sorted_group_files):
            file_path = os.path.join(folder_path.get(), file_name)
            # Try to open and process each TIFF image
            try:
                # Ensure you read TIFF images correctly
                if file_path.lower().endswith('.tiff') or file_path.lower().endswith('.tif'):
                    with Image.open(file_path) as img:
                        img_rgb = img.convert('RGB')



                    # #### FOR THE THICKNESS TESTING CODE
                    # flipped_img = img_rgb.transpose(Image.FLIP_TOP_BOTTOM)  # Vertical flip if needed
                
                    ### FOR THE BIGMAG CODE
                    # Check if the prefix contains 'B' or 'D'
                    if 'B' in prefix or 'D' in prefix:
                        flipped_img = img_rgb.transpose(Image.FLIP_TOP_BOTTOM)  # Vertical flip if needed
                    else:
                        flipped_img = img_rgb  # No flip required




                    # Calculate the numeric part for the new JPEG name
                    cycle, frame = extract_cycle_and_frame(file_name)
                    # Create JPEG filename with the group number at the beginning, e.g., "10000.jpg", "10003.jpg"
                    jpeg_filename = f"{group_number:01d}{cycle:02d}{frame:03d}.jpg"
                    jpeg_path = os.path.join(group_dir, jpeg_filename)
                    # Save the JPEG image with the new name
                    flipped_img.save(jpeg_path)
                    # Extract the base part of the TIFF filename before the hyphen
                    tiff_base_name = extract_name_before_hyphen(file_name)
                    # Add the unique mapping to the set (Group, Cycle, TIFF Base Name)
                    unique_mappings.add((group_number, cycle, tiff_base_name))
                    # Add to the jpeg_to_tiff_mapping for lookup later
                    jpeg_to_tiff_mapping[jpeg_filename] = tiff_base_name
                else:
                    print(f"Warning: {file_name} is not a TIFF file. Skipping.")
            except UnidentifiedImageError:
                print(f"Error: Unable to open image file '{file_name}'. Skipping this file.")
            except Exception as e:
                print(f"Unexpected error with file '{file_name}': {e}. Skipping this file.")
            
        # Save the unique mappings to a CSV file, sorted by group number
        if unique_mappings:
            mapping_file = os.path.join(jpeg_dir, 'jpeg_to_tiff_mapping.xlsx')
            try:
                # Convert set to DataFrame, sort by 'Group', and save as CSV
                df = pd.DataFrame(list(unique_mappings), columns=['Group #', 'Cycle', 'ID'])
                df.sort_values(by=['Group #', 'Cycle'], inplace=True)
                df.to_excel(mapping_file, index=False)
                # print(f"Mapping saved to: {mapping_file}")            
            except Exception as e:
                print(f"Error saving mapping file: {e}")
        else:
            print("No mappings were created, file not saved.")
    
        # Call the SAM2 processing function for the current group
        process_jpeg_group_with_sam2(group_number, group_dir)  
        # Increment the group number for the next set of files
        group_number += 1


    ##### After processing all groups (subdirectories) #####
    # Walk through all directories and subdirectories in img_dir
    for root, dirs, files in os.walk(img_dir):
        # Check if consolidated_data.xlsx exists in the current directory
        if 'data_consolidated.xlsx' in files:
            # Define the path to the Excel file
            consolidated_file_path = os.path.join(root, 'data_consolidated.xlsx')
            # Load the consolidated data Excel file
            try:
                df = pd.read_excel(consolidated_file_path)
            except Exception as e:
                print(f"Error loading the Excel file: {e}")
                continue  # Skip to the next directory
            # Remove duplicates
            df_cleaned = df.drop_duplicates()
            # List of the columns to calculate strains for
            length_columns = ['EHT Length']
            # Iterate over each length column to calculate strains
            for length_column in length_columns:
                # Check if the length column exists
                if length_column in df_cleaned.columns:
                    # Initialize the initial length as None
                    initial_length = None
                    # Iterate over each row to calculate strain dynamically
                    for idx, row in df_cleaned.iterrows():
                        amps = row['Amps']  # Get the value of "Amps" for the current row
                        current_length = row[length_column]  # Get the current length value
                        if amps == 0:  # Reset the initial length when "Amps" is zero
                            initial_length = current_length
                        if amps == 98: 
                            initial_length = current_length
                        # Ensure initial_length is set before calculating strain
                        if initial_length is not None:
                            strain_column = f'Strain {length_columns.index(length_column) + 1}'
                            df_cleaned.at[idx, strain_column] = (current_length - initial_length) / initial_length     
                else:
                    print(f"Column '{length_column}' not found in the DataFrame.")

            # Save the cleaned DataFrame back to the same Excel file
            df_cleaned.to_excel(consolidated_file_path, index=False)
            print(f"Duplicates removed. Cleaned data saved to: {consolidated_file_path}")

            for file in files:
                # Full file path
                file_path = os.path.join(root, file)
                # Check if the file is an image and starts with a number
                if file.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    # Check if the filename starts with a number using a regex pattern
                    if re.match(r'^\d', file):
                        print(f"Deleting image file starting with a number: {file}")
                        os.remove(file_path)  # Delete the file
    plt.close('all')
    print("Processing completed.")

# Function for saving clicks
def save_clicks_to_file(clicks, labels, file_path):
    """Save clicks and labels to a text file, appending if the file already exists."""
    # Open the file in append mode
    with open(file_path, 'a') as f:
        # Write each point and its label
        for point, label in zip(clicks, labels):
            f.write(f"{point[0]},{point[1]},{label}\n")

# Function for loading clicks
def load_clicks_from_file(file_path):
    """Load clicks and labels from a text file."""
    if not os.path.exists(file_path):
        return [], []  # Return empty lists if the file doesn't exist
    clicks = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, label = line.strip().split(',')
            clicks.append([float(x), float(y)])
            labels.append(int(label))
    return clicks, labels

# Function to handle clicks
def on_click(event):
    global clicked_points, clicked_labels, cid
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata
        clicked_points.append([x, y])
        # Determine if it's a positive or negative click based on the current state
        if current_object == "eht":
            if len(clicked_points) <= clicks_per_object_eht_positive:
                clicked_labels.append(1)  # Positive label for clicked points
                print(f"EHT Positive clicked point: {x}, {y}")
            else:
                clicked_labels.append(0)  # Negative label for clicked points
                print(f"EHT Negative clicked point: {x}, {y}")
        elif current_object == "magnet": 
            if len(clicked_points) <= clicks_per_object_magnet_positive:
                clicked_labels.append(1)  # Positive label for clicked points
                print(f"Magnet Positive clicked point: {x}, {y}")
        elif current_object == "grip": 
            if len(clicked_points) <= clicks_per_object_grip_positive:
                clicked_labels.append(1)  # Positive label for clicked points
                print(f"Grip Positive clicked point: {x}, {y}")

        # Redraw the scatter points for both EHT and Piston and Grip
        if current_object == "eht":
            color = 'green' if clicked_labels[-1] == 1 else 'red'
            axs.scatter(x, y, color=color, marker='*', s=200, edgecolor='white', linewidth=1.25)
        elif current_object == "magnet":
            color = 'green' if clicked_labels[-1] == 1 else 'red'
            axs.scatter(x, y, color=color, marker='*', s=200, edgecolor='white', linewidth=1.25)
        elif current_object == "grip":
            color = 'green' if clicked_labels[-1] == 1 else 'red'
            axs.scatter(x, y, color=color, marker='*', s=200, edgecolor='white', linewidth=1.25)
        plt.draw()
        # Check if we have enough clicks for the current object
        if current_object == "eht" and len(clicked_points) == (clicks_per_object_eht_positive + clicks_per_object_eht_negative):
            plt.gcf().canvas.mpl_disconnect(cid)
            plt.close()  # Automatically closes the plot window
        elif current_object == "magnet" and len(clicked_points) == (clicks_per_object_magnet_positive):
            plt.gcf().canvas.mpl_disconnect(cid)
            plt.close()  # Automatically closes the plot window
        elif current_object == "grip" and len(clicked_points) == (clicks_per_object_grip_positive):
            plt.gcf().canvas.mpl_disconnect(cid)
            plt.close()  # Automatically closes the plot window
        
# Function to get EHT dimensions and visualize model predictions
def visualize_predictions(prompts, group_dir, frame_names, predictor, inference_state, ann_frame_idx, group_number):
    """Visualize the points and segmentation masks with outlined contours for specific IDs."""
    global stored_contour_obj_3, consolidated_data_list, dot_y_coord_bottom, dot_y_coord_bottom2, axs, fig, cycle, avg_y    # Global variables shared across functions
    distances_list = []                                                                                                     # Initialize the distances list for exporting lengths and widths
    consolidated_data = []                                                                                                  # Initialize the data matrix for consolidated data
    min_width_length = None                                                                                                 # Initialize to a default value

    jpeg_filename = frame_names[ann_frame_idx]                                                                              # Get the specific JPEG filename for this frame
    jpeg_file_path = os.path.join(group_dir, jpeg_filename)                                                                 # Join jpeg name with group directory
    tiff_file_name = jpeg_to_tiff_mapping.get(jpeg_filename, None)                                                          # Get the TIFF file name from the unique mapping using jpeg_filename
    img = Image.open(jpeg_file_path)                                                                                        # Load the original image
    img_np = np.array(img)                                                                                                  # Convert image to numpy array for OpenCV
    width, height = img.size                                                                                                # Image dimensions
    
    # Processing frame for object visualization 
    largest_contours = {1: None, 2: None}                                                                                   # Dictionary to hold largest contours for ID 1 and 2 and 3
    for obj_id, (points, labels) in prompts.items():                                                                        # Get predicted mask logits
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels)

        for i, out_obj_id in enumerate(out_obj_ids):                                                                        # Processing multiple objects
            if out_obj_id in [1, 2]:                                                                                        # Only process EHT (ID 1) and Magnet (ID 2)
                mask_array = out_mask_logits[i].detach().cpu().numpy()                                                      # Convert to numpy array
                mask_array = (mask_array > 0).astype(np.uint8)                                                              # Create a binary mask
                contours, _ = cv2.findContours(mask_array[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                   # Find contours in the binary mask
                if contours:
                    max_contour = max(contours, key=cv2.contourArea)                                                        # Find the largest contour by area
                    largest_contours[out_obj_id] = max_contour                                                              # Store the largest contour for the specific ID

    # Draw the largest contours 
    for obj_id, contour in largest_contours.items():
        if contour is not None:                                                                                             # Check if a contour was found
            if obj_id == 1:
                color = (0, 255, 0)                                                                                         # Green for ID 1 (eht)
                cv2.drawContours(img_np, [contour], -1, color, thickness=2)                                                 # Outline the largest contour
            if obj_id == 2:
                color = (255, 0, 0)                                                                                         # Blue for ID 2 (magnet)
                cv2.drawContours(img_np, [contour], -1, color, thickness=2)                                                 # Outline the largest contour
    
    # Figure axs
    axs = plt.gca()                     # Get current axis for drawing
    axs.clear()                         # Clear the axes from any previous content
    axs.imshow(img_np[::-1])
    axs.axis('off')
    axs.invert_yaxis()                  # Flip the y-axis
    legend_entries = []
  
    ##### Calculating the Main EHT Dimensions #####
    if largest_contours[1] is not None:                                         # Check if we found a contour for EHT
        contour_obj_1 = np.squeeze(largest_contours[1])                         # Get object 1's contour
        if contour_obj_1.ndim == 1:                                             # Proper shape 
            contour_obj_1 = np.reshape(contour_obj_1, (-1, 2))
        x_coords_obj_1 = contour_obj_1[:, 0]                                    # EHT x coordinates 
        mid_x_obj_1 = (np.min(x_coords_obj_1) + np.max(x_coords_obj_1)) // 2    # Mid-way x-coordinate

        # Check for Object 2 (magnet)
        if largest_contours[2] is not None:                                     # Contour for magnet exists
            contour_obj_2 = np.squeeze(largest_contours[2])                     # Get magnet contour
            contour_area_2 = cv2.contourArea(largest_contours[2])               # Magnet area
            if contour_obj_2.ndim == 1:
                contour_obj_2 = np.reshape(contour_obj_2, (-1, 2))              # Reshaping dimensions
            M = cv2.moments(largest_contours[2])                                # Compute centroid using moments
            if M["m00"] != 0:                                                   # Avoid division by zero
                centroid_x_obj_2 = int(M["m10"] / M["m00"])                     # Centroid x coordinate
                centroid_y_obj_2 = int(M["m01"] / M["m00"])                     # Centroid y coordinate
            else:
                centroid_x_obj_2, centroid_y_obj_2 = 0, 0                       # 0s if no moments
            flipped_centroid_y_obj_2 = height - centroid_y_obj_2                # Flip y-coordinate for the flipped y-axis
        else:
            centroid_x_obj_2, flipped_centroid_y_obj_2 = 0, 0
            print("Object 2 not found.")

        # If both Object 2 (magnet) and Object 3 (stationary pillar) exist, calcualte EHT length 
        if largest_contours[2] is not None:                                     # Magnet
            pillar_y_from_magnet = flipped_centroid_y_obj_2 - 275               # User defined pixels from centroid to piston pillar tip.. MIGHT HAVE TO CHANGE THE 275 HERE based off grip choice (this was for the BIG MAG samples)
            use_y = height - avg_y                                              # Height is image size and avg y comes from the stationary pillars (global variable) -> this gives the stationary grip tip y coorindate
            line_length = np.abs(pillar_y_from_magnet - use_y)                  # Calculate the length of the line   - THIS IS THE MAIN EHT LENGTH MEASUREMENT
        else:
            line_length = 0
        
        # Plotting EHT Length
        axs.plot(centroid_x_obj_2, use_y, 'g*', markersize = 12)                                            # Plotting stationary pillar tip point (length endpoint)
        axs.plot(centroid_x_obj_2, pillar_y_from_magnet, 'g*', markersize = 12)                             # Plotting grip pillar tip point (length endpoint)
        axs.plot([centroid_x_obj_2, centroid_x_obj_2], [pillar_y_from_magnet, use_y], 'g--', linewidth=2)   # Plotting eht length
        legend_entries.append(f"EHT Length: {line_length:.2f}")                                             # Adjust units as needed
            
        ##### Find widths across Object 1 (EHT) #####
        if largest_contours[1] is not None:                 # Ensure EHT contour exists
            y_threshold = 35                                # Define a threshold in pixels for how close y-coordinates should be to the target y-coordinates
            y_coords_near_max = []                          # For storing coordinates
            y_coords_near_min = []                          # For storing coordinates

            # Initialize variables to track maximum widths for top and bottom of EHT
            max_width_top = 0
            max_width_bottom = 0
            max_width_y_coord_top = None
            max_width_y_coord_bottom = None
            min_x_at_max_width_top = None
            max_x_at_max_width_top = None
            min_x_at_max_width_bottom = None
            max_x_at_max_width_bottom = None

            max_y = axs.get_ylim()[1]  # Get the maximum y limit of the current axes

            # Find y-coordinates near piston grip and stationary grip max/min points
            for y in contour_obj_1[:, 1]:
                y_inverted = max_y - y                                      # Inverting y values (python puts 0,0 in top left)
                if abs(y_inverted - pillar_y_from_magnet) < y_threshold:    # Coordinates near piston
                    y_coords_near_max.append(y)
                if abs(y_inverted - use_y) < y_threshold:                   # Coordinates near stationary grip
                    y_coords_near_min.append(y)

            # Combine and deduplicate found y-coordinates
            combined_y_coords = set(y_coords_near_max + y_coords_near_min)

            # Calculate widths and track maximum widths for top and bottom of EHT
            for y_coord in combined_y_coords:
                x_coords_at_y = contour_obj_1[contour_obj_1[:, 1] == y_coord][:, 0]
                if len(x_coords_at_y) > 0:                                              # If there are points at this y-coordinate
                    min_x = np.min(x_coords_at_y)
                    max_x = np.max(x_coords_at_y)
                    width_at_y = max_x - min_x
                    # Track the maximum width for top and bottom regions of EHT
                    if y_coord in y_coords_near_max:                                    # If it's near the top (piston)
                        if width_at_y > max_width_top:
                            max_width_top = width_at_y
                            max_width_y_coord_top = y_coord
                            min_x_at_max_width_top = min_x
                            max_x_at_max_width_top = max_x
                    elif y_coord in y_coords_near_min:                                  # If it's near the bottom (stationary)
                        if width_at_y > max_width_bottom:
                            max_width_bottom = width_at_y
                            max_width_y_coord_bottom = y_coord
                            min_x_at_max_width_bottom = min_x
                            max_x_at_max_width_bottom = max_x
                else:
                    print('No y coords at threshold: {y_threshold}')

        # Plot the maximum widths for top and bottom if they were found
        if max_width_y_coord_top is not None:                                                   # Plotting top width
            axs.plot([min_x_at_max_width_top, max_x_at_max_width_top],
                        [max_y - max_width_y_coord_top, max_y - max_width_y_coord_top], 
                        'g--', linewidth=2, label=f'Max Width Top: {max_width_top}') 
        if max_width_y_coord_bottom is not None:                                                # Plotting bottom width
            axs.plot([min_x_at_max_width_bottom, max_x_at_max_width_bottom],
                        [max_y - max_width_y_coord_bottom, max_y - max_width_y_coord_bottom], 
                        'g--', linewidth=2, label=f'Max Width Bottom: {max_width_bottom}') 
        legend_entries.append(f"Piston Width: {max_width_top:.2f}")                             # Adjust units as needed
        legend_entries.append(f"Stat. Grip Width: {max_width_bottom:.2f}")                      # Adjust units as needed


        ##### GETTING ALL EHT WIDTHS / LENGTHS AT EACH PERCENTAGE OF EHT LENGTH AND WIDTH ##### 
        contour_frame = np.array(img.convert('RGB'), dtype=np.uint8)
        contour = np.squeeze(largest_contours[1])                                                       # Convert the contour to numpy array for easier manipulation
        if contour.ndim == 1:
            contour = np.reshape(contour, (-1, 2))
        x_coords = contour[:, 0]
        y_coords = contour[:, 1]
        max_OG_y = np.max(y_coords)
        y_coords = np.max(y_coords) - y_coords
        new_contour = np.column_stack((x_coords, y_coords))
        smoothed_contour = gaussian_filter1d(new_contour, sigma=0.2, axis=0)                            # Mmoothing tracing (USED FOR MEASUREMENTS... THE NOT SMOOTHED TRACING IS SHOWED IN IMAGES)
        x_values = smoothed_contour[:, 0]
        y_values = smoothed_contour[:, 1]
        theta = np.arctan2(y_values, x_values)
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        distances = np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2)
        cumulative_distances = np.cumsum(np.insert(distances, 0, 0))
        interp_cumulative_distances = np.linspace(0, cumulative_distances[-1], 2000)                    # Could increadse value if want more points... 
        interp_x_values = np.interp(interp_cumulative_distances, cumulative_distances, x_values)
        interp_y_values = np.interp(interp_cumulative_distances, cumulative_distances, y_values)
        sorted_indices = np.argsort(interp_y_values)
        sorted_x_values = np.round(interp_x_values[sorted_indices], 1)
        sorted_y_values = np.round(interp_y_values[sorted_indices], 1)
        sorted_indices_x = np.argsort(interp_x_values)
        sorted_x_values_x = np.round(interp_x_values[sorted_indices_x], 1)
        sorted_y_values_x = np.round(interp_y_values[sorted_indices_x], 1)
        A = np.max(sorted_y_values)
        B = np.min(sorted_y_values)
        max_tissue_length = A - B
        C = np.max(sorted_x_values_x)
        D = np.min(sorted_x_values_x)
        max_tissue_width = C - D
        percentages = np.arange(0, 100, 1)

        ### Finding EHT Length %s that Align with the Width measurements for stationary grip / piston ###
        if max_width_y_coord_top is not None and max_width_y_coord_bottom is not None:                          # Find the corresponding percentages for max_width_y_coord_top and max_width_y_coord_bottom
            percentage_top = np.clip(((max_y - max_width_y_coord_top) / max_tissue_length) * 100, 0, 100)       # Calculate percentage for the maximum width coordinates top and bottom
            percentage_bottom = np.clip(((max_y - max_width_y_coord_bottom) / max_tissue_length) * 100, 0, 100)
            closest_percentage_top = percentages[np.argmin(np.abs(percentages - percentage_top))]               # Find closest percentage in the defined percentages array
            closest_percentage_bottom = percentages[np.argmin(np.abs(percentages - percentage_bottom))]
        else:
            print("Max width coordinates are not available.")
     
        contour_area = cv2.contourArea(largest_contours[1])

        ### Get the ID from the jpeg filename (e.g., "10003.jpg" -> "10003")  ###
        jpeg_id = jpeg_filename.split('.')[0]   # Remove file extension
        amps = int(jpeg_id[-3:])                # Last three digits as amplitude
        cycle = int(jpeg_id[-4])                # 4th to last digit
        print(cycle)

        ### Initialize variables to store minimum width ("mid width" of EHT) and corresponding percentage ###
        min_width_within_range = float('inf')  # Start with infinity to find minimum
        closest_percentage_top = []
        min_width_length = []
        min_x_at_y_min = []
        max_x_at_y_min = []

        for percentage in percentages:                                                  # Go through all length and width measurments at each percentage points of EHT
            length_percentage = (percentage / 100) * max_tissue_length
            length_percentage = np.round(length_percentage, 1)
            width_percentage = (percentage / 100) * max_tissue_width
            width_percentage = np.round(width_percentage, 1)
            width_percentage += np.min(sorted_x_values_x)
            index = np.searchsorted(sorted_y_values, length_percentage, side='left')    # Ensuring points found are sorted properly
            index_x = np.searchsorted(sorted_x_values_x,    width_percentage, side='left')
            indices = [max(0, index - 2), max(0, index - 1), index, min(len(sorted_y_values) - 1, index + 1), min(len(sorted_y_values) - 1, index + 2)]
            indices_x = [max(0, index_x - 2), max(0, index_x - 1), index_x, min(len(sorted_x_values_x) - 1, index_x + 1), min(len(sorted_x_values_x) - 1, index_x + 2)]
            x_values_at_indices = sorted_x_values[indices]
            y_values_at_indices = sorted_y_values_x[indices_x]
            min_x_at_y = min(x_values_at_indices)
            max_x_at_y = max(x_values_at_indices)
            min_y_at_x = min(y_values_at_indices)
            max_y_at_x = max(y_values_at_indices)
            width_at_percentage = max_x_at_y - min_x_at_y     # Width at each percentage
            length_at_percentage = max_y_at_x - min_y_at_x    # Length at each percentage
            shift = height - max_OG_y
           
            distances_list.append(pd.DataFrame({    # Append the length and width data at each percentage to the distances list
                'Image': [tiff_file_name],
                'Amps': [amps],
                'Percentage': [percentage],
                'Length': [length_at_percentage],
                'Width': [width_at_percentage],
                'Area': [contour_area],
            }))
        
            ### MID WIDTH ###
            if 35 <= percentage <= 65:                                  # Check if percentage is within 35% to 65% (narrowing down to center of EHT)
                if 25 < width_at_percentage < min_width_within_range:   # Making sure width is at least 25 pixels wide
                    min_width_within_range = width_at_percentage        # Making sure we find the true min width in the range 
                    min_width_length = length_percentage                # Capture the length percentage where the min width was found
                    min_x_at_y_min = min_x_at_y                         # Store the min x for the minimum width
                    max_x_at_y_min = max_x_at_y                         # Store the max x for the minimum width

        # Check if a valid min_width_length was found
        if min_width_length is None:
            print("No minimum width found in the 35-65 range")
        else:
            axs.plot([min_x_at_y_min, max_x_at_y_min], [min_width_length + shift, min_width_length + shift], 'g--', linewidth=2)    # Plot the minimum width line at the corresponding percentage (min_width_length) 
        legend_entries.append(f"Mid Width: {min_width_within_range:.2f}")                                                           # Append a text-only entry for the legend - adjust units as needed
        legend = axs.legend(legend_entries, loc='upper left', bbox_to_anchor=(1, 0.25), borderaxespad=0.)                           # Create the legend and store it in a variable
        for legend_line in legend.get_lines():
            legend_line.set_visible(False)                                                                                          # Hide the symbols in the legend

        ##### SAVING TO EXCEL #####
        distances_df = pd.concat(distances_list, ignore_index=True)                                                     # Convert the distances list to a single DataFrame
        excel_file_path = os.path.join(group_dir, 'data.xlsx')                                                          # Define the path for the Excel file
        if os.path.exists(excel_file_path):                                                                             # Append the DataFrame to the Excel file
            with pd.ExcelFile(excel_file_path) as xls:                                                                  # Load existing data to find the starting row for appending
                existing_df = pd.read_excel(xls, sheet_name='data')                                                     # Read the existing data to get the current number of rows
                startrow = existing_df.shape[0]                                                                         # Start appending after the last row
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:     # Write the new data without headers
                distances_df.to_excel(writer, index=False, header=False, startrow=startrow, sheet_name='data')
        else:
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:                                # Write the DataFrame to the Excel file with headers
                distances_df.to_excel(writer, index=False, header=True, sheet_name='data')
        print(f"Table saved : Group {group_number} : {tiff_file_name} - {ann_frame_idx}")
    
        # Prepare data for the consolidated DataFrame
        consolidated_data.append(pd.DataFrame({
            'Image': [tiff_file_name],
            'Cycle': [cycle],
            'Amps': [amps],
            'EHT Area': [contour_area],             # EHT area from the contour area calculation
            'Mid Width': [min_width_within_range],  # Mid width length found in the range
            'P. Grip Width': [max_width_top],       # Piston grip width
            'S. Grip Width': [max_width_bottom],    # Stationary grip width
            'EHT Length': [line_length],
        }))

    
    ##### After getting EHT dimensions, convert the consolidated data list to a DataFrame and save #####
    if consolidated_data:                                                                                                           # Check if consolidated_data is not empty
        consolidated_df = pd.concat(consolidated_data, ignore_index=True)
        consolidated_df = consolidated_df.drop_duplicates()                                                                         # Remove duplicate rows 
    else:
        print("No consolidated data to save.")
        return                                                                                                                      # Exit the function if there's no data to save
    consolidated_excel_file_path = os.path.join(group_dir, 'data_consolidated.xlsx')                                                # Define the path for the consolidated Excel file
    if os.path.exists(consolidated_excel_file_path):                                                                                # Check if the consolidated Excel file exists
        with pd.ExcelFile(consolidated_excel_file_path) as xls:                                                                     # Load existing data to find the starting row for appending
            existing_consolidated_df = pd.read_excel(xls, sheet_name='data_consolidated')
            combined_df = pd.concat([existing_consolidated_df, consolidated_df]).drop_duplicates().reset_index(drop=True)           # Combine existing DataFrame with the new one and drop duplicates
            startrow_consolidated = existing_consolidated_df.shape[0]                                                               # Start appending after the last row
        with pd.ExcelWriter(consolidated_excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:        # Append the unique DataFrame to the existing Excel file without headers
            combined_df.to_excel(writer, index=False, header=False, startrow=startrow_consolidated, sheet_name='data_consolidated')
    else:                                                                                                                           # If it does not already exist... 
        with pd.ExcelWriter(consolidated_excel_file_path, engine='openpyxl', mode='w') as writer:                                   # Write the DataFrame to the consolidated Excel file with headers
            consolidated_df.to_excel(writer, index=False, header=True, sheet_name='data_consolidated')
    print(f"Consolidated data saved: Group {group_number} : {tiff_file_name} - {ann_frame_idx}")

    ##### Show / Save the Sample's Plot #####
    plt.title(f"Group {group_number} - {tiff_file_name} - {cycle} - {amps}")
    plt.tight_layout()
    output_image_path = os.path.join(group_dir, f"{tiff_file_name}_{cycle}_{amps}.png")     # Save the figure before drawing and closing
    plt.savefig(output_image_path, bbox_inches='tight')                                     # Save the figure with tight layout
    plt.draw()  
    # plt.show()                                                                            # Comment out if you do not want to look at samples while being processed


##### Main Code #####
# Initializing variables 
clicked_points = []
clicked_labels = []
stored_contour_obj_3 = None     # Initialize a variable to store Object 3's (stationary pillars) contour from the first frame - pillars in same spot every frame
consolidated_data_list = []

# Create a figure
fig, axs = plt.subplots(1, 1, figsize=(10, 10))

###### MAIN FUNCTION FOR PROCESSING EACH GROUP ######
def process_jpeg_group_with_sam2(group_number, group_dir):
    # Prepare for storing user clicks
    global clicked_points, clicked_labels, current_object, clicks_per_object_eht_positive, clicks_per_object_eht_negative, clicks_per_object_grip_positive, clicks_per_object_magnet_positive, jpeg_to_tiff_mapping, axs, cid, frame_0_top_percentage, frame_0_bottom_percentage, dot_y_coord_bottom, dot_y_coord_bottom2, avg_y
    print(f"Running SAM2 model on group {group_number}...")
    
    frame_0_top_percentage = None
    frame_0_bottom_percentage = None
    dot_y_coord_bottom = None
    dot_y_coord_bottom2 = None

    # Initialize the SAM2 predictor
    sam2_checkpoint = r"C:\Users\scoeyman\Desktop\SAM MODEL\sam2_hiera_large.pt"                # Could also be small, or tiny - will need to be changed
    model_cfg = "sam2_hiera_l.yaml"                                                             # If above is small - change l to s, if above is tiny, change l to t
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)           # Building predictor
    inference_state = predictor.init_state(video_path=group_dir)                                # Initialize the inference state using the JPEG frames directory
    prompts = {}                                                                                # Hold all the clicks we add for visualization
    ann_frame_idx = 0                                                                           # ONLY CLICKING POINTS ON FRAME 0
    all_files = os.listdir(group_dir)                                                           # List all files in the JPEG directory and print for debugging
    frame_names = [f for f in all_files if f.endswith('.jpg')]                                  # Filter and sort JPEG files numerically
    frame_names.sort(key=lambda x: int(x.split('.')[0]))                                        # Sorting by the numeric part of the filename
    if not frame_names:                                                                         # Check if any frames exist
        print(f"No valid JPEG images found in {group_dir}. Skipping this group.")
        return

    # Load existing clicks (if they exist)
    eht_clicks_file_path = os.path.join(group_dir, 'eht_clicked_points.txt')
    eht_loaded_points, eht_loaded_labels = load_clicks_from_file(eht_clicks_file_path)
    magnet_clicks_file_path = os.path.join(group_dir, 'magnet_clicked_points.txt')
    magnet_loaded_points, magnet_loaded_labels = load_clicks_from_file(magnet_clicks_file_path)
    grip_clicks_file_path = os.path.join(group_dir, 'grip_clicked_points.txt')
    grip_loaded_points, grip_loaded_labels = load_clicks_from_file(grip_clicks_file_path)

    #### Check for EHT Points ####
    if eht_loaded_points:                               # If clicks exist, use them
        eht_points = eht_loaded_points
        eht_labels = eht_loaded_labels
        ann_obj_id = 1                                  # Object ID for EHT 
        prompts[ann_obj_id] = eht_points, eht_labels    # Store points and labels in prompts
    else: 
        # Prepare for storing user clicks for EHT
        clicked_points = []
        clicked_labels = []
        clicks_per_object_eht_positive = 4                          # Number of positive clicks for EHT
        clicks_per_object_eht_negative = 8                          # Number of negative clicks for EHT
        jpeg_filename = frame_names[ann_frame_idx]                  # Get the specific filename
        jpeg_file_path = os.path.join(group_dir, jpeg_filename)     # Create the full path for the image
        current_object = "eht"                                      # Set current object to EHT  
        ann_obj_id = 1                                              # Unique ID for EHT

        # Clear the current axis and show the image for EHT for clicking
        plt.figure(figsize=(10, 10))                                        # Set figure size to 10x10
        plt.clf()
        plt.title(f"Frame {ann_frame_idx} - Select {clicks_per_object_eht_positive} positive clicks and {clicks_per_object_eht_negative} negative clicks for EHT")
        plt.imshow(Image.open(jpeg_file_path))
        axs = plt.gca()                                                     # Get current axis for drawing
        cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)                                                # Block until the plot is closed
        plt.close()

        # After collecting clicks for EHT, process them
        if clicked_points:
            eht_points = np.array(clicked_points, dtype=np.float32)
            eht_labels = np.array(clicked_labels, dtype=np.int32)
            prompts[ann_obj_id] = eht_points, eht_labels                        # Store points and labels in prompts
            save_clicks_to_file(eht_points, eht_labels, eht_clicks_file_path)   # Saving clicks to file using function

    #### Check for Magnet Points ####
    if magnet_loaded_points:                                        # If clicks exist, use them
        magnet_points = magnet_loaded_points
        magnet_labels = magnet_loaded_labels
        ann_obj_id = 2
        prompts[ann_obj_id] = magnet_points, magnet_labels          # Store points and labels in prompts
    else: 
        # Prepare for storing user clicks for magnet
        clicked_points = []
        clicked_labels = []
        clicks_per_object_magnet_positive = 1                       # Number of positive clicks for magnet
        jpeg_filename = frame_names[ann_frame_idx]                  # Get the specific filename
        jpeg_file_path = os.path.join(group_dir, jpeg_filename)     # Create the full path for the imag
        current_object = "magnet"                                   # Set current object to magnet
        ann_obj_id = 2                                              # Unique ID for magnet

        # Clear the current axis and show the image for magnet
        plt.figure(figsize=(10, 10))                                        # Set figure size to 10x10
        plt.clf()
        plt.title(f"Frame {ann_frame_idx} - Select {clicks_per_object_magnet_positive} positive clicks for center of magnet")
        plt.imshow(Image.open(jpeg_file_path))
        axs = plt.gca()                                                     # Get current axis for drawing
        cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)                                                # Block until the plot is closed
        plt.close()

        # After collecting clicks for magnet, process the clicks
        if clicked_points:
            magnet_points = np.array(clicked_points, dtype=np.float32)
            magnet_labels = np.array(clicked_labels, dtype=np.int32)
            prompts[ann_obj_id] = magnet_points, magnet_labels              # Store points and labels in prompts
            save_clicks_to_file(magnet_points, magnet_labels, magnet_clicks_file_path)

    ##### Check for Grip Points #####
    if grip_loaded_points:                                                  # If clicks exist, use them
        grip_points = grip_loaded_points
        grip_labels = grip_loaded_labels
        ann_obj_id = 3
        prompts[ann_obj_id] = grip_points, grip_labels                      # Store points and labels in prompts
        # Handle average y-coordinate for Object 3 clicked points
        avg_y = np.mean([point[1] for point in grip_points])                # Y coorindate for stationary grips
        avg_x = np.mean([point[0] for point in grip_points])                # Optional if you want the center average (x cooridnate)
        print(f"Grip average point: {avg_x}, {avg_y}")
    else: 
        # Prepare for storing user clicks for Grip
        clicked_points = []
        clicked_labels = []
        clicks_per_object_grip_positive = 2                         # Number of positive clicks for Grip
        jpeg_filename = frame_names[ann_frame_idx]                  # Get the specific filename
        jpeg_file_path = os.path.join(group_dir, jpeg_filename)     # Create the full path for the image
        current_object = "grip"                                     # Set current object to grip
        ann_obj_id = 3                                              # Unique ID for grip

        # Clear the current axis and show the image for Grip
        plt.figure(figsize=(10, 10))                                                # Set figure size to 10x10
        plt.clf()
        plt.title(f"Frame {ann_frame_idx} - Select {clicks_per_object_grip_positive} pillar tips for Stationary Grip")
        plt.imshow(Image.open(jpeg_file_path))
        axs = plt.gca()                                                             # Get current axis for drawing
        cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)                                                        # Block until the plot is closed
        plt.close()
        # After collecting clicks for Grip, process the clicks
        if clicked_points:
            grip_points = np.array(clicked_points, dtype=np.float32)
            grip_labels = np.array(clicked_labels, dtype=np.int32)
            prompts[ann_obj_id] = grip_points, grip_labels                          # Store points and labels in prompts
            save_clicks_to_file(grip_points, grip_labels, grip_clicks_file_path)
            # Handle average y-coordinate for Object 3 clicked points
            avg_y = np.mean([point[1] for point in grip_points])                    # Y coorindate for stationary grips
            avg_x = np.mean([point[0] for point in grip_points])                    # Optional if you want the center average (x cooridnate)
            print(f"Grip average point: {avg_x}, {avg_y}")


    ##### Now process the remaining frames with SAM predictions ##### 
    # All clicks done on frame 0, sam2 model will make predictions on rest of frames
    for ann_frame_idx in range(0, len(frame_names)):        # Start from the second frame
        jpeg_filename = frame_names[ann_frame_idx]  
        jpeg_file_path = os.path.join(group_dir, jpeg_filename)  
        # Perform predictions for each object
        for obj_id in prompts.keys():
            points, labels = prompts[obj_id]
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        # Visualize predictions for the current frame
        visualize_predictions(prompts, group_dir, frame_names, predictor, inference_state, ann_frame_idx, group_number)
        
# Function to delete output files in the folder
def delete_output_files(folder):
    for file_name in os.listdir(folder):
        if 'output' in file_name:
            file_path = os.path.join(folder, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Function to create the GUI
def create_gui():
    global root, folder_path, plate_vars, group_vars, day_vars, well_vars
    global show_images_var, save_images_var, files_to_process, plate_frame, group_frame, day_frame, well_frame
    global container_frame, folder_frame, result_text, options_frame, results_folder_var

    root = tk.Tk()
    root.title("File Filter with Multi-Select")
    root.geometry('1200x800')

    folder_path = tk.StringVar()
    results_folder_var = tk.StringVar()

    plate_vars = {}
    group_vars = {}
    day_vars = {}
    well_vars = {}
    show_images_var = tk.BooleanVar()
    save_images_var = tk.BooleanVar()
    files_to_process = []
    container_frame = tk.Frame(root)
    container_frame.pack(fill=tk.BOTH, expand=True)
    folder_frame = tk.Frame(container_frame)
    folder_frame.pack(side=tk.TOP, fill=tk.X)
    tk.Label(folder_frame, text="Selected Folder:").pack(side=tk.LEFT)
    tk.Entry(folder_frame, textvariable=folder_path, width=50).pack(side=tk.LEFT, padx=10)
    tk.Button(folder_frame, text="Select Folder", command=select_folder).pack(side=tk.LEFT)
    tk.Label(folder_frame, text="Results Folder:").pack(side=tk.LEFT, padx=10)
    tk.Entry(folder_frame, textvariable=results_folder_var, width=50).pack(side=tk.LEFT, padx=10)
    tk.Button(folder_frame, text="Select Results Folder", command=select_results_folder).pack(side=tk.LEFT)
    # Create frames for filter options (Plate, Group, Day, Well)
    options_frame = tk.Frame(container_frame)
    options_frame.pack(side=tk.TOP, fill=tk.X)
    plate_frame = tk.LabelFrame(options_frame, text="Plates", padx=5, pady=5)
    plate_frame.pack(side=tk.LEFT, padx=10, pady=10)
    group_frame = tk.LabelFrame(options_frame, text="Groups", padx=5, pady=5)
    group_frame.pack(side=tk.LEFT, padx=10, pady=10)
    day_frame = tk.LabelFrame(options_frame, text="Days", padx=5, pady=5)
    day_frame.pack(side=tk.LEFT, padx=10, pady=10)
    well_frame = tk.LabelFrame(options_frame, text="Wells", padx=5, pady=5)
    well_frame.pack(side=tk.LEFT, padx=10, pady=10)
    # Add buttons for filtering and processing
    filter_frame = tk.Frame(container_frame)
    filter_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    tk.Button(filter_frame, text="Apply Filters", command=apply_filters).pack(side=tk.LEFT, padx=10)
    tk.Button(filter_frame, text="Process Images", command=process_images).pack(side=tk.LEFT, padx=10)
    tk.Checkbutton(filter_frame, text="Show Images", variable=show_images_var).pack(side=tk.LEFT, padx=10)
    tk.Checkbutton(filter_frame, text="Save Images", variable=save_images_var).pack(side=tk.LEFT, padx=10)
    result_frame = tk.Frame(container_frame)
    result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, height=10)
    result_text.pack(fill=tk.BOTH, expand=True)
    root.mainloop()

if __name__ == "__main__":
    create_gui()
