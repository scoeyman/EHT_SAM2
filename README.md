GUIDELINES FOR PROCESSING EHTS WITH SAM2 MODEL

Step 0: 
- I used python 3.12
- Download packages required in requirements.txt
- Download the SAM MODEL folder
- Have images to be processed in designated folder

Step 1: Editing Code... 
1) Open the "eht_sam2.py" code 
2) Assuming images are named in this format "P1Aw101x10-00filt"
- plate 1, group A, well 1, day 01, cycle 1, amps 0-00
- if a different format, you will need to edit the "get_plate_group_day_well" function
3) Assuming you don't want to change the number of clicks per object
- for eht, 4 positive and 8 negative
- for magnet, 1 positive 
- for stationary pillars, 2 positive (on pillar tips) 
- if you do, you will need to edit the clicks_per_object variables in the process_jpeg_group_with_sam2 function 
4) You will need to edit the path to the sam2_checkpoint under the process_jpeg_group_with_sam2 function
- the checkpoint is in the SAM MODEL folder (I use the sam2_hiera_large.pt)
- you can also change the model_cfg if needed (I use the sam2_hiera_l.yaml) 
- if changing, make sure the checkpoint and model_cfg match (large, small, or tiny) 

Step 2: Running Code... 
1) After making edits, hit run 
2) A dialog box will open
- hit the Select Folder button, locate your image folder, and select the folder itself 
- ignore the Results Folder (not needed)
- once selected, Plates, Groups, Days, and Wells will pop up with the different options from filtering your image names
- you can select specific options to focus on just those images, or leave the boxes uncheck to select all images
3) Hit the Apply Filters button 
- the image names selected based off the filters will populate the white space
4) Hit Process Images to being the processing 

Step 3: Processing... 
1) Where your code is running, in the terminal you should see something like 
- Processing group: P1Aw101, Group Number: 1
- Running SAM2 model on group 1...
2) A figure will appear showing the first frame (Frame 0) for the group being processed
- Pay attention to the figure titles - they will tell you the number of clicks for each object (even if you changed them in Step 1) 
- Click for positive selections first always, and then the negative. Right click and left click on mouse both count so be careful 
5) Click for EHT (object 1)
- you should see green stars for positive clicks and red stars for negative clicks 
- For positive, like to click towards top of EHT near the piston grip (but not on), the bottom of the EHT near the stationary grip (but not on), and then two clicks near the middle of the EHT 
- for negative, i like to click 3 times on each side of the eht where you can see the well wall or bottom and then one click on each of the grips 
6) After clicking the negative clicks, the image should automatically close. You will need to also close the blank Figure 1 that appears. A new figure will then open for selecting the magnet object. 
7) Follow the image title and click in the center of the magnet (object 2).
- The image will automatically close and the next image for selecting stationary pillar object will open. 
8) Follow the image title and click twice on the stationary pillar tips (object 3). The average y coordinate of these clicks will serve as the end point for EHT length throughout the group being processed.
- The image will close and in your terminal you should see the SAM2 model processing the rest of the frames for the group being processed. 
9) Repeat this process for each group. 
- Note, the clicks you select will be saved in a folder numbered by the group number. 

Step 4: Overserving Results 
1) In your image folder.... you will see your .tiff images as well as as "jpeg_frames" folder
- this is where your results are
- this folder contains subfolders for each group named after their group # (1 - X) as well as a mapping file that records the group name for each designated number
- inside of each numbered subfolder you will find the model processing results including 
--- text files for each objects clicked points 
--- images of the tracing and measurements for each frame 
--- a data excel file that includes the ALL the length and width measurements for each percentage of the EHT at each cycle and amp level
--- a consolidated data excel file that includes just the main EHT measurements for each cycle and amp level, as well as EHT strain



Important Notes: 
- if you change the images... the filtering for groups could change and mess up your clicking records... once you have started processing an image folder.. do not add or remove images from that folder 
- if you close the code before finishing all groups, make sure you select the same image folder when you run the code next (the jpeg_folder will be ignored during the image filtering)
--- the groups that have already been clicked will automatically process using those clicks and you will not be asked to click again until a group that hasn't been processed is next to be processed
--- you can watch the groups being processed in your code terminal


Step 5: EHT Stiffness (After Clicking /Processing All Groups) 
1) Open the "getting_stiffness.py" code 
2) Edit the main_directory variable to match your folder being processed 
3) You may need to edit the variables experiment, group, label, match, and day depending on your directory structure
4) This code is hard coded for three cycles of 0-15 amp mechanical testing... edit the variables under "if 'Stiffness' in df.columns and 'R2' in df.columns:" to match your number of cycles and amp values 
5) Run the code to gather data from all "data_consolidated" excel files in the testing folder







