import os
import shutil
import pandas as pd
import glob
import pdb

from pdb import set_trace as st

# Define paths
#SOURCE_DIR = '/data/vision/polina/projects/fetal/common-data/MAP-AFI-allimages/'
#DEST_DIR = '/data/vision/polina/scratch/dey/esra-full-dataset-analysis/multiecho-images/'
#EXCEL_FILE = './SubjectListWithT2s.xlsx'
SOURCE_DIR = '/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel/data'
DEST_DIR = '/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel/multiecho-images/'
EXCEL_FILE = './subj_list_TTTS_test.xlsx' 

# Create the destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Load subject IDs from the Excel file
df = pd.read_excel(EXCEL_FILE)
subject_ids = df.iloc[:, 0].astype(str).tolist()  # First column, convert to string

# Loop through subject IDs and search for matching folders
for subject_id in subject_ids:
    for sess_num in ['01', '02', '03']:
        folder_pattern = f"sub-{subject_id}_ses-{sess_num}"
        matching_folders = glob.glob(os.path.join(SOURCE_DIR, folder_pattern))
    
        for folder in matching_folders:
            # Check if any matching file exists in the folder
            matching_files = glob.glob(os.path.join(folder, '*EPI_SMS2_GRAPPA2*.nii.gz'), recursive=True)
            matching_files = [f for f in matching_files if 'reference' not in f]
            matching_files2 = glob.glob(os.path.join(folder, '*3echoes*placentaT2star*Unaliased*.nii.gz'), recursive=True)
            matching_files2 = [f for f in matching_files2 if 'reference' not in f]

            dest_path = os.path.join(DEST_DIR, os.path.basename(folder), 'echoes')

            if matching_files or matching_files2:
                os.makedirs(dest_path, exist_ok=True)
                print(f"Copying folder {folder} to {dest_path}")
                for fname in matching_files or matching_files2:
                    shutil.copy(fname, os.path.join(dest_path, os.path.basename(fname)))


