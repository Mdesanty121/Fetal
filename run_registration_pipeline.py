#!/usr/bin/env python3
#
#
#  ===============================================================================
# Metadata
# ===============================================================================
__author__ = 'mnd'
__contact__ = 'megan.desanty@childrens.harvard.edu'
__copyright__ = ''
__license__ = ''
__date__ = '05/2025'
__version__ = '0.1'

# ===============================================================================
# Import statements
# ===============================================================================
import sys
import numpy as np
import os
import pickle
import pdb
import glob
import shutil
import pandas as pd
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor
import nibabel as nib
import re

from COPY_step4_segment_all_uninterleaved_echoes import chunks_fullpaths


def usage():
    print("""
    AUTHOR: megan.desanty@childrens,harvard.edu

    DESCRIPTION:
        Main class to complete the registration pipeline

    USAGE:  
        python3 run_registration_pipeline.py
    """)
    sys.exit(0)

# ===============================================================================
# Helper functions
# ===============================================================================
# Function to run split_nii
def run_split_nii(subj, chunk, echo_idx):
    split_nii(
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split/{chunk}/echo_{echo_idx}/',
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo_idx}/',
        True, True,
    )


# Function to save NIfTI slices for one echo
def save_nifti_slices(subj, chunk, echo_idx, arr, aff):
    output_dir = os.getcwd() + f'/multiecho-images/{subj}/echoes_split/{chunk}/echo_{echo_idx}/'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(arr.shape[-1]):
        print(f"{subj} Chunk: {chunk}, Echo: {echo_idx}, Volume: {i}")
        nib.save(
            nib.Nifti1Image(arr[..., i], aff),
            output_dir + f'{subj}_{i:04d}.nii.gz'
        )


def split_nii(in_folder, out_folder, starts_with_even, nnunet_prep=False):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for i, nii_file in enumerate(
            sorted(
                [
                    f
                    for f in os.listdir(in_folder)
                    if f.endswith(".nii") or f.endswith(".nii.gz")
                ]
            )
    ):
        nii = nib.load(os.path.join(in_folder, nii_file))
        affine = nii.affine

        # if nnunet_prep:
        #    affine = np.eye(4)

        def split(img, even):
            img_s = np.copy(img)
            for j in range(1 if even else 0, img.shape[-1], 2):
                if j > 0 and j < img.shape[-1] - 1:
                    img_s[:, :, j] = (img[:, :, j - 1] + img[:, :, j + 1]) / 2
            return img_s

        img = nii.get_fdata()

        img_even = split(img, even=True)
        img_odd = split(img, even=False)
        if starts_with_even:
            img0, img1 = img_even, img_odd
        else:
            img0, img1 = img_odd, img_even

        if nnunet_prep:
            nii0 = nib.Nifti1Image(img0, affine)
            nib.save(nii0, os.path.join(out_folder, "%04d_0000.nii.gz" % (2 * i)))
            nii1 = nib.Nifti1Image(img1, affine)
            nib.save(nii1, os.path.join(out_folder, "%04d_0000.nii.gz" % (2 * i + 1)))
        else:
            nii0 = nib.Nifti1Image(img0, affine)
            nib.save(nii0, os.path.join(out_folder, "%04d.nii.gz" % (2 * i)))
            nii1 = nib.Nifti1Image(img1, affine)
            nib.save(nii1, os.path.join(out_folder, "%04d.nii.gz" % (2 * i + 1)))

# ===============================================================================
# Main Steps
# ===============================================================================

def step1(source_dir, dest_dir, excel_file):
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Load subject IDs from the Excel file
    df = pd.read_excel(excel_file)
    subject_ids = df.iloc[:, 0].astype(str).tolist()

    # Loop through subject IDs and search for matching folders
    for subject_id in subject_ids:
        for sess_num in ['01', '02', '03']:
            folder_pattern = f"sub-{subject_id}_ses-{sess_num}"
            matching_folders = glob.glob(os.path.join(source_dir, folder_pattern))

            for folder in matching_folders:
                # Check if any matching file exists in the folder
                matching_files = glob.glob(os.path.join(folder, '*EPI_SMS2_GRAPPA2*.nii.gz'), recursive=True)
                matching_files = [f for f in matching_files if 'reference' not in f]
                matching_files2 = glob.glob(os.path.join(folder, '*3echoes*placentaT2star*Unaliased*.nii.gz'),recursive=True)
                matching_files2 = [f for f in matching_files2 if 'reference' not in f]

                dest_path = os.path.join(dest_dir, os.path.basename(folder), 'echoes')

                if matching_files or matching_files2:
                    os.makedirs(dest_path, exist_ok=True)
                    print(f"Copying folder {folder} to {dest_path}")
                    for fname in matching_files or matching_files2:
                        shutil.copy(fname, os.path.join(dest_path, os.path.basename(fname)))

def step2(dest_dir):
    # Iterate through each subfolder in the base directory
    for subfolder in os.listdir(dest_dir):
        subfolder_path = os.path.join(dest_dir, subfolder, 'echoes')
        if os.path.isdir(subfolder_path):

            # Get and natsort the files in the current subfolder
            files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
            files = natsorted(files)

            assert len(files) % 3 == 0

            print('subfolder')

            # Chunk files into groups of 3
            for i in range(0, len(files), 3):
                chunk = files[i:i + 3]
                new_subfolder = os.path.join(subfolder_path, f'chunk_{i // 3 + 1}')

                os.makedirs(new_subfolder, exist_ok=True)

                # Move each file in the chunk to the new subfolder
                for file in chunk:
                    src = os.path.join(subfolder_path, file)
                    dest = os.path.join(new_subfolder, file)
                    shutil.move(src, dest)

    print('Files have been chunked and moved successfully.')

def step3(subjs, cwd, max_processes):
    # Loop through all the subjects we are analyzing
    for subj in subjs:
        print(f"Analyzing subject {subj}")
        # Get the paths for the chunks
        chunks_fullpaths = natsorted(glob.glob(cwd + f'/multiecho-images/{subj}/echoes/chunk*'))
        chunks = [os.path.basename(i) for i in chunks_fullpaths]
        # Loop through all the chunks
        for chunk in chunks:
            # Create necessary directories
            # Make echoes dynamic

            # Get all the files for that chunk
            files = glob.glob(cwd + f'/multiecho-images/{subj}/echoes/{chunk}/' + '*.nii.gz')

            # Initialize list to store extracted numbers
            echo_numbers = []

            # Loop through files and extract echo numbers
            for file in files:
                match = re.search(r'_e(\d+)\.nii\.gz$', file)
                if match:
                    echo_numbers.append(int(match.group(1)))

            for echo in echo_numbers:
                # Make folders based on echo number
                os.makedirs(
                    cwd + f'/multiecho-images/{subj}/echoes_split/{chunk}/echo_{echo}/',
                    exist_ok=True
                )
                os.makedirs(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/',
                    exist_ok=True
                )

            # Load combined images
            combined_images = natsorted(glob.glob(
                cwd + f'/multiecho-images/{subj}/echoes/{chunk}/*.nii.gz'
            ))

            assert len(combined_images) == 3

            nii_files = [nib.load(img) for img in combined_images]
            nii_data = [nii.get_fdata() for nii in nii_files]
            nii_affines = [nii.affine for nii in nii_files]

            # Parallel saving of NIfTI slices
            with ProcessPoolExecutor(max_processes) as executor:
                futures = []
                for echo_idx, (arr, aff) in enumerate(zip(nii_data, nii_affines)):
                    futures.append(executor.submit(save_nifti_slices, subj, chunk, echo_idx, arr, aff))

                # Ensure all save tasks are completed before proceeding
                for future in futures:
                    future.result()

            # Parallel split_nii execution
            with ProcessPoolExecutor(max_processes) as executor:
                futures = []
                for echo_idx in range(3):
                    futures.append(executor.submit(run_split_nii, subj, chunk, echo_idx))

                # Ensure all split tasks are completed before proceeding
                for future in futures:
                    future.result()
                    future.result()

    return chunks

def step4(subjs, chunks):

    for subj in subjs:

        for chunk in chunks:
            splitter = ['&', '&', '']

            # Create necessary directories
            for echo in range(3):
                os.makedirs(
                    os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/',
                    exist_ok=True
                )

                print('nnUNetv2_predict -d 999 -i {} -o {} -f 0 1 2 3 4 -c 3d_fullres -chk checkpoint_best.pth'.format(
                    os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/',
                    os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/'
                ))

def main():
    # Globally define current working directory
    cwd = input("Please define the directory where you want all the folders to be created: ")
    # Currently: '/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel'
    #
    # # ================================================================================
    # # Step 1 - copy multiecho time series
    # # ================================================================================
    # # Ask for source_dir, destination_dir, and excel file location
    # source_dir = input("Please enter the destination of your multiecho files: ")
    # # Current location: '/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel/data'
    # dest_dir = input("Please enter the destination of where you want your outputs to be placed: ")
    # # Current location: '/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel/multiecho-images/'
    # excel_file = input("Please enter the location (including the file name) of the excel file containing the subject ID's you want to analyze: ")
    # # Current location: '/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel/subj_list_TTTS_test.xlsx'
    # # Call the first step function
    # step1(source_dir, dest_dir, excel_file)
    # print("Step 1 complete")
    #
    # # ================================================================================
    # # Step 2 - Create chunks of three
    # # ================================================================================
    # step2(dest_dir)
    # print("Step 2 complete")
    #
    # ================================================================================
    # Step 3 - Split & Uninterleave echoes
    # ================================================================================
    # Define subjects as the ones listed in the multi echo images folder
    # User-defined number of parallel processes
    max_processes = input("Please enter the number of parallel processes you want to move forward with: ")
    # Currently 8
    subjects = natsorted(os.listdir(cwd + '/multiecho-images/'))
    chunks = step3(subjects, cwd, max_processes)

    # ================================================================================
    # Step 4 - Segment all uninterleaved echoes
    # ================================================================================
    step4(subjects, chunks)

if __name__ == "__main__":
    main()