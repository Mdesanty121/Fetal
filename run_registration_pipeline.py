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
import subprocess

from skimage.morphology import binary_dilation
from scipy.ndimage import binary_fill_holes

from multiprocessing import Pool

import csv



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


def getNumEchoes(cwd, subj, chunk):
    # Get all the files for that chunk
    files = glob.glob(cwd + f'/multiecho-images/{subj}/echoes/{chunk}/' + '*.nii.gz')

    # Initialize list to store extracted numbers
    echo_numbers = []

    # Loop through files and extract echo numbers
    for file in files:
        match = re.search(r'_e(\d+)\.nii\.gz$', file)
        if match:
            echo_numbers.append(int(match.group(1)))

    return echo_numbers

def runShellScript(script_path):
    # Set environment variables for nnUNet
    os.environ["nnUNet_raw"] = "/neuro/labs/grantlab/research/uterus_data/registration/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "/neuro/labs/grantlab/research/uterus_data/registration/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "/neuro/labs/grantlab/research/uterus_data/registration/nnUNet_results"

    # Make excecutable
    os.chmod(script_path, 0o775)

    # Run script
    try:
        subprocess.run([script_path], check=True)
        print("Prediction script executed successfully.")
    except subprocess.CalledProcessError as e:
        with open("error.log", "a") as log_file:
            log_file.write(f"An error occurred while running the script: {e}\n")


def compose_transform(arg_tuple):
    args, cwd = arg_tuple
    warp, linear, savedir, subj, chunk = args

    output_path = os.path.join(savedir, os.path.basename(warp))
    template_path = os.path.join(cwd,f'multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/T_template0.nii.gz')

    cmd = f'ComposeMultiTransform 3 {output_path} -R {template_path} {warp} {linear}'
    os.system(cmd)


def process_single_image(args):
    i, echo, input_img, input_seg, warp, template, subj, chunk, cwd = args

    output_img = os.path.join(cwd,f'multiecho-images/{subj}/echoes_split_uninterleaved_registered/{chunk}/echo_{echo}/{os.path.basename(input_img)}')
    output_seg = os.path.join(cwd, f'multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/{os.path.basename(input_seg)}')

    os.system(f'antsApplyTransforms -d 3 -e 0 -t {warp} -r {template} -i {input_img} -o {output_img} -n Linear -v')
    os.system(f'antsApplyTransforms -d 3 -e 0 -t {warp} -r {template} -i {input_seg} -o {output_seg} -n GenericLabel -v')

    if not os.path.exists(output_img):
        raise RuntimeError(f"Failed to create output image: {output_img}")
    if not os.path.exists(output_seg):
        raise RuntimeError(f"Failed to create output segmentation: {output_seg}")



def process_echo(args):
    subj, chunk, echo = args

    os.makedirs(
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_averagepostregistration/{chunk}/echo_{echo}/',
        exist_ok=True
    )

    print('Processing subject {} chunk {} echo {}'.format(subj, chunk, echo))

    imgs = natsorted(glob.glob(
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_registered/{chunk}/echo_{echo}/*.nii.gz'))
    segs = natsorted(glob.glob(
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/*.nii.gz'))
    if len(imgs) == 0:
        print('No images found for subject {} chunk {} echo {}'.format(subj, chunk, echo))
        return

    assert len(imgs) == len(segs)

    # Create average image
    ref = nib.load(imgs[0])
    average_img = np.zeros_like(ref.get_fdata()).astype(np.float32)
    print('averaging images')

    for i in range(len(imgs)):
        average_img += nib.load(imgs[i]).get_fdata().astype(np.float32)

    average_img = average_img / len(imgs)

    nib.save(
        nib.Nifti1Image(
            average_img,
            ref.affine,
        ),
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_averagepostregistration/{chunk}/echo_{echo}/average.nii.gz'
    )

def compute_jacobian(args):
    warp, savedir = args
    cmd = 'CreateJacobianDeterminantImage 3 {} {}/{}.nii.gz 1 1'.format(
        warp, savedir, os.path.basename(warp),
    )
    os.system(cmd)


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
            # Get the correct number of echoes for this subject (dynamic)
            echo_numbers = getNumEchoes(cwd, subj, chunk)
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
                for echo_idx in echo_numbers:
                    futures.append(executor.submit(run_split_nii, subj, chunk, echo_idx))

                # Ensure all split tasks are completed before proceeding
                for future in futures:
                    future.result()
                    future.result()

    return chunks

def step4(subjs, chunks, cwd):
    # Define script path
    script_path = os.path.join(cwd + 'run_segment_all_uninterleaved_echoes.sh')
    # Open script for writing
    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n\n')
        print("Script opened for step 4")
        # Loop through subjects we are analyzing
        for subj in subjs:
            for chunk in chunks:
                # Dynamically get the number of echoes for this subject and chunk
                echo_numbers = getNumEchoes(cwd, subj, chunk)
                # Loop through echoes
                for echo in echo_numbers:
                    # Create output directory
                    output_dir = os.path.join(
                        cwd, f'multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/'
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    # Compose input and output paths
                    input_path = os.path.join(
                        cwd, f'multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/'
                    )

                    # Build command
                    command = (
                        f'nnUNetv2_predict -d 999 -i "{input_path}" -o "{output_dir}" '
                        f'-f 0 1 2 3 4 -c 3d_fullres -chk checkpoint_best.pth\n'
                    )

                    f.write(command)
    print("All commands added to script")
    # Make script excecutable & run it
    print("Running script")
    runShellScript(script_path)

def step5(subjs, chunks, cwd):
    # Create the shell script file and write the command to it
    # Define script path
    script_path = os.path.join(cwd + 'run_independent_bfc_echo1.sh')
    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n\n')
        # Loop through all subjects being analyzed
        for subj in subjs:
            # Loop through all chunks for each subject
            for chunk in chunks:

                # Create necessary directories for the first echo
                for echo in [1]:
                    os.makedirs(
                        cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/echo_{echo}/',
                        exist_ok=True
                    )

                    os.makedirs(
                        cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_uterusmask/{chunk}/echo_{echo}/',
                        exist_ok=True
                    )

                    imgs = natsorted(glob.glob(
                        cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/*.nii.gz'))
                    segs = natsorted(glob.glob(
                        cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/*.nii.gz'))

                    assert len(imgs) == len(segs)
                    assert len(imgs) > 0


                    for j, (img, seg) in enumerate(zip(imgs, segs)):

                        if j % 10 == 0:
                            simult = ''
                        else:
                            simult = '&'

                        # Create and save binary version of multilabel seg to be used as BFC mask:
                        nii = nib.load(seg)
                        segarr = nii.get_fdata()
                        segarr = (segarr > 0).astype(np.uint8)

                        segarr = binary_dilation(segarr)
                        segarr = binary_fill_holes(segarr)
                        segarr = segarr.astype(np.uint8)

                        nib.save(
                            nib.Nifti1Image(segarr, nii.affine),
                            '{}/{}'.format(
                                cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_uterusmask/{chunk}/echo_{echo}/',
                                os.path.basename(seg)),
                        )
                        # Create shell script command
                        out_dir = os.path.join(cwd,f'multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/echo_{echo}/')
                        mask_dir = os.path.join(cwd,f'multiecho-images/{subj}/echoes_split_uninterleaved_uterusmask/{chunk}/echo_{echo}/')
                        img_name = os.path.basename(img)
                        seg_name = os.path.basename(seg)

                        command = (
                            f'N4BiasFieldCorrection -r 0 -d 3 -i "{img}" '
                            f'-o ["{out_dir}/{img_name}", placeholder.nii.gz] '
                            f'-w "{mask_dir}/{seg_name}" -v -s 2 -c [400x400x400,0.00] {simult}\n'
                        )
                        # Write command to shell script
                        f.write(command)

    # Run through shell script function
    runShellScript(script_path)

def step6(subjs, chunks, cwd):
    # Loop through each subject we are analyzing
    for subj in subjs:
        # Loop through each chunk for that subject
        for chunk in chunks:

            # Create necessary directories
            for echo in [1]:
                os.makedirs(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_{echo}/',
                    exist_ok=True
                )

                imgs = natsorted(glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/echo_{echo}/*.nii.gz'))
                segs = natsorted(glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/*.nii.gz'))

                assert len(imgs) == len(segs)
                assert len(imgs) > 0

                with open(
                        cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/imgseg_paths.csv',
                        'w') as stream:
                    writer = csv.writer(stream, quoting=csv.QUOTE_NONE, quotechar='', lineterminator='\n')
                    for img, seg in zip(imgs, segs):
                        writer.writerow([os.path.abspath(img), os.path.abspath(seg)])

                with open(
                        cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/atlas_build.sh',
                        'w') as stream:
                    stream.write(
                        'antsMultivariateTemplateConstruction2.sh -d 3 -a 2 -k 2 -o {}/T_ -g 0.25  -j 8 -w 1x0.2  -n 0  -r 1  -i 4  -c 2  -m CC[2]  -m MSQ  -l 1  -t "SyN[0.3, 1.5, 0]" -q 100x100x80x60x30x30  -f 4x4x3x2x1x1  -s 2.5x2.0x1.5x1.0x0.5x0.0  -b 0  imgseg_paths.csv'.format(
                            cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_{echo}/'))

def step7(subjs, chunks, cwd):
    base_dir = '/net/rc-fs-nfs.tch.harvard.edu/FNNDSC-e2/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel/multiecho-images/'
    for subj in subjs:

        for chunk in chunks:
            for echo in [1]:
                # Define paths
                script_dir = os.path.join(cwd,f'multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}')
                script_path = os.path.join(script_dir, 'run_atlas_build.sh')

                # Ensure the directory exists
                os.makedirs(script_dir, exist_ok=True)

                # Write the shell script
                with open(script_path, 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('cd "{}"\n'.format(script_dir))
                    f.write('sh atlas_build.sh\n')

    # Make it executable and run it
    runShellScript(script_path)

def step8(subjs, chunks, cwd):
    for subj in subjs:

        for chunk in chunks:

            # Create necessary directories
            for echo in [1]:
                os.makedirs(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_{echo}/',
                    exist_ok=True
                )

            # Load input images/echoes to warp
            input_e0 = natsorted(
                glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_1/*.nii.gz'
                )
            )

            # Load warps to apply:
            warps = natsorted(
                glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/*[!Inverse]Warp.nii.gz'
                )
            )

            # Check if warps is empty and log to file if so
            if len(warps) == 0:
                with open('empty_warps.log', 'a') as f:
                    warp_path = cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/'
                    f.write(f"Empty warps directory: {warp_path}\n")
                continue

            linears = natsorted(
                glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/*GenericAffine.mat'
                )
            )
            del linears[-1]

            # import pdb; pdb.set_trace()

            assert len(input_e0) > 0
            assert len(warps) == len(input_e0)
            assert len(warps) == len(linears)

            for echo in [1]:
                savedir = cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_{echo}/'
                os.makedirs(
                    savedir,
                    exist_ok=True
                )
                with Pool(16) as pool:
                    args = [(warps[k], linears[k], savedir, subj, chunk) for k in range(len(warps))]
                    inputs = [(a, cwd) for a in args]
                    pool.map(compose_transform, inputs)

def step9(subjs, chunks, cwd):
    for subj in subjs:

        for chunk in chunks:

            # Create necessary directories and load input images/segmentations
            input_imgs = []
            input_segs = []
            # Dynamic echoes
            echo_numbers = getNumEchoes(cwd, subj, chunk)
            # Loop through echoes
            for echo in echo_numbers:
                os.makedirs(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_registered/{chunk}/echo_{echo}/',
                    exist_ok=True
                )
                os.makedirs(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/',
                    exist_ok=True
                )

                input_imgs.append(
                    natsorted(
                        glob.glob(
                            cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/*.nii.gz'
                        )
                    )
                )
                input_segs.append(
                    natsorted(
                        glob.glob(
                            cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/*.nii.gz'
                        )
                    )
                )
            assert len(input_imgs[0]) > 0
            assert len(input_imgs[0]) == len(input_imgs[1])
            assert len(input_imgs[0]) == len(input_imgs[2])

            assert len(input_segs[0]) > 0
            assert len(input_segs[0]) == len(input_segs[1])
            assert len(input_segs[0]) == len(input_segs[2])

            assert len(input_imgs[0]) == len(input_segs[0])
            assert len(input_imgs[1]) == len(input_segs[1])
            assert len(input_imgs[2]) == len(input_segs[2])

            # Load warps to apply:
            warps = natsorted(
                glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_1/*[!Inverse]Warp.nii.gz'
                )
            )

            # Check if warps is empty and log to file if so
            if len(warps) == 0:
                with open('empty_warps_for_other_echoes.log', 'a') as f:
                    warp_path = cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/'
                    f.write(f"Empty warps directory: {warp_path}\n")
                continue

            assert len(warps) == len(input_imgs[0])

            template = cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/T_template0.nii.gz'

            # Call antsApplyTransforms for each echo:
            for echo in echo_numbers:
                with Pool(16) as pool:
                    args = [
                        (i, echo, input_imgs[echo][i], input_segs[echo][i], warps[i], template, subj, chunk, cwd)
                        for i in range(len(input_imgs[echo]))
                    ]
                    pool.map(process_single_image, args)

def step10(subjs, chunks, cwd):
    for subj in subjs:

        for chunk in chunks:
            # Replace the echo loop with parallel processing
            with Pool(3) as pool:  # Create 3 processes, one for each echo
                args = [(subj, chunk, echo) for echo in range(3)]
                pool.map(process_echo, args)

            print('majority voting')
            for echo in [1]:
                os.makedirs(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_averagepostregistration_segmentation/{chunk}/echo_{echo}/',
                    exist_ok=True
                )
                segs = natsorted(glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/*.nii.gz'))

                if len(segs) == 0:
                    print('No segmentations found for subject {} chunk {} echo {}'.format(subj, chunk, echo))
                    continue

                # Create majority vote image
                majority_vote_string = 'ImageMath 3 {} MajorityVoting '.format(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_averagepostregistration_segmentation/{chunk}/echo_{echo}/majority_vote.nii.gz'
                )

                for i in range(len(segs)):
                    majority_vote_string += f' {segs[i]} '

                # import pdb; pdb.set_trace()

                os.system(majority_vote_string)

def step11(subjs, chunks, cwd):
    for subj in subjs:

        for chunk in chunks:
            print(f'{subj} chunk {chunk}')
            # Load warps to apply:
            warps = natsorted(
                glob.glob(
                    cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_1/*.nii.gz'
                )
            )

            # Check if warps is empty and log to file if so
            if len(warps) == 0:
                # with open('empty_warps.log', 'a') as f:
                #    warp_path = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/'
                #    f.write(f"Empty warps directory: {warp_path}\n")
                continue

            # Create necessary directories
            savedir = cwd + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps_logdetjacs/{chunk}/echo_1/'
            os.makedirs(
                savedir,
                exist_ok=True
            )

            # Replace the for loop with parallel processing
            args = [(warp, savedir) for warp in warps]
            with Pool(16) as pool:
                pool.map(compute_jacobian, args)

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
    max_processes = int(input("Please enter the number of parallel processes you want to move forward with: "))
    # Currently 8
    subjects = natsorted(os.listdir(cwd + '/multiecho-images/'))
    chunks = step3(subjects, cwd, max_processes)
    # print("Step 3 complete")

    # ================================================================================
    # Step 4 - Segment all uninterleaved echoes
    # ================================================================================
    # step4(subjects, chunks, cwd)
    # print("Step 4 complete")

    # ================================================================================
    # Step 5 - Copy Independent bfc for echo 1
    # ================================================================================
    # step5(subjects, chunks, cwd)
    # print("Step 5 complete")

    # ================================================================================
    # Step 6 - Create image csv files
    # ================================================================================
    # step6(subjects, chunks, cwd)
    # print("Step 6 complete")

    # ================================================================================
    # Step 7 - Create atlasing calls
    # ================================================================================
    # step7(subjects, chunks, cwd)
    # print("Step 7 complete")

    # ================================================================================
    # Step 8 - Compose linear/nonlinear warps
    # ================================================================================
    step8(subjects, chunks, cwd)
    print("Step 8 complete")

    # ================================================================================
    # Step 9 - Warp other echoes
    # ================================================================================
    # step9(subjects, chunks, cwd)
    # print("Step 9 complete")

    # ================================================================================
    # Step 10 - Create averages and majority votes
    # ================================================================================
    # step10(subjects, chunks, cwd)
    # print("Step 10 complete")

    # ================================================================================
    # Step 11 - Compute lodgetjacs
    # ================================================================================
    # step11(subjects, chunks, cwd)
    # print("Step 11 complete")

if __name__ == "__main__":
    main()