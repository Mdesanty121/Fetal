import numpy as np
import nibabel as nib
import os, glob

from natsort import natsorted
from pdb import set_trace as st
from concurrent.futures import ProcessPoolExecutor


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
        #if nnunet_prep:
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


subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

# User-defined number of parallel processes
max_processes = 8  # Adjust this number based on your system resources

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

# Function to run split_nii
def run_split_nii(subj, chunk, echo_idx):
    split_nii(
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split/{chunk}/echo_{echo_idx}/',
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo_idx}/',
        True, True,
    )


for subj in subjs:
    print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]
    
    #breakpoint()

    for chunk in chunks:
        # Create necessary directories
        for echo in range(3):
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split/{chunk}/echo_{echo}/',
                exist_ok=True
            )
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/',
                exist_ok=True
            )
        #st()
        # Load combined images
        combined_images = natsorted(glob.glob(
            os.getcwd() + f'/multiecho-images/{subj}/echoes/{chunk}/*.nii.gz'
        ))
        assert len(combined_images) == 3

        nii_files = [nib.load(img) for img in combined_images]
        nii_data = [nii.get_fdata() for nii in nii_files]
        nii_affines = [nii.affine for nii in nii_files]
        #st()

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

        """
        # Save split echoes
        for echo_idx, (arr, aff) in enumerate(zip(nii_data, nii_affines)):
            for i in range(arr.shape[-1]):
                print(f"Chunk: {chunk}, Echo: {echo_idx}, Volume: {i}")
                nib.save(
                    nib.Nifti1Image(arr[..., i], aff),
                    os.getcwd() + f'/multiecho-images/{subj}/echoes_split/{chunk}/echo_{echo_idx}/{subj}_{i:04d}.nii.gz'
                )

        # Split NIfTI files
        for echo_idx in range(3):
            split_nii(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split/{chunk}/echo_{echo_idx}/',
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo_idx}/',
                True
            )
        """

