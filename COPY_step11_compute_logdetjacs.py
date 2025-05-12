import numpy as np
import nibabel as nib
import os, glob
from multiprocessing import Pool

from natsort import natsorted

import shutil


def compute_jacobian(args):
    warp, savedir = args
    cmd = 'CreateJacobianDeterminantImage 3 {} {}/{}.nii.gz 1 1'.format(
        warp, savedir, os.path.basename(warp),
    )
    os.system(cmd)


subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

for subj in subjs:
    #print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

    #import pdb; pdb.set_trace()

    for chunk in chunks:
        print(f'{subj} chunk {chunk}')
        # Load warps to apply:
        warps = natsorted(
            glob.glob(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_1/*.nii.gz'
            )
        )

        # Check if warps is empty and log to file if so
        if len(warps) == 0:
            #with open('empty_warps.log', 'a') as f:
            #    warp_path = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/'
            #    f.write(f"Empty warps directory: {warp_path}\n")
            continue
        
        # Create necessary directories
        savedir = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps_logdetjacs/{chunk}/echo_1/'
        os.makedirs(
            savedir,
            exist_ok=True
        )
        
        # Replace the for loop with parallel processing
        args = [(warp, savedir) for warp in warps]
        with Pool(16) as pool:
            pool.map(compute_jacobian, args)

