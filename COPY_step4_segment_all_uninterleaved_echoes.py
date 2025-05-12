import numpy as np
import nibabel as nib
import os, glob

from natsort import natsorted
from pdb import set_trace as st


subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

for subj in subjs:
    #print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

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
    
