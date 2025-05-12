import numpy as np
import nibabel as nib
import os, glob
from multiprocessing import Pool

from natsort import natsorted


def compose_transform(args):
    warp, linear, savedir, subj, chunk = args
    output_path = os.path.join(savedir, os.path.basename(warp))
    template_path = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/T_template0.nii.gz'
    
    cmd = f'ComposeMultiTransform 3 {output_path} -R {template_path} {warp} {linear}'
    os.system(cmd)


subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

for subj in subjs:
    #print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

    for chunk in chunks:

        # Create necessary directories
        for echo in [1]:
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_{echo}/',
                exist_ok=True
            )

        # Load input images/echoes to warp
        input_e0 = natsorted(
            glob.glob(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_0/*.nii.gz'
            )
        )


        # Load warps to apply:
        warps = natsorted(
            glob.glob(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/*[!Inverse]Warp.nii.gz'
            )
        )
        
        # Check if warps is empty and log to file if so
        if len(warps) == 0:
            with open('empty_warps.log', 'a') as f:
                warp_path = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/'
                f.write(f"Empty warps directory: {warp_path}\n")
            continue
            
        linears = natsorted(
            glob.glob(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/*GenericAffine.mat'
            )
        )
        del linears[-1]

        #import pdb; pdb.set_trace()

        assert len(input_e0) > 0
        assert len(warps) == len(input_e0)
        assert len(warps) == len(linears)

        for echo in [1]:
            savedir = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_{echo}/'
            os.makedirs(
                savedir,
                exist_ok=True
            )
            with Pool(16) as pool:
                args = [(warps[k], linears[k], savedir, subj, chunk) for k in range(len(warps))]
                pool.map(compose_transform, args)




