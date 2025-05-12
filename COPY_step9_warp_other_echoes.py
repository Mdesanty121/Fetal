import numpy as np
import nibabel as nib
import os, glob
from multiprocessing import Pool

from natsort import natsorted

def process_single_image(args):
    i, echo, input_img, input_seg, warp, template = args
    output_img = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_registered/{chunk}/echo_{echo}/{os.path.basename(input_img)}'
    output_seg = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/{os.path.basename(input_seg)}'
    
    os.system(f'antsApplyTransforms -d 3 -e 0 -t {warp} -r {template} -i {input_img} -o {output_img} -n Linear -v')
    os.system(f'antsApplyTransforms -d 3 -e 0 -t {warp} -r {template} -i {input_seg} -o {output_seg} -n GenericLabel -v')
    
    #print(f'antsApplyTransforms -d 3 -e 0 -t {warp} -r {template} -i {input_img} -o {output_img} -n Linear -v')
    #print(f'antsApplyTransforms -d 3 -e 0 -t {warp} -r {template} -i {input_seg} -o {output_seg} -n GenericLabel -v')


    # Check if output files were created
    if not os.path.exists(output_img):
        #import pdb; pdb.set_trace()
        raise RuntimeError(f"Failed to create output image: {output_img}")
    if not os.path.exists(output_seg):
        #import pdb; pdb.set_trace()
        raise RuntimeError(f"Failed to create output segmentation: {output_seg}")


subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

for subj in subjs:
    #print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

    for chunk in chunks:

        # Create necessary directories and load input images/segmentations
        input_imgs = []
        input_segs = []

        for echo in range(3):
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_registered/{chunk}/echo_{echo}/',
                exist_ok=True
            )
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/',
                exist_ok=True
            )

            input_imgs.append(
                natsorted(
                    glob.glob(
                        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/*.nii.gz'
                    )
                )
            )
            input_segs.append(
                natsorted(
                    glob.glob(
                        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/*.nii.gz'
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
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_composedwarps/{chunk}/echo_1/*[!Inverse]Warp.nii.gz'
            )
        )

        # Check if warps is empty and log to file if so
        if len(warps) == 0:
            with open('empty_warps_for_other_echoes.log', 'a') as f:
                warp_path = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/'
                f.write(f"Empty warps directory: {warp_path}\n")
            continue

        assert len(warps) == len(input_imgs[0])

        template = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/T_template0.nii.gz'

        # Call antsApplyTransforms for each echo:
        for echo in range(3):
            with Pool(16) as pool:
                args = [(i, echo, input_imgs[echo][i], input_segs[echo][i], warps[i], template) 
                        for i in range(len(input_imgs[echo]))]
                pool.map(process_single_image, args)

