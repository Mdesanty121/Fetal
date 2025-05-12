import numpy as np
import nibabel as nib
import os, glob
from multiprocessing import Pool

from natsort import natsorted


subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

def process_echo(args):
    subj, chunk, echo = args
    
    os.makedirs(
        os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_averagepostregistration/{chunk}/echo_{echo}/',
        exist_ok=True
    )

    print('Processing subject {} chunk {} echo {}'.format(subj, chunk, echo))

    imgs = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_registered/{chunk}/echo_{echo}/*.nii.gz'))
    segs = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/*.nii.gz'))
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

for subj in subjs:
    #print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

    #import pdb; pdb.set_trace()

    for chunk in chunks:
        # Replace the echo loop with parallel processing
        with Pool(3) as pool:  # Create 3 processes, one for each echo
            args = [(subj, chunk, echo) for echo in range(3)]
            pool.map(process_echo, args)

        print('majority voting')
        for echo in [1]:
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_averagepostregistration_segmentation/{chunk}/echo_{echo}/',
                exist_ok=True
            )
            segs = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations_registered/{chunk}/echo_{echo}/*.nii.gz'))

            if len(segs) == 0:
                print('No segmentations found for subject {} chunk {} echo {}'.format(subj, chunk, echo))
                continue

            # Create majority vote image
            majority_vote_string = 'ImageMath 3 {} MajorityVoting '.format(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_averagepostregistration_segmentation/{chunk}/echo_{echo}/majority_vote.nii.gz'
            )

            for i in range(len(segs)):
                majority_vote_string += f' {segs[i]} '

            #import pdb; pdb.set_trace()

            os.system(majority_vote_string)

        """
        template = os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_1/T_template0.nii.gz'
        """

