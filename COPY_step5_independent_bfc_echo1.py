import numpy as np
import nibabel as nib
import os, glob

from natsort import natsorted
from pdb import set_trace as st


from skimage.morphology import binary_dilation
from scipy.ndimage import binary_fill_holes



subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

for subj in subjs:
    #print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

    for chunk in chunks:

        # Create necessary directories
        for echo in [1]:
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/echo_{echo}/',
                exist_ok=True
            )

            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_uterusmask/{chunk}/echo_{echo}/',
                exist_ok=True
            )

            imgs = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved/{chunk}/echo_{echo}/*.nii.gz'))
            segs = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/*.nii.gz'))

            assert len(imgs) == len(segs)
            assert len(imgs) > 0

            #st()

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
                    '{}/{}'.format(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_uterusmask/{chunk}/echo_{echo}/', os.path.basename(seg)),
                )
                #st()

                print('N4BiasFieldCorrection -r 0 -d 3 -i {} -o [{}/{}, placeholder.nii.gz] -w {}/{} -v -s 2 -c [400x400x400,0.00] {}'.format(
                    img,
                    os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/echo_{echo}/',
                    os.path.basename(img),
                    os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_uterusmask/{chunk}/echo_{echo}/',
                    os.path.basename(seg),
                    simult,
                ))
                #st()
    
