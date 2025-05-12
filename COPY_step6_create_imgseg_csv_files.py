import numpy as np
import nibabel as nib
import os, glob,csv

from natsort import natsorted
from pdb import set_trace as st


subjs = natsorted(os.listdir(os.getcwd() + '/multiecho-images/'))

for subj in subjs:
    print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

    for chunk in chunks:

        # Create necessary directories
        for echo in [1]:
            os.makedirs(
                os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_{echo}/',
                exist_ok=True
            )

            imgs = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/echo_{echo}/*.nii.gz'))
            segs = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_segmentations/{chunk}/echo_{echo}/*.nii.gz'))

            assert len(imgs) == len(segs)
            assert len(imgs) > 0

            #st()

            with open(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/imgseg_paths.csv', 'w') as stream:
                writer = csv.writer(stream, quoting=csv.QUOTE_NONE, quotechar='',  lineterminator='\n')
                for img, seg in zip(imgs, segs):
                    writer.writerow([os.path.abspath(img), os.path.abspath(seg)])
        
            with open(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/atlas_build.sh', 'w') as stream:
                stream.write('antsMultivariateTemplateConstruction2.sh -d 3 -a 2 -k 2 -o {}/T_ -g 0.25  -j 8 -w 1x0.2  -n 0  -r 1  -i 4  -c 2  -m CC[2]  -m MSQ  -l 1  -t "SyN[0.3, 1.5, 0]" -q 100x100x80x60x30x30  -f 4x4x3x2x1x1  -s 2.5x2.0x1.5x1.0x0.5x0.0  -b 0  imgseg_paths.csv'.format(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc_atlased/{chunk}/echo_{echo}/'))


            #st()

