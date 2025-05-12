import numpy as np
import os, glob, csv

from natsort import natsorted
from pdb import set_trace as st


base_dir = '/net/rc-fs-nfs.tch.harvard.edu/FNNDSC-e2/neuro/labs/grantlab/research/uterus_data/megan/registration_Neel/multiecho-images/'
subjs = [os.path.basename(filename) for filename in glob.glob(os.path.join(base_dir, "sub-MAP-*"))]


for subj in subjs:
    #print(subj)
    chunks_fullpaths = natsorted(glob.glob(os.getcwd() + f'/multiecho-images/{subj}/echoes/chunk*'))
    chunks = [os.path.basename(i) for i in chunks_fullpaths]

    for chunk in chunks:
        for echo in [1]:
            print('cd {}; sh atlas_build.sh'.format(os.getcwd() + f'/multiecho-images/{subj}/echoes_split_uninterleaved_indepbfc/{chunk}/'))
