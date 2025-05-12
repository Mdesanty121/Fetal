import os
import shutil
from natsort import natsorted

from pdb import set_trace as st

# Base directory containing the subfolders
base_dir = './multiecho-images/'

# Iterate through each subfolder in the base directory
for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder, 'echoes')
    #st()

    if os.path.isdir(subfolder_path):
        
        # Get and natsort the files in the current subfolder
        files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        files = natsorted(files)

        assert len(files) % 3 == 0

        print('subfolder')
        
        # Chunk files into groups of 3
        for i in range(0, len(files), 3):
            chunk = files[i:i + 3]
            new_subfolder = os.path.join(subfolder_path, f'chunk_{i//3 + 1}')
            #st()

            os.makedirs(new_subfolder, exist_ok=True)
            
            # Move each file in the chunk to the new subfolder
            for file in chunk:
                src = os.path.join(subfolder_path, file)
                dest = os.path.join(new_subfolder, file)
                shutil.move(src, dest)

print('Files have been chunked and moved successfully.')
