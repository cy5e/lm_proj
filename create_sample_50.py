import shutil
import os

DIR_ORIGINAL_DATASET = './train/'
DIR_AUG_DATA = './aug_data/'
DIR_SAVE = './sample_50/'

if not os.path.exists(DIR_SAVE):
    os.makedirs(DIR_SAVE)
with open('train50.txt') as f:
    for line in f:
        line = line.strip()
        split_idx = line.find('/')
        class_id = line[:split_idx]
        path_dest = DIR_SAVE + class_id
        if not os.path.exists(path_dest):
            os.makedirs(path_dest)
        img_id = line[split_idx+1:]
        if img_id.startswith('aug_'):
            path_to_file = DIR_AUG_DATA
        else:
            path_to_file = DIR_ORIGINAL_DATASET
        path_src += line
        shutil.copyfile(path_src, path_dest)