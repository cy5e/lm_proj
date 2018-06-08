#import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
#import tarfile

import tensorflow as tf
#from tensorflow.python.framework import graph_util
#from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

## python get_valid_test ./image_dir valid_percent test_precent out_file_tag
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage=0, validation_percentage=0):
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = collections.OrderedDict()
  tr_ind_f = open('/home/chienyi/landmark_data/data/sample_train.txt','r')
  de_ind_f = open('/home/chienyi/landmark_data/data/sample_val.txt','r')
  te_ind_f = open('/home/chienyi/landmark_data/data/sample_test.txt','r')

  for line in tr_ind_f.readlines():
      label, img_name = line.strip().split('/')
      if label not in result:
          result[label] = {'dir':label,'training':[],'testing':[],'validation':[]}
      result[label]['training'].append(img_name)
  for line in de_ind_f.readlines():
      label, img_name = line.strip().split('/')
      result[label]['validation'].append(img_name)
  for line in te_ind_f.readlines():
      label, img_name = line.strip().split('/')
      result[label]['testing'].append(img_name)
      
  return result


if __name__ == '__main__':
    image_lists = create_image_lists(sys.argv[1], int(sys.argv[3]), int(sys.argv[2]))
    print(image_lists)




            
        
        
