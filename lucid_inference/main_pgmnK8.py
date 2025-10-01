import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import math
from glob import glob

from predict_pgmn_hpc import generate_pgmn
from ss1_stich import ss1_stich


######step0: generate cws tiles from single-cell pipeline

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', help='path to cws data')
parser.add_argument('-o', '--save_dir', dest='save_dir', help='path to save all output files', default=None)
parser.add_argument('-s', '--save_dir_ss1', dest='save_dir_ss1', help='path to save all ss1 files', default=None)
parser.add_argument('-p', '--pattern', dest='file_name_pattern', help='pattern in the files name', default='*.ndpi')
parser.add_argument('-n', '--nfile', dest='nfile', help='the nfile-th file', default=0, type=int)
parser.add_argument('-ps', '--patch_size', dest='patch_size', help='the size of the input', default=768, type=int)
parser.add_argument('-sf', '--scale_factor', dest='scale', help='how many times to scale compared to x20', default=0.0625, type=float)
parser.add_argument('-nJ', '--number_pods', dest='nJob', help='how many pods to be used in K8s', default=32, type=int)
parser.add_argument('-b2b', '--brown_to_black', dest='brown2balck', help='wheter to convert brown color to black', action='store_true')
args = parser.parse_args()

#If a user runs the script with python script.py --brown_to_black, 
#then the brown2black attribute within the namespace returned by parser.parse_args() will be set to True.
#If the script is run without the -b2b or --brown_to_black flags, 
#then brown2black will be False, indicating that the conversion from brown to black should not occur.

datapath=args.data_dir
nfile=args.nfile
file_pattern=args.file_name_pattern
files = sorted(glob(os.path.join(datapath, file_pattern)))
njob = args.nJob

if len(files) <= njob:
        start_file = nfile
        end_file = nfile + 1
else:
        file_job = len(files) // njob +1
        start_file = nfile * file_job
        end_file = nfile * file_job + file_job

for i in range(start_file, end_file):
    ######step1: generate growth pattern for tiles
    generate_pgmn(datapath=args.data_dir, save_dir=args.save_dir, file_pattern=args.file_name_pattern, nfile=i,
                patch_size=args.patch_size, patch_stride=192, nClass=2, brown_black=args.brown2balck)

    #######step2: stich to ss1 level
    ss1_stich(cws_folder=args.data_dir, annotated_dir=args.save_dir, output_dir=args.save_dir_ss1,  nfile=i, file_pattern=args.file_name_pattern, downscale=args.scale)
