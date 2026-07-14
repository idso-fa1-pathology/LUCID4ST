import os
import sys
import argparse
from glob import glob

from predict_pgmn import generate_pgmn, load_segmentation_model
from ss1_stich import ss1_stich

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cws_tiling'))
from main_tiles import run_generate_cws


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', help='path to raw whole-slide images')
parser.add_argument('-o', '--save_dir', dest='save_dir', help='root output dir; cws_tiling/, mask_cws/ and mask_ss1/ are created inside it', default=None)
parser.add_argument('-m', '--model_checkpoint', dest='model_checkpoint', help='path to the trained model checkpoint used for prediction', required=True)
parser.add_argument('-p', '--pattern', dest='file_name_pattern', help='pattern in the files name', default='*.ndpi')
parser.add_argument('-n', '--nfile', dest='nfile', help='the nfile-th file', default=0, type=int)
parser.add_argument('-ps', '--patch_size', dest='patch_size', help='the size of the input', default=512, type=int)
parser.add_argument('-sf', '--scale_factor', dest='scale', help='how many times to scale compared to x20', default=0.0625, type=float)
parser.add_argument('-nJ', '--number_jobs', dest='nJob', help='how many parallel jobs to split the file list across', default=1, type=int)
parser.add_argument('-b2b', '--brown_to_black', dest='brown2balck', help='whether to convert brown color to black', action='store_true')
args = parser.parse_args()


datapath = args.data_dir
nfile = args.nfile
file_pattern = args.file_name_pattern

# output sub-directories, all derived from the single --save_dir root
cws_dir = os.path.join(args.save_dir, 'cws_tiling')
mask_dir = os.path.join(args.save_dir, 'mask_cws')
ss1_dir = os.path.join(args.save_dir, 'mask_ss1')

files = sorted(glob(os.path.join(datapath, file_pattern)))
njob = args.nJob

if len(files) <= njob:
        start_file = nfile
        end_file = nfile + 1
else:
        file_job = len(files) // njob +1
        start_file = nfile * file_job
        end_file = nfile * file_job + file_job

# this job's own slice of the raw slide list (slicing safely clamps to len(files))
job_files = files[start_file:end_file]

# load the model once and reuse it for every slide in this job
model = load_segmentation_model(args.model_checkpoint)

for wsi in job_files:
    wsi_name = os.path.basename(wsi)

    ######step0: tile this slide (skip if already tiled)
    if os.path.exists(os.path.join(cws_dir, wsi_name, 'Ss1.jpg')):
        print('CWS tiles already exist for %s, skipping tiling' % wsi_name)
    else:
        run_generate_cws(wsi_input=wsi, output_dir=cws_dir, out_mpp=0.22,
                         file_name_pattern=file_pattern, parallel=False)

    ######step1: generate pgmn masks for tiles
    generate_pgmn(datapath=cws_dir, save_dir=mask_dir, model=model, file_pattern=file_pattern, file_name=wsi_name,
                patch_size=args.patch_size, patch_stride=192, nClass=2, brown_black=args.brown2balck)

    #######step2: stich to ss1 level
    ss1_stich(cws_folder=cws_dir, annotated_dir=mask_dir, output_dir=ss1_dir, file_name=wsi_name, file_pattern=file_pattern, downscale=args.scale)
