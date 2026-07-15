import os
import sys
import argparse
from glob import glob

from predict_slide_tbed import generate_tbed, load_segmentation_model
from post_process import post_process, find_tme_mask


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', help='raw slides (with --tiling) or CWS data containing <slide>/Ss1.jpg (without --tiling)')
parser.add_argument('-o', '--save_dir', dest='save_dir', help='root output dir; mask_tbed/ (and cws_tiling/ when --tiling) are created inside it', default=None)
parser.add_argument('-m', '--model_checkpoint', dest='model_checkpoint', required=True, help='path to the trained tumor-bed model checkpoint')
parser.add_argument('-p', '--pattern', dest='file_name_pattern', help='pattern in the files name', default='*.ndpi')
parser.add_argument('-n', '--nfile', dest='nfile', help='the nfile-th file', default=0, type=int)
parser.add_argument('-ps', '--patch_size', dest='patch_size', help='the size of the patch cropped from Ss1', default=1536, type=int)
parser.add_argument('-ins', '--input_size', dest='input_size', help='the size of the model input', default=512, type=int)
parser.add_argument('-nC', '--number_class', dest='nClass', help='how many classes to segment', default=2, type=int)
parser.add_argument('-nJ', '--number_jobs', dest='nJob', help='how many parallel jobs to split the file list across', default=1, type=int)
parser.add_argument('-t', '--tiling', dest='tiling', help='run CWS tiling first (from lucid_inference/cws_tiling) to build Ss1 from raw slides', action='store_true')
parser.add_argument('-tme', '--tme_dir', dest='tme_dir', help='folder of TME tissue masks; when given, post-processing refines each bed mask (writes mask_tme_tbed/)', default=None)
args = parser.parse_args()


datapath = args.data_dir
nfile = args.nfile
file_pattern = args.file_name_pattern

# output sub-directories derived from the single --save_dir root
mask_dir = os.path.join(args.save_dir, 'mask_tbed')
cws_dir = os.path.join(args.save_dir, 'cws_tiling')
post_dir = os.path.join(args.save_dir, 'mask_tme_tbed')

files = sorted(glob(os.path.join(datapath, file_pattern)))
njob = args.nJob

if len(files) <= njob:
        start_file = nfile
        end_file = nfile + 1
else:
        file_job = len(files) // njob + 1
        start_file = nfile * file_job
        end_file = nfile * file_job + file_job


job_files = files[start_file:end_file]

######step0 (optional): tiling. If enabled, build CWS tiles (incl. Ss1) from the raw
# slides using the shared tiler in lucid_inference/cws_tiling; otherwise the input is
# assumed to already contain <slide>/Ss1.jpg and we run tube inference directly.
if args.tiling:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lucid_inference', 'cws_tiling'))
    from main_tiles import run_generate_cws

    for wsi in job_files:
        wsi_name = os.path.basename(wsi)
        if os.path.exists(os.path.join(cws_dir, wsi_name, 'Ss1.jpg')):
            print('CWS tiles already exist for %s, skipping tiling' % wsi_name)
            continue
        run_generate_cws(wsi_input=wsi, output_dir=cws_dir, out_mpp=0.22,
                         file_name_pattern=file_pattern, parallel=False)
    infer_dir = cws_dir
else:
    infer_dir = datapath


model = load_segmentation_model(args.model_checkpoint)

for wsi in job_files:
    wsi_name = os.path.basename(wsi)
    ######step1: generate the slide-level tumor-bed mask (resolved by name, not by index)
    generate_tbed(datapath=infer_dir, save_dir=mask_dir, model=model, file_pattern=file_pattern, file_name=wsi_name,
                  patch_size=args.patch_size, patch_stride=512, input_size=args.input_size, nClass=args.nClass)

    ######step2 (optional): post-process the bed mask against the TME tissue mask
    if args.tme_dir:
        stem = os.path.splitext(wsi_name)[0]
        tbed_png = os.path.join(mask_dir, stem + '_tbed.png')
        tme_png = find_tme_mask(args.tme_dir, stem)
        if os.path.exists(tbed_png) and tme_png is not None:
            post_process(tbed_png, tme_png, os.path.join(post_dir, stem + '_tme_tbed.png'))
        else:
            print('[post] skip %s (missing bed mask or TME mask)' % stem)
