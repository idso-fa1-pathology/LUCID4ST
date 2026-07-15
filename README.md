# AI-guided spatial path-omics

This repository is for the manuscript "AI-guided spatial path-omics reveals pollutant-induced microenvironment reprogramming in lung adenocarcinoma". It guides you to generate PM (particulate-matter / anthracosis) deposit masks on whole-slide images with a well-trained SegFormer model for semantic segmentation.

The model is binary: every pixel is classified as either `background` (class 0) or `anthracosis` (class 1). The pipeline includes training (`lucid_train`), evaluation (`lucid_evaluation`), and inference (`lucid_inference`).

### 1. Training data

Download the training/testing set from Zenodo: <https://zenodo.org/records/17363430> (DOI `10.5281/zenodo.17363430`). Files are access-restricted for peer review; log in to Zenodo to request/confirm access.

`lucid_train/train_segformer.py` expects the extracted data as an `image/` folder of PNG patches and a matching `mask/` folder of `mask_<name>.png` label patches (each mask a single-channel PNG where anthracosis pixels are non-zero). You point the training script at these folders on the command line in step 4 (`-i`/`-l`).

### 2. Clone this repository

```         
git clone https://github.com/idso-fa1-pathology/LUCID4ST.git
cd LUCID4ST
```

The steps below assume commands are run from inside this repository folder unless stated otherwise.

### 3. Building the environment

```         
docker build -t lucid .
```

Mount this repo into the container at `/App` (the image's `WORKDIR`), and pass `--gpus all` so TensorFlow/PyTorch can see the GPU. You'll also need mounts for anything the container reads or writes that lives outside the repo -- the extracted training set (step 1), the raw slides you want to run inference on, and an output directory. Because the container is run with `--rm`, anything written to a path that isn't mounted is lost when the container exits, so the output mount in particular is required, not optional:

```         
docker run --gpus all -it --rm \
    -v /path/to/LUCID4ST:/App \
    -v /path/to/datasets:/data \
    -v /path/to/raw_slides:/slides \
    -v /path/to/output:/output \
    lucid
```

This drops you into a bash shell inside the container at `/App`, with `lucid_train`, `lucid_evaluation`, and `lucid_inference` directly available. The model saved by training lives inside the repo (`lucid_inference/model/...`), so it comes along for free with the `/App` mount -- no separate mount needed for it. Mount the parent folder that holds both datasets at `/data`, so `training_set/` and `testing_set/` sit side by side (`/data/training_set` and `/data/testing_set`). From there, run the commands in steps 4-6, using `/data/...` for the training and test data, `/slides` for the inference input, and `/output` for results.

### 4. Training

```         
cd ./lucid_train
```

Pass the data folders (from step 1) on the command line -- nothing in the script needs editing:

```         
/usr/bin/python3 train_segformer.py \
    -i /data/training_set/image \
    -l /data/training_set/mask \
    -o ../lucid_inference/model
```

`-i`/`--image_dir` is the training folder of PNG image patches, `-l`/`--mask_dir` the training folder of `mask_<name>.png` labels, and `-o`/`--output_dir` (default `../lucid_inference/model`) is where the fine-tuned model is written.

### 5. Evaluation

`evaluate.py` runs prediction and metric scoring in a single command over a test set laid out as `image/` (PNG patches) and `mask/` (ground-truth labels, named `mask_<name>.png` or `<name>.png`):

```         
cd ./lucid_evaluation
/usr/bin/python3 evaluate.py \
    -d /data/testing_set \
    -m ../lucid_inference/model/mit-b3-finetuned-anthracosis-e60-lr00001adam-s512 \
    -o /output
```

`-d` is the test-set root (containing `image/` and `mask/`), `-m` is the trained model, and `-o` is where results are written (defaults to `-d`). `-ps` is the model input size (default 512, matching training). It writes:

-   predicted masks to `<-o>/evaluation_mask/` (one PNG per input patch, same file name), and
-   `<-o>/evaluation_metrics.csv` with one row per patch: `file_name`, `dice`, `precision`, `recall`, `iou`, `accuracy`, and `gt_label` (the second-to-last token of the file name split by `_`).

### 6. Inference

Inference goes directly from raw whole-slide images to a stitched slide-level anthracosis mask with a single command. Nothing in the scripts needs to be edited -- the model checkpoint is passed on the command line with `-m`.

```         
cd ./lucid_inference
/usr/bin/python3 main_pgmn.py \
    -d /slides \
    -o /output \
    -m model/mit-b3-finetuned-anthracosis-e60-lr00001adam-s512 \
    -p "*.svs" \
    -ps 512 \
    -n 0 \
    -sf 0.125 \
    -nJ 1
```

`main_pgmn.py` runs three steps per slide, in order: (0) tile the slide into CWS tiles, (1) predict per tile with `predict_pgmn.py`, then (2) stitch to slide level with `ss1_stich.py`.

`-d` is the folder of raw slides and `-o` is a single output root -- `cws_tiling/`, `mask_cws/`, and `mask_ss1/` are created inside it automatically.

`-m` is the path to your trained model (the one saved by step 4, `lucid_inference/model/mit-b3-finetuned-anthracosis-e60-lr00001adam-s512`; relative to `lucid_inference` it is just `model/...`).

`-p` must match the slide file names (e.g. `"*.svs"`, `"*.ndpi"`).

`-ps` is the patch size (512, the extraction window) and `-sf` is the slide-level downscale factor.

`-n`/`-nJ` split the file list across parallel jobs (`-n` is the 0-based job index, `-nJ` the number of jobs; use `-n 0 -nJ 1` to process everything on one machine). Each job tiles, predicts, and stitches only its own slice of the file list -- slides are matched by name, so jobs can run concurrently against the same `-o` without interfering.

Output layout under `-o`:

```         
/output/cws_tiling/            CWS tiles (Da*.jpg, Ss1.jpg, param.p)
/output/mask_cws/              per-tile anthracosis prediction
/output/mask_ss1/              stitched slide-level mask
```

### 7. Color code of the output masks

The model has two classes, so each pixel in the output masks is colored by its predicted class (the `argmax` of the model's per-class logits):

| Class ID | Color (RGB)     | Tissue type |
|----------|-----------------|-------------|
| 0        | (0, 0, 0)       | background  |
| 1        | (255, 255, 255) | anthracosis |

Anthracosis (PM deposit) appears white on a black background in both the per-tile masks (`mask_cws`) and the stitched slide-level mask (`mask_ss1`).

### 8. Tumor boundary inference

The repository also includes a separate tumor boundary pipeline: a SegFormer (`nvidia/mit-b5`) model that segments the tumor boundary from H&E slides. Inference is a standalone step under `tube_inference/` and does not depend on the anthracosis model above. Lung pollutant index (LPI) at distinct compartment can therefore be calculated.

**Download the model checkpoint** from Hugging Face and place it under `tube_inference/model/` -- this exact path is what the `-m` flag below points to:

```         
pip install -U huggingface_hub
```

``` python
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

repo_id = "idso-fa1-pathology/<TUBE-MODEL-REPO>"   # <-- replace with the tumor boundary model repo
subfolder = "mit-b5-finetuned-tbed-s512-Ss1x1536"
# IMPORTANT: save it under this exact name/path -- it's what the -m flag below points to.
output_dir = Path("./tube_inference/model/mit-b5-finetuned-tbed-s512-Ss1x1536")
output_dir.mkdir(parents=True, exist_ok=True)

for filename in ["config.json", "tf_model.h5"]:
    downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, repo_type="model")
    shutil.copy(downloaded_file, output_dir / filename)
```

**Run inference:**

```         
cd ./tube_inference
/usr/bin/python3 main_tbed.py \
    -d /slides \
    -o /output \
    -m model/mit-b5-finetuned-tbed-s512-Ss1x1536 \
    -p "*.svs" \
    -t \
    -ps 1536 \
    -ins 512 \
    -nC 2 \
    -n 0 \
    -nJ 1
```

`main_tbed.py` loads the model once and, per slide, produces a slide-level tumor boundary mask. `-m` is the downloaded checkpoint (relative to `tube_inference` it is just `model/...`). `-t`/`--tiling` toggles CWS tiling: with `-t`, `-d` is a folder of raw slides and tiling (shared code in `lucid_inference/cws_tiling`) builds `Ss1` into `<-o>/cws_tiling/` first; without `-t`, `-d` must already contain `<slide>/Ss1.jpg`. `-ps` is the patch cropped from `Ss1`, `-ins` the model input size, `-nC` the number of classes, and `-n`/`-nJ` split the slide list across parallel jobs (matched by name). Tumor boundary masks are written to `<-o>/mask_tbed/`.

**Optional post-processing.** Add `-tme /path/to/tme_masks` to refine each tumor boundary mask against a TME tissue mask -- it drops alveoli/muscle/adipose regions, removes connected components smaller than 10000 px, and smooths the result. Refined masks are written to `<-o>/mask_tme_tbed/`. All post-processing hyperparameters are fixed in `tube_inference/post_process.py`; only the TME mask path is passed in.

Output layout under `-o`:

```         
/output/cws_tiling/       CWS tiles incl. Ss1.jpg (only with -t)
/output/mask_tbed/        slide-level tumor boundary mask (<slide>_tbed.png)
/output/mask_tme_tbed/    post-processed mask (only with -tme; <slide>_tme_tbed.png)
```
