# AI-guided spatial path-omics

This repository is for the manuscript "AI-guided spatial path-omics reveals pollutant-induced microenvironment reprogramming in lung adenocarcinoma". It guides you to generate PM (particulate-matter / anthracosis) deposit masks on whole-slide images with a well-trained SegFormer model for semantic segmentation.

The model is binary: every pixel is classified as either `background` (class 0) or `anthracosis` (class 1). The pipeline includes training (`lucid_train`), evaluation (lucid_evaluation), and inference (`lucid_inference`).

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
    -v /path/to/training_set:/data \
    -v /path/to/raw_slides:/slides \
    -v /path/to/output:/output \
    lucid
```

This drops you into a bash shell inside the container at `/App`, with `lucid_train`, `lucid_evaluation`, and `lucid_inference` directly available. The model saved by training lives inside the repo (`lucid_inference/model/...`), so it comes along for free with the `/App` mount -- no separate mount needed for it. From there, run the commands in steps 4-6, using `/data/...` for training data, `/slides` for the inference input, and `/output` for results.

### 4. Training

```         
cd ./lucid_train
```

Pass the data folders (from step 1) on the command line -- nothing in the script needs editing:

```         
/usr/bin/python3 train_segformer.py \
    -i /data/image \
    -l /data/mask \
    -o ../lucid_inference/model
```

`-i`/`--image_dir` is the training folder of PNG image patches, `-l`/`--mask_dir` the training folder of `mask_<name>.png` labels, and `-o`/`--output_dir` (default `../lucid_inference/model`) is where the fine-tuned model is written.

### 5. Evaluation

Evaluation runs the trained model on held-out test patches and scores the predictions with the Dice coefficient. Both scripts take no command-line arguments -- edit the paths near the top before running.

Generate patch-level predictions:

```         
cd ./lucid_evaluation
```

In `predict_patch_pgmn_segformer.py`, set `model_checkpoint` to your trained model (`lucid_inference/model/mit-b3-finetuned-anthracosis-e60-lr00001adam-s512`), and set `datapath` (test patches) and `save_dir` (where predicted masks are written), then run:

```         
/usr/bin/python3 predict_patch_pgmn_segformer.py
```

Score against ground truth:

```         
/usr/bin/python3 evaluation_pgmn.py
```

Edit `pred_mask` (the `save_dir` from the previous script), `gt_mask` (folder of `mask_<name>.png` ground-truth masks), and the output CSV path at the top of `evaluation_pgmn.py`. It writes a per-patch Dice table to that CSV.

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
