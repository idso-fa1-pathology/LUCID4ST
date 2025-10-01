# AI-guided spatial path-omics
This repository is for the manuscript "AI-guided spatial path-omics reveals pollutant-induced microenvironment reprogramming in lung adenocarcinoma" submitted to Nature Cancer. It could guide you to generate PM deposit masks with a well-trained AI model for semantic segmentation.

### Generating tiles for whole slide images
Please follow instructions in https://github.com/xi11/AIgrading/generating_tiles.

### Building env with Dockerfile

### Training
```
cd ./lucid_train
python train_segformer.py
```

### Inference
```
cd ./lucid_inference
python main_pgmnK8.py \
    -d /path/to/demo \
    -o /path/to/demo/mit-b3-finetuned-anthracosis-e60-lr00001adam-s512/mask_cws \
    -s /path/to/demo/mit-b3-finetuned-anthracosis-e60-lr00001adam-s512/mask_ss1 \ 
    -p "ima*" \  #to adjust
    -ps 512 \
    -n {{index}} \
    -sf 0.125 \
    -nJ 32

```