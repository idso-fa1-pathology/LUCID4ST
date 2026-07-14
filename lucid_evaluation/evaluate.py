import os
import argparse
from glob import glob

import numpy as np
import cv2
from PIL import Image
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSemanticSegmentation


# openCV: BGR -- background, anthracosis
class_colors = [(0, 0, 0), (255, 255, 255)]


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height, output_width = seg_arr.shape
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')
    return seg_img


def binary_metrics(pred, gt):
    """Per-patch metrics for the foreground (anthracosis) class."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    dice = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 1.0
    return {'dice': dice, 'precision': precision, 'recall': recall, 'iou': iou, 'accuracy': accuracy}


def find_mask(mask_dir, stem):
    """Match an image stem to its ground-truth mask (mask_<stem>.png or <stem>.png)."""
    for cand in ('mask_%s.png' % stem, '%s.png' % stem):
        path = os.path.join(mask_dir, cand)
        if os.path.exists(path):
            return path
    return None


def predict_patch(model, img, patch_size):
    """Resize a patch to patch_size, run the model, and return a per-pixel class map at the original size."""
    x = cv2.resize(img, (patch_size, patch_size)).astype(np.float32) / 255.0
    x = tf.expand_dims(x, axis=0)
    x = tf.transpose(x, (0, 3, 1, 2))
    out = model.predict(x)
    logit = tf.transpose(out.logits, (0, 2, 3, 1))
    logit = np.array(tf.squeeze(logit, axis=0))
    logit = cv2.resize(logit, (img.shape[1], img.shape[0]))
    return logit.argmax(axis=2)


def main(data_dir, model_checkpoint, output_dir, patch_size):
    image_dir = os.path.join(data_dir, 'image')
    mask_dir = os.path.join(data_dir, 'mask')
    pred_dir = os.path.join(output_dir, 'evaluation_mask')
    os.makedirs(pred_dir, exist_ok=True)

    model = TFAutoModelForSemanticSegmentation.from_pretrained(model_checkpoint)

    rows = []
    images = sorted(glob(os.path.join(image_dir, '*.png')))
    print('Found %d test images in %s' % (len(images), image_dir))
    for img_path in images:
        file_name = os.path.basename(img_path)
        stem = os.path.splitext(file_name)[0]

        img = np.array(Image.open(img_path).convert('RGB'))
        pred = predict_patch(model, img, patch_size)
        cv2.imwrite(os.path.join(pred_dir, file_name), get_colored_segmentation_image(pred, len(class_colors)))

        row = {'file_name': file_name}
        mask_path = find_mask(mask_dir, stem)
        if mask_path is not None:
            gt = cv2.imread(mask_path)[:, :, 0]
            gt_bin = (gt > 127).astype(np.uint8)
            row.update(binary_metrics(pred.astype(np.uint8), gt_bin))
        else:
            print('[WARN] no ground-truth mask for %s; metrics left blank' % file_name)
            row.update({'dice': np.nan, 'precision': np.nan, 'recall': np.nan, 'iou': np.nan, 'accuracy': np.nan})

        # gt_label = second-to-last token when the file name (without extension) is split by '_'
        parts = stem.split('_')
        row['gt_label'] = parts[-2] if len(parts) >= 2 else ''
        rows.append(row)

    df = pd.DataFrame(rows, columns=['file_name', 'dice', 'precision', 'recall', 'iou', 'accuracy', 'gt_label'])
    csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    df.to_csv(csv_path, index=False)
    print('Wrote %d rows to %s' % (len(df), csv_path))
    if len(df):
        print('Mean metrics:')
        print(df[['dice', 'precision', 'recall', 'iou', 'accuracy']].mean(numeric_only=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', dest='data_dir', required=True, help='testing set root containing image/ and mask/ subfolders')
    parser.add_argument('-m', '--model_checkpoint', dest='model_checkpoint', required=True, help='path to the trained model checkpoint')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default=None, help='where evaluation_mask/ and evaluation_metrics.csv are written (default: data_dir)')
    parser.add_argument('-ps', '--patch_size', dest='patch_size', type=int, default=512, help='model input size (matches training)')
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.data_dir
    main(args.data_dir, args.model_checkpoint, output_dir, args.patch_size)
