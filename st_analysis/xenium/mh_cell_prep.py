#!/usr/bin/env python3
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None  

RES_UM = 0.2125  # µm per pixel



def draw_and_save_quadrat_grid(image, q_pix, out_path="quadrat_grid_overlay.png", color=(255,255,255), width=1):
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        image_pil = Image.fromarray(image)
    else:
        image_pil = image

    draw = ImageDraw.Draw(image_pil)
    W, H = image_pil.size

    # Draw vertical lines
    for x in range(0, W, q_pix):
        draw.line([(x, 0), (x, H-1)], fill=color, width=width)
    # Draw horizontal lines
    for y in range(0, H, q_pix):
        draw.line([(0, y), (W-1, y)], fill=color, width=width)

    image_pil.save(out_path)
    print(f"Quadrat overlay saved to {out_path}")


def rgb_eq(img: np.ndarray, rgb: tuple) -> np.ndarray:
    r, g, b = rgb
    return (img[..., 0] == r) & (img[..., 1] == g) & (img[..., 2] == b)

def main(csv_path, mask_path, png_path, qedge_um, out_csv, overlay_path):
    print("Loading CSV and mask image...")
    cells = pd.read_csv(csv_path)
    mask_tif  = imread(mask_path)
    mask_png = imread(png_path)
    
    print("Resizing PNG mask to match TIF mask shape...")
    mask = resize(
        mask_png,
        mask_tif.shape,
        order=0,           # Nearest-neighbor interpolation (preserves labels/colors)
        preserve_range=True,
        anti_aliasing=False
    ).astype(mask_png.dtype)  # Cast back to original dtype (usually uint8)

    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError("Mask must be an RGB image of shape (H, W, 3)")
    H, W, _ = mask.shape

    print("Building RGB pixel masks...")
    anth_mask    = rgb_eq(mask, (255, 255, 255))
    tbed_mask    = rgb_eq(mask, (135, 133, 186))
    nontbed_mask = rgb_eq(mask, (0, 128, 0))

    print("Converting cell coordinates (µm → pixel)...")
    cells["ix"] = np.clip((cells["x_centroid"] / RES_UM).round(), 0, W-1).astype(int)
    cells["iy"] = np.clip((cells["y_centroid"] / RES_UM).round(), 0, H-1).astype(int)

    print("Assigning quadrat indices for cells...")
    q_pix = int(round(qedge_um / RES_UM))
    cells["qx"] = cells["ix"] // q_pix
    cells["qy"] = cells["iy"] // q_pix

    # Crop everything to full quadrat blocks
    h_crop = (H // q_pix) * q_pix
    w_crop = (W // q_pix) * q_pix
    cells = cells[(cells["ix"] < w_crop) & (cells["iy"] < h_crop)].copy()

    draw_and_save_quadrat_grid(mask, q_pix, overlay_path)

    print("Assigning per-cell flags (anthracosis, tumor bed)...")
    cells["is_anth"] = anth_mask[cells["iy"], cells["ix"]]
    cells["is_tbed"] = tbed_mask[cells["iy"], cells["ix"]]

    print("Computing per-quadrat pixel counts (fast numpy)...")
    anth_mask_c    = anth_mask[:h_crop, :w_crop]
    tbed_mask_c    = tbed_mask[:h_crop, :w_crop]
    nontbed_mask_c = nontbed_mask[:h_crop, :w_crop]

    nqy = h_crop // q_pix
    nqx = w_crop // q_pix

    def block_sum(mask_bin):
        return mask_bin.reshape(nqy, q_pix, nqx, q_pix).sum(axis=(1, 3))

    n_anth  = block_sum(anth_mask_c)
    tbed_px = block_sum(tbed_mask_c)
    ntbed_px= block_sum(nontbed_mask_c)

    total_px = n_anth + tbed_px + ntbed_px
    is_tbed = np.full_like(tbed_px, fill_value=-1)  # initialize with -1

    # Now assign proper values where there are valid mask pixels
    valid_mask = total_px > 0
    is_tbed[valid_mask] = (tbed_px[valid_mask] >= ntbed_px[valid_mask]).astype(int)

    area_df = pd.DataFrame({
        "qx": np.repeat(np.arange(nqx), nqy),
        "qy": np.tile(np.arange(nqy), nqx),
        "anthracosis_pix": n_anth.T.flatten(),
        "is_tbed": is_tbed.T.flatten(),
    })

    print("Summarizing per-quadrat, per-cell-type counts (wide format)...")
    # Use the column name in your CSV: 'cell_label' or 'cell_type'
    cell_col = "cell_label" if "cell_label" in cells.columns else "cell_type"
    ct_wide = (
        cells.groupby(["qx","qy",cell_col], observed=True)
             .size()
             .reset_index(name="n_cells")
             .pivot_table(index=["qx","qy"],
                          columns=cell_col,
                          values="n_cells",
                          fill_value=0,
                          aggfunc="sum")
             .rename(columns=lambda c: f"{c}")
             .reset_index()
    )

    print("Merging all tables and writing output CSV...")
    out = (
        ct_wide
        .merge(area_df, on=["qx","qy"], how="left")
        .fillna({"anthracosis_pix": 0, "is_tbed": -1})
        .astype({"is_tbed": int})
    )
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    # 5k
    #cells_csv = "/path/10x_xenium/0_cell/5k/xenium_5k_Tcell_type_visium.csv"
    #mask_tif  = "/pat/10x_xenium/raw_register_v2/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_registered.ome.tif"
    #mask_png = "/path/10x_xenium/fullresoverlay_pgmn_alveoli_tbedraw_remove160000/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_coregistered_pyramid.tif_alveoli_tbed.png"
    
    # v1
    cells_csv = "/path/10x_xenium/0_cell/v1/xenium_v1_cell_type_marker.csv"
    mask_tif  = "/path/10x_xenium/raw_register_v2/Xenium_V1_humanLung_Cancer_FFPE_he_image_registered.ome.tif"
    mask_png = "/path/10x_xenium/fullresoverlay_pgmn_alveoli_tbedraw_remove160000/Xenium_V1_humanLung_Cancer_FFPE_he_image_coregistered_pyramid.tif_alveoli_tbed.png"
    
    qedge_um  = 20
    out_csv   = f"/path/10x_xenium/0_cell/v1/quadrat{qedge_um}um_counts.csv"
    overlaypath = f"/path/10x_xenium/0_cell/v1/quadrat{qedge_um}um_overlay.jpg"
    main(cells_csv, mask_tif, mask_png, qedge_um, out_csv, overlaypath)
    
