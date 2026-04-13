import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pycocotools import mask as coco_mask
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from config import ANNOTATIONS_TRAIN, ANNOTATIONS_TEST
# Output folder for the heatmap figures
OUTPUT_DIR=Path("./fashionpedia_heatmaps")
OUTPUT_DIR.mkdir(exist_ok=True)
#normalise all masks into
CANVAS_H =256
CANVAS_W =256

def load_annotations(json_path: str) -> tuple[dict, dict, dict]:
    """Returns (categories, images, annotations) dicts keyed by their ids."""
    print(f"Loading {json_path} …")
    with open(json_path) as f:
        data=json.load(f)

    categories ={c["id"]: c for c in data["categories"]}
    images={i["id"]: i for i in data["images"]}
    annotations=data["annotations"]
    return categories,images,annotations

def build_density_maps(annotations:list[dict],images:dict,categories:dict) ->dict[int, np.ndarray]:
    density: dict[int, np.ndarray] = defaultdict(lambda: np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32))
    counts: dict[int, int] = defaultdict(int)

    for ann in tqdm(annotations, desc="Processing annotations"):
        cat_id  = ann["category_id"]
        img_meta = images.get(ann["image_id"])
        if img_meta is None:
            continue

        img_h, img_w = img_meta["height"], img_meta["width"]
        seg = ann.get("segmentation")
        if not seg:
            x, y, bw, bh = ann["bbox"]
            binary = np.zeros((img_h, img_w), dtype=np.uint8)
            x1, y1 = int(x), int(y)
            x2, y2 = min(int(x + bw), img_w), min(int(y + bh), img_h)
            binary[y1:y2, x1:x2] = 1
        else:
            if isinstance(seg, dict):
                rle = seg
            else:
                rle = coco_mask.frPyObjects(seg, img_h, img_w)
                rle = coco_mask.merge(rle)
            binary = coco_mask.decode(rle)   

        # Resize to canvas using block-averaging
        from skimage.transform import resize as sk_resize
        small = sk_resize(
            binary.astype(np.float32),
            (CANVAS_H, CANVAS_W),
            anti_aliasing=True,
            preserve_range=True,
        )
        density[cat_id]+=small
        counts[cat_id]+=1
    result = {}
    for cat_id, heatmap in density.items():
        result[cat_id] = heatmap / (heatmap.max() + 1e-8)
    return result, counts

CMAP="inferno"
def plot_all_heatmaps_grid(density_maps: dict[int, np.ndarray],categories: dict,counts: dict,title: str,save_path: Path,ncols: int = 6):
    """Plot every category in a single large grid figure."""
    cat_ids=sorted(density_maps.keys())
    n =len(cat_ids)
    nrows =(n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*2.8,nrows*3.2),facecolor="#0d0d0d",)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, cat_id in zip(axes_flat, cat_ids):
        hm=density_maps[cat_id]
        name=categories[cat_id]["name"]
        n_inst=counts[cat_id]

        im=ax.imshow(hm, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{name}\n(n={n_inst})",fontsize=7,color="#f0e6d3",pad=3)
        ax.axis("off")

        # Thin colourbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5, colors="#888")
        cbar.outline.set_edgecolor("#333")

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title,fontsize=16,color="#f0e6d3",y=1.01,fontweight="bold")
    plt.tight_layout(pad=1.2)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_single_heatmap(heatmap: np.ndarray,category_name: str,n_instances: int,save_path: Path):
    """High-resolution single-category heatmap with overlaid contours."""
    fig,ax= plt.subplots(figsize=(6, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im =ax.imshow(heatmap, cmap=CMAP, vmin=0, vmax=1)
    levels =[0.25, 0.5, 0.75]
    cs =ax.contour(heatmap, levels=levels, colors="white", linewidths=0.6, alpha=0.5)
    ax.clabel(cs, fmt={l: f"{int(l*100)}%" for l in levels}, fontsize=7, colors="white")
    for frac in [0.25, 0.5, 0.75]:
        ax.axhline(CANVAS_H * frac, color="#444", lw=0.5, ls="--")
        ax.axvline(CANVAS_W * frac, color="#444", lw=0.5, ls="--")

    ax.set_title(f"{category_name}  ·  {n_instances} instances",color="#f0e6d3",fontsize=13,pad=10,)
    ax.set_xlabel("← left  ·  image width  ·  right →", color="#888", fontsize=8)
    ax.set_ylabel("↑ top  ·  image height  ·  bottom ↓", color="#888", fontsize=8)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised density", color="#888", fontsize=8)
    cbar.ax.tick_params(colors="#888")

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    cats_tr,imgs_tr,anns_tr =load_annotations(ANNOTATIONS_TRAIN)
    cats_te,imgs_te,anns_te =load_annotations(ANNOTATIONS_TEST)

    categories={**cats_tr,**cats_te}
    images={**imgs_tr,**imgs_te}
    annotations=[ann for ann in (anns_tr + anns_te) if ann["category_id"] <= 27]

    print(f"\nTotal categories:{len(categories)}")
    print(f"Total images:{len(images)}")
    print(f"Total annotations:{len(annotations)}\n")

    # Build density maps
    density_maps,counts=build_density_maps(annotations, images, categories)
    plot_all_heatmaps_grid(density_maps,categories,counts,title="Fashionpedia - Spatial Distribution of Fashion Categories",save_path=OUTPUT_DIR / "all_categories_grid.png")

    indiv_dir=OUTPUT_DIR / "individual"
    indiv_dir.mkdir(exist_ok=True)

    for cat_id, heatmap in tqdm(density_maps.items(), desc="Saving individual maps"):
        name=categories[cat_id]["name"].replace("/", "_").replace(" ", "_")
        fname=indiv_dir / f"{cat_id:03d}_{name}.png"
        plot_single_heatmap(heatmap, categories[cat_id]["name"], counts[cat_id], fname)

if __name__ == "__main__":
    main()