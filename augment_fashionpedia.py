import os
import config
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from pathlib import Path

AUG_FRACTION=0.20   # fraction of training images to augment
DUAL_AUG_PROB=0.50   # probability of applying TWO augmentations on the same image
TEXTURE_PROB=0.40 # probability of changing clothes textures while doing the data augmenting
USE_MIXUP_PROB=0 # probability of applying mixup
OUTPUT_DIR= os.path.join(config.ROOT, "augmented")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"),  exist_ok=True)
NO_TEXTURE_IDS = [5, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27]

def load_pair(img_name:str):
    mask_name =img_name.split(".")[0] + "_seg.png"
    img  =np.array(Image.open(os.path.join(config.TRAIN_IMG,  img_name )).convert("RGB"))
    mask =np.array(Image.open(os.path.join(config.TRAIN_MASK, mask_name)).convert("L"))
    return img, mask, mask_name

def save_pair(img:np.ndarray, mask:np.ndarray, stem:str, tag:str):
    Image.fromarray(img ).save(os.path.join(OUTPUT_DIR, "images", f"{stem}__{tag}.jpg"))
    Image.fromarray(mask).save(os.path.join(OUTPUT_DIR, "masks",  f"{stem}__{tag}_seg.png"))


## augmentations
def aug_rotation(img,mask,angle_range=(-15, 15)):
    """Slight rotation – keeps full image, pads with border pixels."""
    h, w =img.shape[:2]
    angle =random.uniform(*angle_range)
    M =cv2.getRotationMatrix2D((w/2,h/2), angle,1.0)
    img_r= cv2.warpAffine(img,  M,(w,h), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    mask_r= cv2.warpAffine(mask, M,(w,h), flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_r, mask_r


def aug_translation(img, mask, max_shift=0.15):
    """Random x/y shift up to "max_shift" fraction of image dimensions."""
    h, w =img.shape[:2]
    tx=int(random.uniform(-max_shift, max_shift)*w)
    ty= int(random.uniform(-max_shift, max_shift)*h)
    M=np.float32([[1, 0, tx], [0, 1, ty]])
    img_t=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
    mask_t=cv2.warpAffine(mask,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_t, mask_t


def aug_skew(img,mask,skew_range=(-0.10,0.10)):
    """Horizontal shear / skew."""
    h, w=img.shape[:2]
    shear= random.uniform(*skew_range)
    M =np.float32([[1, shear, 0], [0, 1, 0]])
    img_s= cv2.warpAffine(img,  M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    mask_s= cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_s,mask_s

def aug_scale_crop(img, mask, scale_range=(0.70, 0.95)):
    """Random crop that simulates zoom-in scaling; output is resized back to original."""
    h, w =img.shape[:2]
    scale =random.uniform(*scale_range)
    new_h,new_w = int(h * scale), int(w * scale)
    y0 =random.randint(0, h - new_h)
    x0 =random.randint(0, w - new_w)
    img_c= img [y0:y0+new_h, x0:x0+new_w]
    mask_c= mask[y0:y0+new_h, x0:x0+new_w]
    img_c= cv2.resize(img_c,  (w, h), interpolation=cv2.INTER_LINEAR)
    mask_c= cv2.resize(mask_c, (w, h), interpolation=cv2.INTER_NEAREST)
    return img_c, mask_c

def aug_hflip(img, mask):
    """Horizontal flip – mirror image and mask."""
    return np.fliplr(img).copy(),np.fliplr(mask).copy()


def aug_color_jitter(img, mask, brightness=0.5, contrast=0.5, saturation=0.5):
    img_pil=Image.fromarray(img)
    factors=[(ImageEnhance.Brightness,random.uniform(1-brightness,1+brightness)),(ImageEnhance.Contrast,random.uniform(1-contrast,1+contrast)),(ImageEnhance.Color,random.uniform(1-saturation,1+saturation))]
    random.shuffle(factors)
    for enhancer_cls, factor in factors:
        img_pil =enhancer_cls(img_pil).enhance(factor)
    return np.array(img_pil), mask

def aug_fabric_texture(img: np.ndarray, mask: np.ndarray, fabrics: list) -> tuple[np.ndarray, np.ndarray]:
    """Replace one randomly chosen garment segment with a tiled fabric texture,
    preserving original shading via a multiply blend. Mask is unchanged."""
    if not fabrics:
        return img, mask
    h, w = mask.shape
    uniques =np.unique(mask)
    uniques =uniques[uniques!=0]
    valid_ids =[u for u in uniques if u not in NO_TEXTURE_IDS]
    if not valid_ids:
        return img, mask

    target_id= random.choice(valid_ids)
    fabric_path= os.path.join(config.FABRICS, random.choice(fabrics))
    fabric_np= np.array(Image.open(fabric_path).convert("RGB"))

    fabric_tiled= np.tile(fabric_np,(h // fabric_np.shape[0] + 1, w // fabric_np.shape[1] + 1, 1))[:h, :w, :]

    #apply fabric texture while preserving original lighting/folds
    shading =cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
    shading =np.stack([shading]*3, axis=-1)
    blended =(fabric_tiled.astype(float) * shading).astype(np.uint8)

    img_out= img.copy()
    img_out[mask == target_id] = blended[mask == target_id]
    return img_out,mask
##### strong augmentations

def aug_mosaic(img, mask, grid=(2, 2)):
    """
    combine 4 random crops from the SAME image at different scales
    """
    h, w=img.shape[:2]
    rows,cols= grid
    out_img= np.zeros_like(img)
    out_mask= np.zeros_like(mask)
    ch,cw=h//rows,w//cols
    for r in range(rows):
        for c in range(cols):
            scale=random.uniform(0.5, 0.9)
            sh, sw= int(h * scale), int(w * scale)
            y0=random.randint(0, h - sh)
            x0=random.randint(0, w - sw)
            patch_img= cv2.resize(img [y0:y0+sh, x0:x0+sw], (cw, ch), interpolation=cv2.INTER_LINEAR)
            patch_mask = cv2.resize(mask[y0:y0+sh, x0:x0+sw], (cw, ch), interpolation=cv2.INTER_NEAREST)
            out_img [r*ch:(r+1)*ch, c*cw:(c+1)*cw]= patch_img
            out_mask[r*ch:(r+1)*ch, c*cw:(c+1)*cw]= patch_mask
    return out_img, out_mask
def aug_cutout(img, mask, n_holes=3, hole_frac=0.12):
    """
    Randomly zero-out rectangular regions.  Mask is left intact.
    """
    h, w  = img.shape[:2]
    result = img.copy()
    for _ in range(n_holes):
        hh =int(h * random.uniform(0.05, hole_frac))
        hw =int(w * random.uniform(0.05, hole_frac))
        y  =random.randint(0, h - hh)
        x  =random.randint(0, w - hw)
        # fill with dataset mean (approx) instead of black for realism
        fill = np.array([123, 116, 103], dtype=np.uint8)
        result[y:y+hh, x:x+hw] = fill
    return result, mask   # mask unchanged

def aug_mixup(img, mask, img2, mask2, alpha_range=(0.3, 0.6)):
    """
    Pixel-level blend of two images
    """
    alpha = random.uniform(*alpha_range)
    h, w  = img.shape[:2]
    img2_r  = cv2.resize(img2,  (w, h))
    blended = cv2.addWeighted(img, alpha, img2_r, 1 - alpha, 0)
    # keep mask from the primary image
    return blended, mask
#####


# augmentations
LIGHT_AUGS = [("rotation",aug_rotation),("translation",aug_translation),("skew",aug_skew),("scale_crop",aug_scale_crop),("horizontal_flip", aug_hflip),("color_jitter", aug_color_jitter)]
STRONG_AUGS = [("mosaic",aug_mosaic),("cutout",aug_cutout)]

def pick(exclude=None, strong=False):
    if strong:
        pool = [x for x in LIGHT_AUGS+STRONG_AUGS if x[0] != exclude]
    else:
        pool = [x for x in LIGHT_AUGS if x[0] != exclude]
    return random.choice(pool)

def save_debug_grid(aug_results, output_path):
    n = len(aug_results)
    fig, axs = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axs = axs[np.newaxis, :]

    for row, (tag, orig_img, orig_mask, aug_img, aug_mask) in enumerate(aug_results):
        axs[row,0].imshow(orig_img)
        axs[row,1].imshow(aug_img)
        axs[row,2].imshow(orig_mask, cmap="tab20")
        axs[row,3].imshow(aug_mask, cmap="tab20")

        axs[row,1].set_title(tag, fontsize=9,color="dimgray", style="italic", pad=4)
        axs[row,0].set_ylabel(f"sample {row+1}",fontsize=10, rotation=0, labelpad=60, va="center")

        for ax in axs[row]:
            ax.axis("off")

    # column headers as text above the whole figure, not per-axes titles
    col_labels = ["Original","Augmented","Original Mask","Augmented Mask"]
    for col, label in enumerate(col_labels):
        ax=axs[0, col]
        pos=ax.get_position()
        fig.text(pos.x0+pos.width/2,pos.y1+0.01,label,ha="center", va="bottom",fontsize=12, fontweight="bold",transform=fig.transFigure)
    #plt.subplots_adjust(hspace=0.4)
    plt.savefig(output_path, dpi=80, bbox_inches="tight")
    plt.close(fig)

def main():
    mask_set=set(os.listdir(config.TRAIN_MASK))
    images=[f for f in os.listdir(config.TRAIN_IMG) if f.split(".")[0] + "_seg.png" in mask_set]
    fabrics=os.listdir(config.FABRICS) if hasattr(config, "FABRICS") and os.path.isdir(config.FABRICS) else []
    n_aug  = max(1, int(len(images)*AUG_FRACTION))
    selected = random.sample(images,n_aug)          
    print(f"Augmenting {n_aug}/{len(images)} images(dual_prob={DUAL_AUG_PROB})")
    seen_tags=set()
    debug_rows=[]          
    all_debug_rows =[]

    for idx, img_name in enumerate(selected):
        try:
            img, mask, _ = load_pair(img_name)
        except FileNotFoundError:
            continue

        stem = Path(img_name).stem
        use_dual=random.random()<DUAL_AUG_PROB # dual aug on same image
        use_texture = random.random() <TEXTURE_PROB and bool(fabrics)

        aug_names = []
        aug_img, aug_mask = img.copy(), mask.copy()

        # First augmentation
        lname, lfn = pick(exclude=None, strong=False)
        aug_img, aug_mask = lfn(aug_img, aug_mask)
        aug_names.append(lname)

        # Optional second augmentation
        if use_dual:
            sname,sfn = pick(exclude=lname, strong=True)
            aug_img,aug_mask = sfn(aug_img,aug_mask)
            aug_names.append(sname)
        if random.random() < USE_MIXUP_PROB and len(images) > 1 and not any(t in aug_names for t in ("mosaic", "cutout")):
            other_name = random.choice([x for x in images if x != img_name])
            try:
                img2, mask2, _ = load_pair(other_name)
                aug_img, aug_mask = aug_mixup(aug_img, aug_mask, img2, mask2)
                aug_names.append("mixup")
            except FileNotFoundError:
                pass
            
        #fabric replacement
        if use_texture:
            aug_img, aug_mask = aug_fabric_texture(aug_img, aug_mask, fabrics)
            aug_names.append("fabric_texture")

        tag = "+".join(aug_names)
        save_pair(aug_img, aug_mask, stem, tag)

        for single_tag in aug_names:
            if single_tag not in seen_tags:
                try:
                    if single_tag == "fabric_texture":
                        ex_img, ex_mask = aug_fabric_texture(img.copy(), mask.copy(), fabrics)
                    elif single_tag == "mixup":
                        other_name = random.choice([x for x in images if x != img_name])
                        img2, mask2, _ = load_pair(other_name)
                        ex_img, ex_mask = aug_mixup(img.copy(), mask.copy(), img2, mask2)
                    else:
                        _, fn = next(x for x in LIGHT_AUGS + STRONG_AUGS if x[0] == single_tag)
                        ex_img, ex_mask = fn(img.copy(), mask.copy())

                    debug_rows.append((single_tag, img.copy(), mask.copy(), ex_img, ex_mask))
                    seen_tags.add(single_tag)
                except (StopIteration, FileNotFoundError):
                    pass  # skip silently if example can't be generated

        # store every result for later plot
        all_debug_rows.append((tag, img.copy(),mask.copy(),aug_img.copy(),aug_mask.copy()))
        
        if (idx + 1)%100 == 0:
            print(f"{idx+1}/{n_aug} done")
    
    for i in range(6): # to generate 6 different subplots with different images
        n_debug = min(6, len(all_debug_rows))
        debug_rows = random.sample(all_debug_rows, n_debug)
        if debug_rows:
            save_debug_grid(debug_rows, os.path.join(OUTPUT_DIR, f"augmentation_overview{i}.png"))
    print("Done augmented pairs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
