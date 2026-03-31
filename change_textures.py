import os
import config
import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

images=os.listdir(config.TRAIN_IMG)
masks=os.listdir(config.TRAIN_MASK)
fabrics = os.listdir(config.FABRICS)
os.makedirs(os.path.join(config.ROOT, "generated"), exist_ok=True)
no_possible_ids = [5, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27]
ph, pw = 15, 15
wanted= int(len(images)*0.05)
selected = random.choices(images, k=wanted)

for j,path_img in enumerate(selected):
    mask_path = path_img.split(".")[0] + "_seg.png"
    if mask_path not in masks: continue
    img= Image.open(os.path.join(config.TRAIN_IMG, path_img)).convert("RGB")
    img_orig = np.array(img)
    img_result = np.array(img).copy()
    mask = Image.open(os.path.join(config.TRAIN_MASK, mask_path)).convert("L")
    mask_np =np.array(mask)
    uniques = np.unique(mask_np)
    uniques = uniques[uniques!=0]
    filtered_uniques = [unique for unique in uniques if unique not in no_possible_ids]
    if len(filtered_uniques)==0: continue
    selected_fabrics = random.choices(fabrics,k=1)
    selected_item =random.choices(filtered_uniques, k=1)
    h,w = mask_np.shape
    mask_selected = (mask_np ==selected_item[0])
    fabric = Image.open(os.path.join(config.FABRICS, selected_fabrics[0])).convert("RGB")
    fabric_np = np.array(fabric)
    # Resize and tile the fabric to match image dimensions
    fabric_tiled = np.tile(fabric_np, (int(h/fabric_np.shape[0])+1, int(w/fabric_np.shape[1])+1, 1))
    fabric_tiled = fabric_tiled[:h, :w, :]

    # 2. Get the Shading (Original Lighting)
    # Convert original image to grayscale to capture folds and shadows
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
    # Normalize shading to range [0, 1] so it acts as a multiplier
    shading = img_gray.astype(float) / 255.0
    shading = np.stack([shading]*3, axis=-1) # Make it 3-channel for RGB multiplication

    # 3. Apply the Multiply Blend
    # This applies the texture but keeps the original shadows/highlights
    blended_fabric = (fabric_tiled.astype(float) * shading).astype(np.uint8)

    # 4. Mask and Overlay
    img_result = img_orig.copy()
    mask_selected = (mask_np == selected_item[0])
    img_result[mask_selected] = blended_fabric[mask_selected]
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(img_orig)
    axs[0].set_title("original")
    axs[0].axis("off")
    axs[1].imshow(img_result)
    axs[1].set_title("Generated")
    axs[1].axis("off")
    plt.savefig(f"{config.ROOT}/generated/generated_{j}.png")
    plt.close(fig)