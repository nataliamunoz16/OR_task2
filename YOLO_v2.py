import os
import sys
import json
import shutil
import random
import time
import argparse
from pathlib import Path
from itertools import product
import numpy as np
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm
import yaml
from ultralytics import YOLO
import config  
import torch

MAIN_CLASSES=("shirt, blouse","top,t-shirt,sweatshirt","sweater","cardigan","jacket","vest","pants","shorts","skirt","coat","dress","jumpsuit","cape","glasses","hat","headband,head covering,hair accessory","tie","glove","watch","belt","leg warmer","tights,stockings","sock","shoe","bag,wallet","scarf","umbrella")
NUM_CLASSES=len(MAIN_CLASSES)        

def coco_to_yolo_segmentation(annotation_file:str,img_dir:Path,out_label_dir:Path,class_ids_to_keep:list[int],split_name: str = "")->list[str]:
    """
    Converts a COCO annotation JSON to per-image YOLO segmentation .txt files.
    """
    out_label_dir.mkdir(parents=True, exist_ok=True)
    coco=COCO(annotation_file)
    keep_set=set(class_ids_to_keep)         
    cat_id_to_yolo={cid: idx for idx, cid in enumerate(sorted(keep_set))}

    valid_img_paths = []

    for img_id in tqdm(coco.getImgIds(), desc=f"Converting {split_name} annotations"):
        img_info=coco.imgs[img_id]
        file_name=img_info["file_name"]
        img_path=img_dir / file_name

        if not img_path.exists():
            continue 

        img_w=img_info["width"]
        img_h=img_info["height"]

        ann_ids=coco.getAnnIds(imgIds=img_id)
        anns=coco.loadAnns(ann_ids)
        lines = []
        for ann in anns:
            cat_id = ann["category_id"]        
            if cat_id not in keep_set:
                continue
            if ann.get("iscrowd", 0):
                continue

            seg=ann.get("segmentation", [])
            if not seg or isinstance(seg,dict):
                x,y,w,h=ann["bbox"]
                poly=[x,y,x+w,y,x+w,y+h,x,y+h]
            else:
                # Use the largest polygon
                poly = max(seg, key=len)

            if len(poly) < 6:            # need at least 3 points
                continue

            coords=np.array(poly, dtype=np.float32).reshape(-1, 2)
            coords[:, 0]/=img_w
            coords[:, 1]/=img_h
            coords=np.clip(coords, 0.0, 1.0)

            yolo_cls=cat_id_to_yolo[cat_id]
            coord_str=" ".join(f"{v:.6f}" for v in coords.flatten())
            lines.append(f"{yolo_cls} {coord_str}")

        if not lines:
            continue

        stem=Path(file_name).stem
        label_path=out_label_dir/f"{stem}.txt"
        label_path.write_text("\n".join(lines))
        valid_img_paths.append(str(img_path))
    return valid_img_paths


def build_dataset_yaml(yolo_dataset_dir:Path,train_img_paths:list[str],val_img_paths:list[str],test_img_paths:list[str],class_names:tuple) -> Path:
    def write_list(name, paths):
        p = yolo_dataset_dir / f"{name}.txt"
        p.write_text("\n".join(paths))
        return str(p)

    train_txt=write_list("train",train_img_paths)
    val_txt=write_list("val",val_img_paths)
    test_txt=write_list("test",test_img_paths)

    yaml_content = {"path":str(yolo_dataset_dir),"train":train_txt,"val":val_txt,"test":test_txt,"nc":len(class_names),"names":list(class_names),}

    yaml_path = yolo_dataset_dir / "fashionpedia.yaml"
    
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"[dataset.yaml] saved → {yaml_path}")
    return yaml_path

def build_semantic_mask(result, h: int, w: int) -> np.ndarray:
    pred_mask=np.zeros((h, w), dtype=np.int64)

    if result.masks is None:
        return pred_mask

    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(confs) 

    for i in order:
        cls_id = int(result.boxes.cls[i].item())
        mask   = result.masks.data[i].cpu().numpy()
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        pred_mask[mask_r > 0.5] = cls_id + 1  

    return pred_mask


def build_gt_semantic_mask(label_path: Path, h: int, w: int) -> np.ndarray:
    gt_mask = np.zeros((h, w), dtype=np.int32)

    if not label_path.exists():
        return gt_mask

    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        poly = coords.astype(np.int32)
        cv2.fillPoly(gt_mask, [poly], color=cls_id + 1)

    return gt_mask.astype(np.int64)


def compute_pixel_metrics(model: YOLO, img_paths: list[str],label_dir: Path, img_size: int,num_classes: int, device: str) -> dict:
    #+1 for background at index 0
    n = num_classes + 1
    tp = np.zeros(n, dtype=np.float64)
    fp = np.zeros(n, dtype=np.float64)
    fn = np.zeros(n, dtype=np.float64)
    correct_per_class = np.zeros(n, dtype=np.float64)
    total_per_class   = np.zeros(n, dtype=np.float64)
    total_correct = 0
    total_pixels  = 0

    for img_path in tqdm(img_paths, desc="Computing pixel metrics"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # ground-truth mask
        stem= Path(img_path).stem
        label_path= label_dir / f"{stem}.txt"
        gt_mask= build_gt_semantic_mask(label_path, h, w)

        # predicted semantic mask
        results= model.predict(img_path, imgsz=img_size,device=device, verbose=False)
        pred_mask = build_semantic_mask(results[0], h, w)

        # accumulate
        total_correct+=int((pred_mask== gt_mask).sum())
        total_pixels+=h*w

        for c in range(n):
            pred_c=pred_mask==c
            gt_c =gt_mask==c
            tp[c] += int((pred_c &  gt_c).sum())
            fp[c] += int((pred_c & ~gt_c).sum())
            fn[c] += int((~pred_c & gt_c).sum())
            correct_per_class[c] += int((pred_c & gt_c).sum())
            total_per_class[c]   += int(gt_c.sum())

    eps =1e-6
    iou_per_class = tp/ (tp + fp + fn + eps)
    dice_per_class= 2 * tp / (2 * tp + fp + fn + eps)
    acc_per_class= correct_per_class / (total_per_class + eps)
    accuracy= total_correct / (total_pixels + eps)

    fg = slice(1, None)

    return {
        "mIoU":               float(iou_per_class.mean()),
        "mDice":              float(dice_per_class.mean()),
        "mDice_no_bg":        float(dice_per_class[fg].mean()),
        "accuracy":           float(accuracy),
        "mean_acc":           float(acc_per_class.mean()),
        "mean_acc_no_bg":     float(acc_per_class[fg].mean()),
        "dice_per_class":     dice_per_class[fg].tolist(),
        "accuracy_per_class": acc_per_class[fg].tolist(),
        "iou_per_class":      iou_per_class[fg].tolist(),
    }

def train_yolo(yaml_path:Path,
                #grid-search parameters
                model_size:str="s",#n/s/m/l/x
                yolo_version:str="yolo11",# yolov8 | yolo11
                img_size:int=384,#320/512/640/768 better if it is multiple of 32 (as YOLO does 5 downsampling layers)
                epochs:int=20,
                batch_size:int=8,
                lr0:float=1e-3,# initial learning rate
                lrf:float=0.01,# final lr factor  (lr_final = lr0*lrf)
                weight_decay:float=5e-4,
                momentum:float=0.937,
                warmup_epochs:int=3,
                patience:int=15,# early stopping patience
                #others
                pretrained:bool=True,
                device:str="auto",
                output_dir:Path=None,
                seed:int=42)->dict:

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"          # first GPU
        else:
            device = "cpu"

    #model weights string
    weights=f"{yolo_version}{model_size}-seg.pt" if pretrained else f"{yolo_version}{model_size}-seg.yaml"

    #experiment name for logging & saving
    run_name = (f"{yolo_version}{model_size}_"f"img{img_size}_ep{epochs}_bs{batch_size}_"f"lr{lr0}_wd{weight_decay}_")

    if output_dir is None:
        output_dir=Path(config.RESULTS)/"yolo_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training:{run_name}")
    print(f"{'='*60}\n")

    t0=time.time()
    model=YOLO(weights)
    model.train(data= str(yaml_path),epochs=epochs,imgsz=img_size,batch=batch_size,lr0=lr0,lrf=lrf,weight_decay=weight_decay,momentum=momentum,warmup_epochs=warmup_epochs,patience=patience,device=device,
        seed=seed,name=run_name,project=str(output_dir),exist_ok=True,verbose=True)
    train_time=time.time()-t0

    #evaluate on test
    best_weights = output_dir / run_name / "weights" / "best.pt"

    if best_weights.exists():
        model_best=YOLO(str(best_weights))
        yolo_dataset_dir=yaml_path.parent
        val_img_paths= (yolo_dataset_dir / "val.txt").read_text().strip().splitlines()
        test_img_paths= (yolo_dataset_dir / "test.txt").read_text().strip().splitlines()

        val_label_dir= yolo_dataset_dir / "labels" / "val"
        test_label_dir = yolo_dataset_dir / "labels" / "test"

        print("\nEvaluating pixel-wise metrics on validation set …")
        val_metrics = compute_pixel_metrics(
            model_best, val_img_paths, val_label_dir,
            img_size, NUM_CLASSES, device
        )

        val_metrics_path = output_dir / run_name / "val_metrics.json"
        val_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(val_metrics_path, "w") as f:
            json.dump(val_metrics, f, indent=4)

        print("\nEvaluating pixel-wise metrics on test set …")
        test_metrics = compute_pixel_metrics(
            model_best, test_img_paths, test_label_dir,
            img_size, NUM_CLASSES, device
        )

        test_metrics_path = output_dir / run_name / "test_metrics.json"
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
    else:
        print("Warning: best.pt not found, skipping evaluation.")
        val_metrics  = {}
        test_metrics = {}

    summary = {"run_name":run_name,"yolo_version":yolo_version,"model_size":model_size,"img_size":img_size,"epochs":epochs,"batch_size":batch_size,"lr0":lr0,"lrf":lrf,"weight_decay":weight_decay,
        "momentum":momentum, "warmup_epochs":warmup_epochs,"patience":patience,"pretrained":pretrained,"train_time_s":round(train_time, 1),
        "mDice_no_bg":round(test_metrics.get("mDice_no_bg", -1), 4),"accuracy":round(test_metrics.get("accuracy",-1), 4),"mIoU":round(test_metrics.get("mIoU",-1), 4),"mDice":round(test_metrics.get("mDice",-1), 4)}

    run_summary_path=output_dir/run_name/"run_summary.json"
    with open(run_summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    return summary

def grid_search(yaml_path: Path, grid: dict, fixed: dict, output_dir: Path) -> None:
    """
    Runs every combination of values in grid with the fixed params in fixed.
    Saves a combined leaderboard JSON to output_dir/grid_search_results.json.
    Example grid:
        grid = {"model_size":["s","m"],"img_size":[512,640],"lr0":[1e-3, 5e-4]}
    """
    keys=list(grid.keys())
    combos=list(product(*[grid[k] for k in keys]))
    print(f"Grid search: {len(combos)} combinations over {keys}")

    all_results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combos)}] params: {params}")
        cfg = {**fixed, **params, "output_dir": output_dir, "yaml_path": yaml_path}
        result = train_yolo(**cfg)
        all_results.append(result)

        # save running leaderboard after each run
        leaderboard = sorted(all_results, key=lambda r: r["mDice_no_bg"], reverse=True)
        lb_path = output_dir / "grid_search_results.json"
        with open(lb_path, "w") as f:
            json.dump(leaderboard, f, indent=4)

    print("Grid search complete. Top-3 configurations:")
    for r in leaderboard[:3]:
        print(f"{r['run_name']}  mDice_no_bg={r['mDice_no_bg']}  accuracy={r['accuracy']}")


#Data preparation: ONLY necessry to run it once

def prepare_data(yolo_dataset_dir:Path, class_ids_to_keep:list[int],seed:int=42) -> Path:
    yaml_path=yolo_dataset_dir/"fashionpedia.yaml"
    if yaml_path.exists():
        print(f"[prepare_data] dataset.yaml already exists at {yaml_path}, skipping conversion.")
        return yaml_path

    print("[prepare_data] Converting COCO → YOLO-seg labels …")

    #convert train and test labels
    train_label_dir=yolo_dataset_dir/"labels"/"train"
    train_img_paths=coco_to_yolo_segmentation(annotation_file=config.ANNOTATIONS_TRAIN,img_dir=Path(config.TRAIN_IMG),out_label_dir=train_label_dir,class_ids_to_keep=class_ids_to_keep,split_name="train")
    train_img_paths.sort() # Ensure deterministic starting order

    test_label_dir=yolo_dataset_dir/"labels"/"test"
    test_img_paths=coco_to_yolo_segmentation(annotation_file=config.ANNOTATIONS_TEST,img_dir=Path(config.TEST_IMG),out_label_dir=test_label_dir,class_ids_to_keep=class_ids_to_keep,split_name="test")
    test_img_paths.sort() # Ensure deterministic starting order

    val_size=len(test_img_paths)
    train_total_count=len(train_img_paths)
    
    if val_size > train_total_count:
        raise ValueError(f"Validation size ({val_size}) is larger than train set ({train_total_count}).")

    #Use the exact same torch generator logic to have the same validation set
    generator = torch.Generator().manual_seed(seed)
    
    # random_split internally uses randperm to shuffle indices
    indices = torch.randperm(train_total_count, generator=generator).tolist()
    train_size = train_total_count - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_paths = [train_img_paths[i] for i in train_indices]
    val_paths = [train_img_paths[i] for i in val_indices]
    test_paths = test_img_paths

    val_label_dir = yolo_dataset_dir / "labels" / "val"
    val_label_dir.mkdir(parents=True, exist_ok=True)
    for img_path in val_paths:
        stem = Path(img_path).stem
        src = train_label_dir / f"{stem}.txt"
        dst = val_label_dir / f"{stem}.txt"
        if src.exists():
            shutil.copy(src, dst)

    return build_dataset_yaml(yolo_dataset_dir=yolo_dataset_dir,train_img_paths=train_paths,val_img_paths=val_paths,test_img_paths=test_paths,class_names=MAIN_CLASSES)



def main():
    yolo_dataset_dir=Path(config.ROOT)/"yolo_fashionpedia"
    class_ids_to_keep=list(range(NUM_CLASSES))                           

    #One run: default
    SINGLE_RUN = dict(yolo_version ="yolo11",model_size="s",img_size=384,epochs=100,batch_size=16,lr0=0.01,lrf=0.01,weight_decay=5e-4,momentum=0.937,warmup_epochs=3,patience=15,
        #others
        pretrained=True,device="auto",seed=42)

    #Grid search
    # Set DO_GRID_SEARCH = True to run all combinations below.
    DO_GRID_SEARCH=False
    
    GRID={"model_size":["s","m"],"img_size":[384, 192],"lr0":[1e-2, 1e-3]}
    # GRID_2 = {"model_size":["s", "m"],"img_size":[384, 640],"lr0":[1e-2, 1e-3],"weight_decay":[1e-4, 5e-4, 1e-3]}

    # params that are fixed during grid search (everything not in GRID)
    FIXED_PARAMS=dict(yolo_version="yolo11",epochs=20,batch_size=8,lrf=0.01,weight_decay=5e-4,momentum=0.937,warmup_epochs=3,patience=15,pretrained=True,device="auto",seed=42)
    yaml_path=prepare_data(yolo_dataset_dir, class_ids_to_keep)

    output_dir=Path(config.RESULTS)/"yolo_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if DO_GRID_SEARCH:
        grid_search(yaml_path, GRID, FIXED_PARAMS, output_dir)
    else:
        train_yolo(yaml_path=yaml_path, output_dir=output_dir, **SINGLE_RUN)


if __name__ == "__main__":
    main()