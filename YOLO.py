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
    """
    Writes train/val/test image-list .txt files and the dataset.yaml.
    YOLO accepts a plain text file listing absolute image paths.
    """
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

def train_yolo(yaml_path:Path,
                #grid-search parameters
                model_size:str="s",          #n/s/m/l/x
                yolo_version:str="yolo11",     # yolov8 | yolo11
                img_size:int=384,           #320/512/640/768 better if it is multiple of 32 (as YOLO does 5 downsampling layers)
                epochs:int=20,
                batch_size:int=8,
                lr0:float=1e-3,          # initial learning rate
                lrf:float=0.01,          # final lr factor  (lr_final = lr0*lrf)
                weight_decay:float=5e-4,
                momentum:float=0.937,
                warmup_epochs:int=3,
                patience:int=15,            # early stopping patience
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
    best_weights=output_dir/run_name/"weights"/"best.pt"
    if best_weights.exists():
        model_best=YOLO(str(best_weights))
        test_metrics=model_best.val(data=str(yaml_path),split="test",imgsz=img_size,device=device,name=run_name+"_test",project=str(output_dir),exist_ok=True,)
        seg_map50=float(test_metrics.seg.map50)
        seg_map50_95=float(test_metrics.seg.map)
        box_map50=float(test_metrics.box.map50)
        box_map50_95=float(test_metrics.box.map)
    else:
        print("Warning: best.pt not found, skipping test evaluation.")
        seg_map50 = seg_map50_95 = box_map50 = box_map50_95 = -1.0

    summary = {"run_name":run_name,"yolo_version":yolo_version,"model_size":model_size,"img_size":img_size,"epochs":epochs,"batch_size":batch_size,"lr0":lr0,
        "lrf":lrf,"weight_decay": weight_decay,"momentum":momentum,"warmup_epochs":warmup_epochs,"patience":patience,"pretrained":pretrained,"train_time_s":round(train_time, 1),
        # test metrics
        "seg_mAP50":round(seg_map50,4),"seg_mAP50-95":round(seg_map50_95,4),
        "box_mAP50":round(box_map50,4),"box_mAP50-95":round(box_map50_95,4)}

    metrics_path=output_dir/run_name/"test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path,"w") as f:
        json.dump(summary, f, indent=4)

    print(f"Test seg-mAP 50:{seg_map50:.4f} seg-mAP 50-95: {seg_map50_95:.4f}")
    print(f"box-mAP 50:{box_map50:.4f} box-mAP 50-95: {box_map50_95:.4f}")
    return summary


# ──────────────────────────────────────────────
# STEP 4 – Lightweight grid search
# ──────────────────────────────────────────────

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
        leaderboard = sorted(all_results, key=lambda r: r["seg_mAP50-95"], reverse=True)
        lb_path = output_dir / "grid_search_results.json"
        with open(lb_path, "w") as f:
            json.dump(leaderboard, f, indent=4)

    print("Grid search complete. Top-3 configurations:")
    for r in leaderboard[:3]:
        print(f"{r['run_name']} seg-mAP50-95={r['seg_mAP50-95']}")


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

    print(f"Final split: train={len(train_paths)} | val={len(val_paths)} (matches test size) | test={len(test_paths)}")

    return build_dataset_yaml(yolo_dataset_dir=yolo_dataset_dir,train_img_paths=train_paths,val_img_paths=val_paths,test_img_paths=test_paths,class_names=MAIN_CLASSES)



def main():
    yolo_dataset_dir=Path(config.ROOT)/"yolo_fashionpedia"
    class_ids_to_keep=list(range(NUM_CLASSES))                           

    #One run
    SINGLE_RUN = dict(yolo_version ="yolo11",model_size="s",img_size=640,epochs=20,batch_size=8,lr0=1e-3,lrf=0.01,weight_decay=5e-4,momentum=0.937,warmup_epochs=3,patience=15,
        #others
        pretrained=True,device="auto",seed=42)

    #Grid search
    # Set DO_GRID_SEARCH = True to run all combinations below.
    DO_GRID_SEARCH=False
    GRID={"model_size":["s","m"],"img_size":[512, 640],"lr0":[1e-3, 5e-4]}

    # params that are fixed during grid search (everything not in GRID)
    FIXED_PARAMS=dict(yolo_version="yolov8",epochs=20,batch_size=8,lrf=0.01,weight_decay=5e-4,momentum=0.937,warmup_epochs=3,patience=15,pretrained=True,device="auto",seed=42)
    yaml_path=prepare_data(yolo_dataset_dir, class_ids_to_keep)

    output_dir=Path(config.RESULTS) / "yolo_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if DO_GRID_SEARCH:
        grid_search(yaml_path, GRID, FIXED_PARAMS, output_dir)
    else:
        train_yolo(yaml_path=yaml_path, output_dir=output_dir, **SINGLE_RUN)


if __name__ == "__main__":
    main()