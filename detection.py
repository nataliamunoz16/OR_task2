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

def copy_images_to_dataset(yolo_dataset_dir: Path, train_paths: list[str], val_paths: list[str], test_paths: list[str]):
    """
    Physically copies images from source paths to the yolo_dataset_dir/images folder.
    """
    for split, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        dest_dir=yolo_dataset_dir / "images" / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying {split} images to {dest_dir}...")
        for p in tqdm(paths, desc=f"Copying {split}"):
            src=Path(p)
            dst=dest_dir / src.name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)

def coco_to_yolo_detection(annotation_file:str,img_dir:Path,out_label_dir:Path,class_ids_to_keep:list[int],split_name:str = "")->list[str]:
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
            x,y,w,h = ann["bbox"]
            if w<=1 or h<=1:
                continue

            x_center =(x+w/2)/img_w
            y_center =(y+h/2)/img_h
            w = w/img_w
            h =h/img_h
            x_center =np.clip(x_center, 0.0, 1.0)
            y_center =np.clip(y_center, 0.0, 1.0)
            w =np.clip(w, 0.0, 1.0)
            h =np.clip(h, 0.0, 1.0)

            yolo_cls=cat_id_to_yolo[cat_id]
            lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

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

    yaml_content = {"path":str(yolo_dataset_dir),"train":train_txt,"val":val_txt,"test":test_txt,"nc":len(class_names),"names":list(class_names)}

    yaml_path = yolo_dataset_dir / "fashionpedia.yaml"
    
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"[dataset.yaml] saved --> {yaml_path}")
    return yaml_path

def train_yolo(yaml_path:Path,
                model_size:str="s",
                yolo_version:str="yolo11",
                img_size:int=384,
                epochs:int=20,
                batch_size:int=8,
                lr0:float=1e-3,
                lrf:float=0.01,
                weight_decay:float=5e-4,
                momentum:float=0.937,
                warmup_epochs:int=3,
                patience:int=15,
                pretrained:bool=True,
                device:str="auto",
                output_dir:Path=None,
                seed:int=42)->dict:

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"

    weights=f"{yolo_version}{model_size}.pt" if pretrained else f"{yolo_version}{model_size}.yaml"
    run_name = (f"{yolo_version}{model_size}_"f"img{img_size}_ep{epochs}_bs{batch_size}_"f"lr{lr0}_wd{weight_decay}_")

    if output_dir is None:
        output_dir=Path(config.RESULTS)/"yolo_detection_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training:{run_name}")
    print(f"{'='*60}\n")

    t0=time.time()
    model=YOLO(weights)
    model.train(data= str(yaml_path),epochs=epochs,imgsz=img_size,batch=batch_size,lr0=lr0,lrf=lrf,weight_decay=weight_decay,momentum=momentum,warmup_epochs=warmup_epochs,patience=patience,device=device,
        seed=seed,name=run_name,project=str(output_dir),exist_ok=True,verbose=True)
    train_time=time.time()-t0

    # evaluate on test
    best_weights=output_dir/run_name/"weights"/"best.pt"
    if best_weights.exists():
        model_best=YOLO(str(best_weights))
        test_metrics=model_best.val(data=str(yaml_path),split="test",imgsz=img_size,device=device,name=run_name+"_test",project=str(output_dir),exist_ok=True)
        box_map50=float(test_metrics.box.map50)
        box_map50_95=float(test_metrics.box.map)
    else:
        print("Warning: best.pt not found, skipping test evaluation.")
        box_map50 = box_map50_95 = -1.0

    summary = {"run_name":run_name,"yolo_version":yolo_version,"model_size":model_size,"img_size":img_size,"epochs":epochs,"batch_size":batch_size,"lr0":lr0,
        "lrf":lrf,"weight_decay": weight_decay,"momentum":momentum,"warmup_epochs":warmup_epochs,"patience":patience,"pretrained":pretrained,"train_time_s":round(train_time, 1),
        # test metrics
        "box_mAP50":round(box_map50,4),"box_mAP50-95":round(box_map50_95,4)}

    metrics_path=output_dir/run_name/"test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path,"w") as f:
        json.dump(summary, f, indent=4)

    print(f"box-mAP 50:{box_map50:.4f} box-mAP 50-95: {box_map50_95:.4f}")
    return summary


# Data preparation: ONLY necessary to run it once

def prepare_data(yolo_dataset_dir:Path, class_ids_to_keep:list[int],seed:int=42) -> Path:
    yaml_path=yolo_dataset_dir/"fashionpedia.yaml"
    if yaml_path.exists():
        print(f"[prepare_data] dataset.yaml already exists at {yaml_path}, skipping conversion.")
        return yaml_path

    print("[prepare_data] Converting COCO --> YOLO labels")

    # convert train and test labels
    train_label_dir=yolo_dataset_dir/"labels"/"train"
    train_img_paths=coco_to_yolo_detection(annotation_file=config.ANNOTATIONS_TRAIN,img_dir=Path(config.TRAIN_IMG),out_label_dir=train_label_dir,class_ids_to_keep=class_ids_to_keep,split_name="train")
    train_img_paths.sort()

    test_label_dir=yolo_dataset_dir/"labels"/"test"
    test_img_paths=coco_to_yolo_detection(annotation_file=config.ANNOTATIONS_TEST,img_dir=Path(config.TEST_IMG),out_label_dir=test_label_dir,class_ids_to_keep=class_ids_to_keep,split_name="test")
    test_img_paths.sort()

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
    val_label_dir = yolo_dataset_dir / "labels" / "val"
    val_label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in val_paths:
        stem = Path(img_path).stem
        src = train_label_dir / f"{stem}.txt"
        dst = val_label_dir / f"{stem}.txt"
        if src.exists():
            shutil.copy(src, dst)
    test_paths = test_img_paths
    copy_images_to_dataset(yolo_dataset_dir, train_paths, val_paths, test_paths)

    final_train_paths = [str(yolo_dataset_dir / "images" / "train" / Path(p).name) for p in train_paths]
    final_val_paths = [str(yolo_dataset_dir / "images" / "val" / Path(p).name) for p in val_paths]
    final_test_paths = [str(yolo_dataset_dir / "images" / "test" / Path(p).name) for p in test_paths]

    print(f"Final split: train={len(final_train_paths)} | val={len(final_val_paths)} (matches test size) | test={len(final_test_paths)}")

    return build_dataset_yaml(yolo_dataset_dir=yolo_dataset_dir,train_img_paths=final_train_paths,val_img_paths=final_val_paths,test_img_paths=final_test_paths,class_names=MAIN_CLASSES)

def predict_bb(model_path, image_dir, output_json, conf, device):
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"          # first GPU
        else:
            device = "cpu"
    model=YOLO(str(model_path))
    print('hola')
    image_paths=os.listdir(image_dir)
    image_paths = [image_dir / name for name in image_paths]
    image_paths = sorted(image_paths)
    results={}
    batch_size=64
    for i in range(0, len(image_paths), batch_size):
        batch=image_paths[i:i+batch_size]
        res=model.predict(source=[str(p) for p in batch], conf=conf, device=device, verbose=True)
        for img_path, result in zip(batch, res):
            boxes=[]
            if result.boxes is not None and len(result.boxes)>0:
                xyxy=result.boxes.xyxy.cpu().numpy()
                confs=result.boxes.conf.cpu().numpy()
                classes=result.boxes.cls.cpu().numpy().astype(int)
                for box, score, idclass in zip(xyxy, confs, classes):
                    x1,y1,x2,y2=box.tolist()
                    boxes.append({"class_id":int(idclass), "confidence":float(score), "bbox": [float(x1), float(y1), float(x2), float(y2)]})
            results[img_path.name]=boxes
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    return results


            

def main():
    yolo_dataset_dir=Path(config.ROOT)/"yolo_fashionpedia_detection"
    class_ids_to_keep=list(range(NUM_CLASSES))                           

    SINGLE_RUN = dict(yolo_version ="yolo11",model_size="s",img_size=384,epochs=20,batch_size=8,lr0=1e-3,lrf=0.01,weight_decay=5e-4,momentum=0.937,warmup_epochs=3,patience=15,pretrained=True,device="auto",seed=42)

    #yaml_path=prepare_data(yolo_dataset_dir,class_ids_to_keep)

    output_dir=Path(config.RESULTS) / "yolo_runs_detection"
    output_dir.mkdir(parents=True, exist_ok=True)
    #summary=train_yolo(yaml_path=yaml_path, output_dir=output_dir, **SINGLE_RUN)
    summary = {"run_name": "yolo11s_img384_ep20_bs8_lr0.001_wd0.0005_"}
    best_model_path=output_dir / summary["run_name"] / "weights" / "best.pt"
    out_dir = output_dir / summary["run_name"] / "predicted_bb_train_val.json"
    predict_bb(model_path=best_model_path, image_dir=Path(config.TRAIN_IMG), output_json=out_dir, conf=0.25, device="cpu")
    out_dir = output_dir / summary["run_name"] / "predicted_bb_test.json"
    predict_bb(model_path=best_model_path, image_dir=Path(config.TEST_IMG), output_json=out_dir, conf=0.25, device="cpu")

if __name__ == "__main__":
    main()