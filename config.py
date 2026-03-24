from pathlib import Path

ROOT = Path("/home/natalia/Escritorio/MAI/OR")
TRAIN_IMG= ROOT/"train2020/train/"
TEST_IMG =ROOT/"val_test2020/test/"
TRAIN_MASK = ROOT/"mask_train/"
TEST_MASK= ROOT/"mask_val/"
MODELS = ROOT/"models/"
RESULTS = ROOT/"results/"
ANNOTATIONS_TRAIN = "/home/natalia/Escritorio/MAI/OR/instances_attributes_train2020.json"
ANNOTATIONS_TEST = "/home/natalia/Escritorio/MAI/OR/instances_attributes_val2020.json"