
import os, random
import numpy as np
import torch
from pathlib import Path

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR      = Path(".")
DATA_DIR      = BASE_DIR / "archive"
IMG_DIR_1     = DATA_DIR / "HAM10000_images_part_1"
IMG_DIR_2     = DATA_DIR / "HAM10000_images_part_2"
META_PATH     = DATA_DIR / "HAM10000_metadata.csv"
OUT_DIR       = BASE_DIR / "outputs"
PLOTS_DIR     = OUT_DIR / "plots"
MODELS_DIR    = OUT_DIR / "models"
SYNTHETIC_DIR = OUT_DIR / "synthetic_images"
IMG_SIZE      = 224
BATCH_SIZE    = 8
NUM_CLASSES   = 7
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 15
LR_PHASE1     = 1e-3
LR_PHASE2     = 1e-4
DROPOUT_RATE  = 0.4
WEIGHT_DECAY  = 1e-4
Z_DIM         = 100
GAN_FEAT_G    = 64
GAN_FEAT_D    = 64
GAN_IMG_SIZE  = 64
GAN_EPOCHS    = 200
LR_GAN        = 2e-4
BETA1_GAN     = 0.5
TARGET_COUNT  = 6705
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS  = {i: c for c, i in CLASS_TO_IDX.items()}
CLASS_FULLNAME = {"akiec":"Actinic Keratosis","bcc":"Basal Cell Carcinoma","bkl":"Benign Keratosis","df":"Dermatofibroma","mel":"Melanoma","nv":"Melanocytic Nevi","vasc":"Vascular Lesion"}
