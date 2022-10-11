from glob import glob
import argparse
import logging
import math
import random
import sys
from copy import deepcopy
import torch
import cv2
import os
import time
from pathlib import Path
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, save_one_box, scale_coords
from utils.plots import Annotator, colors
from sklearn.metrics import confusion_matrix
from models.MDA_Net import *
from utils.datasets import create_dataloader

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

# set model
weights = "runs/train/exp/weights/best.pt"
model_yolo = attempt_load(weights, map_location="cuda:0")
DANet = DANet()
module_1 = model_yolo.model[:11]
module_2 = model_yolo.model[:15]
module_3 = model_yolo.model[:22]

#setup dataset


train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                            hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
                                            workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                            prefix=colorstr('train: '))
mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
nb = len(train_loader)  # number of batches
assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]


