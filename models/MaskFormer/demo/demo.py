# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from MaskFormer.mask_former import add_mask_former_config
from MaskFormer.demo.predictor import VisualizationDemo


# constants
WINDOW_NAME = "MaskFormer demo"


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file("../models/MaskFormer/configs/mapillary-vistas-65/maskformer_R50_bs16_300k.yaml")
    # cfg.merge_from_list(args.opts)
    # cfg.MODEL.WEIGHTS = 'model_final_f3fc73.pkl'
    # cfg.freeze()
    return cfg

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def get_maskformer(mapillary_pretrained=False):
#     cfg = setup_cfg()
#     demo = VisualizationDemo(cfg)
#     model = demo.predictor.model
#     args.opts['MODEL.WEIGHTS'] = 'model_final_f3fc73.pkl'

    cfg = setup_cfg()
    if mapillary_pretrained:
        cfg.MODEL.WEIGHTS = 'model_final_f3fc73.pkl'
    else:
        cfg.MODEL.WEIGHTS = ''
    demo = VisualizationDemo(cfg)
    model = demo.predictor.model
    return model
