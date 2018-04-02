#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import fnmatch
import json

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis_text as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default="configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml",
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default="model_final.pkl",
        type=str
    )
    parser.add_argument(
        '--output-file',
        dest='output_file',
        help='filename for JSON output',
        default='infer.json',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='png',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def riglob(target, pattern):
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = riglob(args.im_or_folder, "*." + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    with open(args.output_file, "w") as f:
        for i, im_name in enumerate(im_list):
            logger.info('Processing ' + im_name)
            im = cv2.imread(im_name)
            if im is None:
                continue
            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, im, None, timers=timers
                )
            logger.info('Inference time: {:.3f}s'.format(time.time() - t))
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
            if i == 0:
                logger.info(
                    ' \ Note: inference on the first image will be slower than the '
                    'rest (caches and auto-tuning need to warm up)'
                )

            result = vis_utils.vis_one_image(
                im_name,
                cls_boxes,
                cls_segms,
                dataset=dummy_coco_dataset
            )
            f.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
