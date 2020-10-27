import argparse
import os
import platform
import shutil
from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.general import (strip_optimizer, set_logging)
from utils.torch_utils import select_device

from camera import Camera




def detect():
    out, source, weights, imgsz, line_orientation, line_position, position_in = \
        opt.output, opt.source, opt.weights, opt.img_size, opt.line_orientation, opt.line_position, opt.position_in

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    


    Camera("id1",model,device,imgsz,source,line_orientation,line_position,position_in).start()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--line-orientation', type=str, default='vertical', help='vertical or horizontal')
    parser.add_argument('--line-position', type=float, default=0.5, help='position of the line [0 < position < 1]')
    parser.add_argument('--position-in', type=str, default='left', help='what to consider in or out. if line orientation is vertical, accepted are left or right. if orientation is horizontal, accepted are top or bottom')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
