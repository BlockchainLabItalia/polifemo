import argparse
import os
import platform
import shutil
from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.general import (strip_optimizer, set_logging, check_img_size)
from utils.torch_utils import select_device

from camera import Camera

import json

import signal
import sys

def sigint_handler(signal, frame):
    print('Interrupted')
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)



def detect(configuration_data):
    weights, imgsz, cameras = \
        configuration_data["arguments"]["weights"], configuration_data["arguments"]["img_size"], configuration_data["cameras"]

    # Initialize
    set_logging()
    device = select_device(configuration_data["arguments"]["device"])

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    half = device.type != 'cpu'  # half precision only supported on CUDA

    if half:
        model.half()  # to FP16
            
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for camera in cameras:
        Camera(camera["name"],model,device,imgsz,camera["source"],camera["line_orientation"],camera["line_position"],camera["position_in"],configuration_data["databases"]).start()



if __name__ == '__main__':
    with open('./configuration.json') as f:
        configuration_data = json.load(f)
    
    with torch.no_grad():
        detect(configuration_data)
