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
from utils.datasets import LoadStreams

from camera import Camera
from CameraStream import CameraStream
from myInflux import MyInflux

import json

import signal
import sys


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

    influx_config = configuration_data["databases"]
    db_client = MyInflux(influx_config["hostname"],influx_config["database_name"],influx_config["port"])

    for camera in cameras:

        camera_process = CameraStream(camera["source"], model, device, camera["line_orientation"], camera["line_position"],camera["position_in"], db_client, img_size=imgsz, cameraID=camera["name"])
        

        camera_process.start_processing()
        print('started')
        #Camera(camera["name"],model,device,imgsz,camera["source"],camera["line_orientation"],camera["line_position"],camera["position_in"],configuration_data["databases"]).start()



if __name__ == '__main__':
    with open('./configuration.json') as f:
        configuration_data = json.load(f)
    
    with torch.no_grad():
        detect(configuration_data)
