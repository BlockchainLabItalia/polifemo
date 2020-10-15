import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from myInflux import MyInflux
from Person import Person
from MotionPrediction import associate_points

people = []

# Connect to InfluxDB
client = MyInflux("id1")


def count_people():
    global client

    people_in, people_out = 0,0
    going_in, going_out = 0,0

    for p in people:
        if p.position == 'IN':
            if p.crossed:
                going_in = going_in + 1
            people_in = people_in + 1
        elif p.position == 'OUT':
            if p.crossed:
                going_out = going_out + 1
            people_out = people_out + 1
    
    client.write_crossed(going_in, going_out)
    client.write_revealed(people_in, people_out)


def detect(save_img=False):
    out, source, weights, imgsz, line_orientation, line_position, position_in = \
        opt.output, opt.source, opt.weights, opt.img_size, opt.line_orientation, opt.line_position, opt.position_in
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16


    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names


    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    global people

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                s, im0 = '%g: ' % i, im0s[i].copy()
            else:
                s, im0 = '', im0s

            detected_peolple = []
            
            height, width = img.shape[2:]   
            s += '%gx%g ' % (height, width) # print string

            height0, width0 = im0.shape[0], im0.shape[1]   

            scale_height, scale_width = height/height0, width/width0 


            # Define Line

            if line_orientation == 'vertical':
                line = width*line_position
            else:
                line = height*line_position
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for xyxy in reversed(det):

                    if names[int(xyxy[-1])] == 'person':
                        
                        c1, c2 = (int(xyxy[0]) * scale_width, int(xyxy[1]) * scale_height), (int(xyxy[2]) * scale_width, int(xyxy[3]) * scale_height)
                        center = ((c1[0]+c2[0])/2, (c1[1]+c2[1])/2)

                        position = ''
                        if line_orientation == 'vertical':
                            if center[0] < line and position_in == 'left':
                                position = 'IN'
                            elif center[0] > line and position_in == 'right':
                                position = 'IN'
                            else:
                                position = 'OUT'
                        else:
                            if center[1] < line and position_in == 'top':
                                position = 'IN'
                            elif center[1] > line and position_in == 'bottom':
                                position = 'IN'
                            else:
                                position = 'OUT'

                        detected_peolple.append(Person(center,position))
                        print(position)


            if len(people) > 0 and len(detected_peolple) > 0:
                people = associate_points(people, detected_peolple, width, height)
            else:
                people = detected_peolple

            count_people()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))



    print('Done. (%.3fs)' % (time.time() - t0))


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
