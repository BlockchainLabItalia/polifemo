import cv2
import queue
from threading import Thread
import time
import numpy as np
from myInflux import MyInflux
from Person import Person
from MotionPrediction import associate_points

from utils.torch_utils import time_synchronized
from utils.general import (
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh)
import torch
import torch.backends.cudnn as cudnn


class CameraStream :
    def __init__(self, rtsp_address, model, device, line_orientation, line_position, position_in, db, img_size=640, cameraID="stream"):
        self.mode = 'images'
        self.img_size = img_size
        self.source = rtsp_address
        self.name = cameraID
        self.model = model
        self.device = device
        self.line_orientation = line_orientation
        self.position_in = position_in
        self.line_position = line_position
        self.db = db


    def start_processing(self):
        self.queue = queue.Queue()
        self.people = []

        print('%s: %s... ' % (self.name, self.source))
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened(), 'Failed to open %s' % self.source
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100
        assert cap.grab(), 'No frame from %s' % self.source  # guarantee first frame
        thread1 = Thread(target=self.update, args=([cap]), daemon=True)
        print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
        thread1.start()

        point1 = point2 = (0, 0)

        if self.line_orientation == 'vertical':
            point1 = (w*self.line_position, 0)
            point1 = (w*self.line_position, h)
        elif self.line_orientation == 'horizzontal':
            point1 = (0, int(h*self.line_position))
            point1 = (w, int(h*self.line_position))
        else:
            point1 = (int(self.line_position["point1"]["x"]), int(self.line_position["point1"]["y"]))
            point2 = (int(self.line_position["point2"]["x"]), int(self.line_position["point2"]["y"]))

        self.line = (point1, point2)

        thread2 = Thread(target=self.execute_analisys, daemon=True)
        thread2.start()
        thread1.join()
        thread2.join()

    def update(self, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            _ = cap.grab()
            if n == 4 and _:   # read every 4th frame
                _, img0 = cap.retrieve()
                self.queue.put(img0)
                n = 0
            time.sleep(0.06)  # wait time
        print('video capture stopped')

    def count_people(self):

        people_in, people_out = 0,0
        going_in, going_out = 0,0

        for p in self.people:
            if p.position == 'IN':
                if p.crossed:
                    going_in = going_in + 1
                people_in = people_in + 1
            elif p.position == 'OUT':
                if p.crossed:
                    going_out = going_out + 1
                people_out = people_out + 1

        if going_in or going_out:
            self.db.write_crossed(going_in, going_out, self.name)
        if people_in or people_out:
            self.db.write_revealed(people_in, people_out, self.name)

    def execute_analisys(self):
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        cudnn.benchmark = True  # set True to speed up constant image size inference
        

        
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        ##if half:
        ##    self.model.half()  # to FP16
            
        ##img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        ##_ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once


        while True:    
            if not self.queue.empty():
                img0 = self.queue.get()
                im0_shape = img0.shape
                # Letterbox
                img = [letterbox(img0, new_shape=self.img_size)[0]]

                # Stack
                img = np.stack(img, 0)

                # Convert
                img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(self.device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = self.model(img, augment=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, 0.4, 0.5, None, agnostic=False)




                # Process detections
                for _, det in enumerate(pred):  # detections per image

                    detected_peolple = []

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_shape).round()

                        for xyxy in reversed(det):

                            if names[int(xyxy[-1])] == 'person':

                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                center = ((c1[0]+c2[0])/2, (c1[1]+c2[1])/2)

                                position = ''
                                if self.line_orientation == 'vertical':
                                    if center[0] < self.line[0][0] and self.position_in == 'left':
                                        position = 'IN'
                                    elif center[0] > self.line[0][0] and self.position_in == 'right':
                                        position = 'IN'
                                    else:
                                        position = 'OUT'
                                elif self.line_orientation == 'horizzontal':
                                    if center[1] < self.line[0][1] and self.position_in == 'top':
                                        position = 'IN'
                                    elif center[1] > self.line[0][1] and self.position_in == 'bottom':
                                        position = 'IN'
                                    else:
                                        position = 'OUT'
                                else:
                                    if isLeft(self.line[0], self.line[1], center) and self.position_in == 'left':
                                        position = 'IN'
                                    elif not isLeft(self.line[0], self.line[1], center) and self.position_in == 'right':
                                        position = 'IN'
                                    else:
                                        position = 'OUT'

                                detected_peolple.append(Person(center,position))


                    if len(self.people) > 0 and len(detected_peolple) > 0:
                        self.people = associate_points(self.people, detected_peolple, im0_shape[0], im0_shape[1])
                    else:
                        self.people = detected_peolple

                    Thread(target=self.count_people(), daemon=True).start()
                    self.queue.task_done()
                    #print('execute_analisys done. remaining %g element in the queue' % self.queue_2.qsize())



def isLeft(a, b, c):
    return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)