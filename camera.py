from threading import Thread
from myInflux import MyInflux
from Person import Person
from MotionPrediction import associate_points

from utils.torch_utils import time_synchronized
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh)
import torch
import torch.backends.cudnn as cudnn
from utils.datasets import LoadStreams, LoadImages

class Camera (Thread):
    def __init__(self, cameraID, model, device, imgsz, ip_address, line_orientation, line_position, position_in, influx_config):
        Thread.__init__(self)
        self.people = []
        self.cameraID = cameraID
        self.db_client = MyInflux(cameraID, influx_config["hostname"],influx_config["database_name"],influx_config["port"])
        self.model = model
        self.imgsz = imgsz
        self.source = ip_address
        self.device = device
        self.line_orientation = line_orientation
        self.line_position = line_position
        self.position_in = position_in

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
    
        self.db_client.write_crossed(going_in, going_out)
        self.db_client.write_revealed(people_in, people_out)

    def run(self):
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        ##imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        imgsz = self.imgsz

        cudnn.benchmark = True  # set True to speed up constant image size inference

        dataset = LoadStreams(self.source, img_size=imgsz)

        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        ##if half:
        ##    self.model.half()  # to FP16
            
        ##img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        ##_ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once


        for _, img, im0s in dataset:
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
            for i, det in enumerate(pred):  # detections per image
                im0 =  im0s[i].copy()

                detected_peolple = []

                height, width = img.shape[2:]  

                height0, width0 = im0.shape[0], im0.shape[1]   

                scale_height, scale_width = height/height0, width/width0 


                # Define Line

                if self.line_orientation == 'vertical':
                    line = width*self.line_position
                else:
                    line = height*self.line_position

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for xyxy in reversed(det):

                        if names[int(xyxy[-1])] == 'person':

                            c1, c2 = (int(xyxy[0]) * scale_width, int(xyxy[1]) * scale_height), (int(xyxy[2]) * scale_width, int(xyxy[3]) * scale_height)
                            center = ((c1[0]+c2[0])/2, (c1[1]+c2[1])/2)

                            position = ''
                            if self.line_orientation == 'vertical':
                                if center[0] < line and self.position_in == 'left':
                                    position = 'IN'
                                elif center[0] > line and self.position_in == 'right':
                                    position = 'IN'
                                else:
                                    position = 'OUT'
                            else:
                                if center[1] < line and self.position_in == 'top':
                                    position = 'IN'
                                elif center[1] > line and self.position_in == 'bottom':
                                    position = 'IN'
                                else:
                                    position = 'OUT'

                            detected_peolple.append(Person(center,position))


                if len(self.people) > 0 and len(detected_peolple) > 0:
                    self.people = associate_points(self.people, detected_peolple, width, height)
                else:
                    self.people = detected_peolple

                self.count_people()

        # Print time (inference + NMS)
        print('%s Done.' % (self.cameraID))





