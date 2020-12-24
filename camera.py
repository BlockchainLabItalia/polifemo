from threading import Thread
from myInflux import MyInflux
from Person import Person
from MotionPrediction import associate_points

from utils.torch_utils import time_synchronized
from utils.general import (
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh)
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

    def isLeft(self, a, b, c):
        return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])) > 0

    def run(self):
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        ##imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        imgsz = self.imgsz

        cudnn.benchmark = True  # set True to speed up constant image size inference
        

        
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        ##if half:
        ##    self.model.half()  # to FP16
            
        ##img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        ##_ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once


        while True:    
            dataset = LoadStreams(self.source, img_size=imgsz)
            try:
                for _, img, im0s, hasFrame in dataset:
                    if hasFrame:
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


                            height0, width0 = im0.shape[0], im0.shape[1]   
                            point1 = (0,0)
                            point2 = (0,0)
                            line = 0

                            # Define Line

                            if self.line_orientation == 'vertical':
                                line = width0*self.line_position
                            elif self.line_orientation == 'horizzontal':
                                line = height0*self.line_position
                            else:
                                point1 = (int(self.line_position["point1"]["x"]), int(self.line_position["point1"]["y"]))
                                point2 = (int(self.line_position["point2"]["x"]), int(self.line_position["point2"]["y"]))

                            if det is not None and len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                                for xyxy in reversed(det):

                                    if names[int(xyxy[-1])] == 'person':

                                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                        center = ((c1[0]+c2[0])/2, (c1[1]+c2[1])/2)

                                        position = ''
                                        if self.line_orientation == 'vertical':
                                            if center[0] < line and self.position_in == 'left':
                                                position = 'IN'
                                            elif center[0] > line and self.position_in == 'right':
                                                position = 'IN'
                                            else:
                                                position = 'OUT'
                                        elif self.line_orientation == 'horizzontal':
                                            if center[1] < line and self.position_in == 'top':
                                                position = 'IN'
                                            elif center[1] > line and self.position_in == 'bottom':
                                                position = 'IN'
                                            else:
                                                position = 'OUT'
                                        else:
                                            if self.isLeft(point1, point2, center) and self.position_in == 'left':
                                                position = 'IN'
                                            elif not self.isLeft(point1, point2, center) and self.position_in == 'right':
                                                position = 'IN'
                                            else:
                                                position = 'OUT'

                                        detected_peolple.append(Person(center,position))


                            if len(self.people) > 0 and len(detected_peolple) > 0:
                                self.people = associate_points(self.people, detected_peolple, width0, height0)
                            else:
                                self.people = detected_peolple

                            self.count_people()
                    else:
                        print('no image in this cycle')

            except Exception as e:
                print('an error occured')
                print(e)
                dataset.cap.release()
        # Print time (inference + NMS)
        print('%s Camera Closed.' % (self.cameraID))