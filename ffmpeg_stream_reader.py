import os
import time

from threading import Thread

import cv2
import numpy as np

import ffmpeg

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')

            process = (
                ffmpeg
                .input(s)
                .filter('scale', size='hd720')
                .output('-', format='rawvideo', pix_fmt='bgr24')
                .run_async(pipe_stdout=True)
                )

            # self.w, self.h = 1920, 1080 # filter -> size = 'hd1080'
            self.w, self.h = 1280, 720 # filter -> size = 'hd720'
            dim = self.w * self.h * 3
            
            time.sleep(5)

            frame = process.stdout.read(dim)  # guarantee first frame

            self.hasFrame = (len(frame) == dim)

            assert self.hasFrame, 'Failed to open %s' % s

            frame = np.frombuffer(frame, dtype=np.uint8, count=-1)
            self.imgs[i] = np.reshape(frame, (self.h, self.w, 3))

            thread = Thread(target=self.update, args=([i, process, dim]), daemon=True)
            print(' success (%gx%g).' % (self.w, self.h))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap, dim):
        # Read next stream frame in a daemon thread
        while cap.poll() is None:

            frame = cap.stdout.read(dim)  
            self.hasFrame =  (len(frame) == dim)

            if not self.hasFrame:
                break 
            
            frame = np.frombuffer(frame, dtype=np.uint8, count=-1)
            self.imgs[index] = np.reshape(frame, (self.h, self.w, 3))
            time.sleep(0.01)  # wait time
        print('video capture stopped')

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        #if cv2.waitKey(1) == ord('q'):  # q to quit
        #    cv2.destroyAllWindows()
        #    raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, self.hasFrame

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

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