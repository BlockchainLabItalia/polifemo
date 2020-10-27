FROM ultralytics/yolov5:latest
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["count.py","--source", "rtsp://192.168.8.193/11", "--device", "0"]