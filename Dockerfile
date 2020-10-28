FROM ultralytics/yolov5:latest
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN apt update
RUN apt install -y libgl1-mesa-glx
RUN cat append_to_requirements.txt >> requirements.txt
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["count.py"]