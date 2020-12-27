#FROM ultralytics/yolov5:latest
#COPY . /usr/src/app
#WORKDIR /usr/src/app
#RUN apt update
#RUN apt install -y libgl1-mesa-glx
#RUN cat append_to_requirements.txt >> requirements.txt
#RUN pip install -r requirements.txt
#ENTRYPOINT ["python"]
#CMD ["count.py"]

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.03-py3

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt gsutil influxdb

RUN apt update
RUN apt install -y libgl1-mesa-glx

RUN apt install -y ffmpeg
RUN apt install -y python3-opencv

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

ENTRYPOINT ["python"]
CMD ["count.py"]