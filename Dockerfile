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

RUN apt-get update ; \
    apt-get install -y git build-essential gcc make yasm autoconf automake \
    cmake libtool checkinstall libmp3lame-dev pkg-config libunwind-dev \
    zlib1g-dev libssl-dev libgl1-mesa-glx

RUN apt-get update \
    && apt-get clean \
    && apt-get install -y --no-install-recommends libc6-dev libgdiplus wget software-properties-common

#RUN RUN apt-add-repository ppa:git-core/ppa && apt-get update && apt-get install -y git

RUN wget https://www.ffmpeg.org/releases/ffmpeg-4.0.2.tar.gz
RUN tar -xzf ffmpeg-4.0.2.tar.gz; rm -r ffmpeg-4.0.2.tar.gz
RUN cd ./ffmpeg-4.0.2; ./configure --enable-gpl --enable-libmp3lame --enable-decoder=mjpeg,png \
    --enable-encoder=png --enable-openssl --enable-nonfree


RUN cd ./ffmpeg-4.0.2; make
RUN cd ./ffmpeg-4.0.2; make install

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

ENTRYPOINT ["python"]
CMD ["count.py"]