FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime


RUN apt-get update && apt-get upgrade -y
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# OpenCV
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install opencv-python
RUN pip install opencv-contrib-python

# pip
WORKDIR /tmp
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# torch
RUN pip install torchsummary
RUN pip install torch_optimizer

# clearn
RUN rm -rf /var/lib/apt/lists/*
