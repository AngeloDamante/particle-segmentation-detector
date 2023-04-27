FROM anibali/pytorch:2.0.0-cuda11.8

# setting
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true
USER root

# add utility layer
RUN apt-get update 
RUN apt-get -y upgrade 
RUN apt-get install -y --no-install-recommends \
    nano \
    wget \
    curl \
    xz-utils

# add python settings
ENV PYTHONPATH "${PYTHONPATH}:/src:/include:/plots"

# add vision layer
RUN apt-get -y update && apt-get -y install libgl1 && apt-get install -y ffmpeg libsm6 libxext6

# add project requirements layer
RUN apt-get update && pip3 install numpy \
	&& pip3 install Pillow \
	&& pip3 install matplotlib \
	&& pip3 install opencv-python==4.5.5.64 \
	&& pip3 install scikit-learn \
	&& pip3 install scipy \
	&& pip3 install pandas \
	&& pip3 install ipython \
	&& pip3 install jupyter \
	&& pip3 install tqdm
