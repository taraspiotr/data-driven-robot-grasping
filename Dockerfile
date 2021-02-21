FROM nvidia/cudagl:10.1-base

WORKDIR /auxiliary_tasks
COPY ./requirements.txt .

RUN apt-get update -q \
    && apt-get install -y libglew2.0 wget vim python3.7 unzip virtualenv
RUN apt-get install -y python3-pip cmake python3.7-dev git
RUN python3.7 -m pip install --upgrade pip \
    && python3.7 -m pip install -r requirements.txt \
    && python3.7 -m pip install git+https://github.com/astooke/rlpyt