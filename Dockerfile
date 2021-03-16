FROM nvidia/cudagl:10.1-base

WORKDIR /auxiliary_tasks
COPY ./requirements.txt .

RUN DEBIAN_FRONTEND=noninteractive apt-get update -q \
    && apt-get install -y wget vim python3.8 unzip virtualenv
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip cmake python3.8-dev git
RUN DEBIAN_FRONTEND=noninteractive python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install -r requirements.txt
