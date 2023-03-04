# This dockerfile was originally based on mujoco-py's dockerfile.
# The Conda install parts are based on https://hub.docker.com/r/continuumio/miniconda3/dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04

# now install actual OS-level dependencies
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    curl \
    xserver-xorg-dev \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libegl1-mesa  \
    xvfb \
    rsync \
    gcc \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
  && chmod +x /usr/local/bin/patchelf

# tini is a simple init which is used by the official Conda Dockerfile (among
# other things). It can do stuff like reap zombie processes & forward signals
# (e.g. from "docker stop") to subprocesses. This may be useful if our code
# breaks in such a way that it creates lots of zombies or cannot easily be
# killed (e.g. maybe a Python extension segfaults and doesn't wait on its
# children, which keep running). That said, I (Sam) haven't yet run into a
# situation where it was necessary with our il-representations code base, at
# least as of October 2020.
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# create a user with the right UID etc.
ARG UID
ARG USERNAME
RUN groupadd -g "$UID" "$USERNAME"
RUN useradd -r -d /homedir -s /bin/bash -u "$UID" -g "$USERNAME" "$USERNAME"
RUN mkdir -p /homedir && chown -R "$USERNAME:$USERNAME" /homedir
USER $USERNAME
WORKDIR /homedir

# Install Conda and make it the default Python
ENV PATH /homedir/.local/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /homedir/conda.sh || true \
  && bash /homedir/conda.sh -b -p /homedir/conda || true \
  && /homedir/conda/bin/conda init --all --user \
  && rm /homedir/conda.sh
ENV BASH_ENV=/homedir/.bashrc
SHELL ["/bin/bash", "-c"]
RUN conda update -n base -c defaults conda \
  && conda install -c anaconda python=3.7 \
  && conda clean -ay

# FIXME(sam): MineRL deps disabled 2023-02-20 since we never had time to do
# MineRL expts for the EIRLI paper. Should decide whether we want to remove
# everything MineRL-related entirely.
# # MineRL installed separately because pip installs from Github don't work with submodules
# # COPY minecraft_setup.sh /root/minecraft_setup.sh
# # RUN bash /root/minecraft_setup.sh

# install MuJoCo 2.0 and a license key
RUN mkdir -p /homedir/.mujoco \
  && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
  && unzip mujoco.zip -d /homedir/.mujoco \
  && rm mujoco.zip \
  && wget https://www.roboti.us/file/mjkey.txt -O /homedir/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /homedir/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

# Install remaining dependencies
COPY requirements.txt /homedir/requirements.txt
COPY tp/ /homedir/tp/
USER root
RUN chown -R "$USERNAME:$USERNAME" /homedir/tp /homedir/requirements.txt
USER $USERNAME
RUN CFLAGS="-I/opt/conda/include" pip install --no-cache-dir -U "pip>=21.3.1,<22.0.0"
RUN CFLAGS="-I/opt/conda/include" pip install --no-cache-dir -r /homedir/requirements.txt
# also install CUDA 11.1 & Torch 1.10, since that seems to work with A100
RUN conda install pytorch=1.10.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y \
  && conda clean -ay

# copy in source
ARG SRC_TARBALL
COPY $SRC_TARBALL /homedir/eirli.tar.gz
USER root
RUN chown -R "$USERNAME:$USERNAME" /homedir/eirli.tar.gz
USER $USERNAME
RUN tar xf /homedir/eirli.tar.gz && rm /homedir/eirli.tar.gz
RUN cd /homedir/eirli && pip install --no-cache-dir -e .

# add scripts in cloud/ to $PATH
ENV PATH=/homedir/eirli/cloud:$PATH

# This is useful for making the X server work (but will break unless the X
# server is on the right port)
ENV DISPLAY=:0

ENTRYPOINT ["/homedir/eirli/cloud/entrypoint"]
