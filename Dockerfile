# Based on mujoco-py's Dockerfile, but with the following changes:
# - No nvidia stuff (so no GPU support, but CPU rendering is still there)
# - Uses Conda Python 3.7 instead of Python 3.6
# The Conda bits are based on https://hub.docker.com/r/continuumio/miniconda3/dockerfile
FROM ubuntu:18.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
  && chmod +x /usr/local/bin/patchelf

RUN mkdir -p /root/.mujoco \
  && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
  && unzip mujoco.zip -d /root/.mujoco \
  && rm mujoco.zip
# fake MuJoCo key; we'll add a real one at run time
RUN touch /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

# This is a simple init copied from Conda dockerfile. Zombie-reaping
# capabilities are probably useful for any Dockerfile.
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Install Conda and make it the default Python
ENV PATH /opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/conda.sh || true \
  && bash /root/conda.sh -b -p /opt/conda || true \
  && rm /root/conda.sh
RUN conda update -n base -c defaults conda \
  && conda install -c anaconda python=3.7 cudatoolkit=11 \
  && conda clean -ay

# Install dependencies
COPY requirements.txt /root/requirements.txt
RUN CFLAGS="-I/opt/conda/include" pip install --no-cache-dir -r /root/requirements.txt

# Always run under simple init
ENTRYPOINT [ "/usr/bin/tini", "--" ]
