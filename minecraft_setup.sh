# A script to run within Docker to setup Minecraft
add-apt-repository ppa:openjdk-r/ppa
apt-get update

apt-get purge openjdk-8*

apt-get install -y openjdk-8-jre-headless=8u162-b12-1
apt-get install -y openjdk-8-jdk-headless=8u162-b12-1
apt-get install -y openjdk-8-jre=8u162-b12-1
apt-get install -y openjdk-8-jdk=8u162-b12-1

git clone --recurse-submodules https://github.com/HumanCompatibleAI/minerl.git
cd minerl
CFLAGS="-I/opt/conda/include" pip install --no-cache-dir -e .