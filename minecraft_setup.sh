# A script to run within Docker to setup Minecraft
add-apt-repository ppa:openjdk-r/ppa
apt-get update

apt-get purge openjdk-8*

apt-get install openjdk-8-jre-headless=8u162-b12-1
apt-get install openjdk-8-jdk-headless=8u162-b12-1
apt-get install openjdk-8-jre=8u162-b12-1
apt-get install openjdk-8-jdk=8u162-b12-1
apt-get install -y openjdk-8-jdk

git clone --recurse-submodules https://github.com/decodyng/minerl.git
cd minerl
git checkout limited_data_iterator
CFLAGS="-I/opt/conda/include" pip install --no-cache-dir -e .