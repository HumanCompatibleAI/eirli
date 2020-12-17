add-apt-repository ppa:openjdk-r/ppa
apt-get update
apt-get install -y openjdk-8-jdk

git clone --recurse-submodules https://github.com/decodyng/minerl.git
cd minerl
git checkout limited_data_iterator
CFLAGS="-I/opt/conda/include" pip install --no-cache-dir -e .