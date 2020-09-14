# Adaptation of the base CircleCI dockerfile so that it works better our Ray GCP
# experiments. Specific changes:
#
# - Adds Conda and makes it the default Python interpreter.
# - Preinstalls all Python dependencies so that Ray doesn't have to do that each
#   it spins up a worker node.
# - Possibly does some CUDA stuff? IDK how I'm going to deal with that.

FROM humancompatibleai/il-representations:latest

ENV ENV /root/.profile
