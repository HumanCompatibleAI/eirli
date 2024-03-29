.. _reproduction:


Reproduction of Benchmark Paper Experiments 
===========================================

Getting data and setting up an output directory for runs
--------------------------------------------------------

To reproduce our results, you will first need to download and extract our `demonstration
dataset
<https://drive.google.com/drive/folders/1TtadELS449ciefeyCoohYS4bOX3PrS1O?usp=share_link>`_
(around 8GiB). We'll assume that the extracted dataset directory is at
``/path/to/extracted/data``.

You'll also want to create a new directory to store the results of your runs.
This will need to hold ~200GiB if you run all the experiments (although it won't
need as much space if you only run one experiment at a time and delete output
files as you go).  We'll assume that this directory is at ``/path/to/runs/dir``.

Building the code
-----------------

The easiest way to use our code is as a Docker image. We have a script that
takes a snapshot of the git repository and bakes it into a Docker image with the
``Dockerfile`` in the root of this repo. The script can also set the UID and GID
inside the container to match the user's UID and GID on the host machine, so
that files written from inside the container will be owned by the user on the
host machine. To run the script, use the following command:

.. highlight:: bash

::

    cd /path/to/eirli-git-repository  # change the path
    # -u and -n control the UID and the username inside the container, respectively.
    # -d controls the start of the image name (e.g. -d foobar results in an
    # image calleed "foobar/<something>"). You can change this to your Docker
    # Hub username if you want to push to Docker Hub later on.
    ./cloud/build_docker.sh -u $UID -n $USER -d docker-hub-user

This may take a few minutes because it needs to install all the system-level and
Python-level dependencies in the Docker image. The final Docker image will be
about 17GiB.

You can verify that the Docker image was built correctly by running the
following command:

.. highlight:: bash

::

    docker run -it --rm docker-hub-user/eirli-$USER:latest \
        bash -c 'echo "Hello, world!" && pkill -9 ray'

This will start up an instance of  Ray (a distributed computing framework) and
an X server (for environment rendering) inside the container. This will produce
some messages and warnings.  About 30 seconds after that, it will print "Hello,
world!" and then exit.

Running the code
----------------

Once you've built a Docker image, you launch experiments in new Docker
containers:

.. highlight:: bash

::

    # configuration variables
    # (CHANGE PATHS!)
    path_to_extracted_data=/path/to/extracted/data
    path_to_store_runs=/path/to/runs/dir
    image_name=docker-hub-user/eirli-$USER:latest
    cpus=24
    memory=100g
    # can also use, e.g., device=1,2 or device=3 or device=0,3,4 etc.
    # (if using more GPUs, you should increase the CPU and memory limits
    # proportionally)
    gpus="device=0"

    run_in_docker() {
        docker run -it --rm \
            --cpus="$cpus" --memory="$memory" --gpus="$gpus" --shm-size=15g \
            --volume "$(realpath "$path_to_extracted_data"):/data:ro" \
            --volume "$(realpath "$path_to_store_runs"):/runs:rw" \
            "$image_name" "$@"
    }

    # Joint training experiments (dm_control, MAGICAL, Procgen)
    run_in_docker bc_jt_expts_dmc.sh /data /runs
    run_in_docker bc_jt_expts_magical.sh /data /runs
    run_in_docker bc_jt_expts_procgen.sh /data /runs

    # BC + pretrained repL experiments (dm_control, MAGICAL, Procgen)
    run_in_docker bc_pretrain_expts_dmc.sh /data /runs
    run_in_docker bc_pretrain_expts_magical.sh /data /runs
    run_in_docker bc_pretrain_expts_procgen.sh /data /runs

    # GAIL + pretrained repL experiments (dm_control, MAGICAL, Procgen)
    run_in_docker gail_expts_dmc.sh /data /runs
    run_in_docker gail_expts_magical.sh /data /runs
    run_in_docker gail_expts_procgen.sh /data /runs

On one A6000 GPU, each of these scripts will take somewhere between a few days
(for the GAIL experiments) and a couple of weeks (BC pretraining experiments) to
complete.

Read this if you have less than 40GiB of VRAM
+++++++++++++++++++++++++++++++++++++++++++++

These scripts were tuned to run on GPUs with 40GiB+ of memory, such as the A6000
or the 40G A100. If you have a GPU with less VRAM then you might run out of
memory, since each script runs 10+ jobs in parallel on each GPU. To run fewer
experiments per GPU, you can edit the job launch scripts in ``cloud/`` (e.g.
``bc_jt_expts_dmc.sh``). The specific section you need to edit looks like this:

.. highlight:: bash

::

    gpu_default=0.11
    declare -A gpu_overrides=(
        ["repl_tcpc8_192"]="0.16"
        ["repl_simclr_192"]="0.16"
    )

These variables indicate what fraction of the GPU memory to use for each job
(with overrides for some representation learning algorithms that use more
memory). You can increase these fractions to decrease the number of jobs per
GPU. Once you're done, you'll need to rebuild the Docker image and re-run the
above commands (the rebuild should be much faster because everything except the
EIRLI source code will have been cached by Docker).