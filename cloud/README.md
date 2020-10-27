# Tools for spinning up GCP clusters

## Spinning up and using your own cluster

For the purpose of this document, a "cluster" is a set of machines managed by a
single copy of the Ray autoscaler. It consists of (1) a config file on your
local machine with the cluster details, (2) a head node running on GCP, and (3)
one or more (preemptible) worker nodes running on GCP. The intent behind our
cluster setup is that each of us should be able to run our own "private" cluster
and submit whatever jobs we want to that; the current cluster config does not
make it easy for several users to submit jobs to the same running cluster. Thus,
the typical usage patter for a cluster will look something like this:

1. Copy the base cluster config to your own config file and customize it.
2. Spin up the cluster with the Ray autoscaler and set up some basic monitoring.
3. Submit jobs to the cluster/interact with the cluster.
4. Once the jobs are done, tear down the cluster. Importantly, this involves
   both an automated teardown step that you can run with one command, and a
   manual cleanup step where you remove stopped instances from the GCP cloud
   console).

Each of these steps is described separately in the following subsections.

### Step 1: Create your own cluster config

You can create your own cluster config by copying the template in
`gcp_cluster_template.yaml` and then changing all the occurrences of
`XXX-CUSTOMIZE-THIS-XXX` in the template:

```bash
# this directory
cd cloud
# you can call it whatever you want (here it is gcp_cluster_mine.yaml)
cp gcp_cluster_template.yaml gcp_cluster_mine.yaml
# this command will highlight the parts that you need to change
grep -nHC 1 --color=always XXX-CUSTOMIZE-THIS-XXX gcp_cluster_mine.yaml
```

There are three parts you need to change. The first is the _name_ of the cluster:

```bash
gcp_cluster_template.yaml-5-# Unique identifier for head node and workers of this cluster
gcp_cluster_template.yaml:6:cluster_name: il-rep-XXX-CUSTOMIZE-THIS-XXX
gcp_cluster_template.yaml-7-
```

When Ray sorts through the list of GCP instances associated with our GCP project
ID, it will use the `cluster_name` to infer which resources belong to your
cluster. Thus I suggest using something like `il-rep-<your name>` that is (1)
likely to be unique to our project, and (2) will easily identify who is running
jobs on the GCloud dashboard. You should take care not to change the
`cluster_name` in your config file after spinning up a cluster, because then Ray
will be unable to detect which resources belong to the cluster when you try to
submit jobs to it or shut it down.

The next config option is the branch that you want to pull:

```bash
gcp_cluster_template.yaml-176-    || git -c core.sshCommand="ssh -v -i /root/.ssh/ssh_deploy_key -o StrictHostKeyChecking=no" \
gcp_cluster_template.yaml:177:      clone --depth 1 -b XXX-CUSTOMIZE-THIS-XXX \
gcp_cluster_template.yaml-178-      git@github.com:HumanCompatibleAI/il-representations.git ~/il-rep
```

If you want to run code on master, then you can change this to `master`.
Typically you'll be working on a development branch, though, so you should push
that branch to Github and then change the name in the config file to match. For
example, I'm using the GCP branch, so the line above is `clone --depth 1 -b gcp`
for me. Remember that machines on the cluster will be _pulling from Github_, not
using the code stored on your own machine, so you need to push all the changes
you want to use (e.g. Sacred configs) _before_ starting experiments.

The final config option to change is the name used to distinguish directories
for holding runs, which are all stored in the same place on our project NFS
volume:

```bash
gcp_cluster_template.yaml-207-    && ( test -e data || ln -vs "${CLIENT_MOUNT_POINT}/il-demos/" data )
gcp_cluster_template.yaml:208:    && cluster_name="cluster-XXX-CUSTOMIZE-THIS-XXX"
gcp_cluster_template.yaml-209-    && runs_path="${CLIENT_MOUNT_POINT}/cluster-data/${cluster_name}"
```

This can match the `cluster_name` key defined earlier in the file, but it does
not have to. Personally, I change this line to use a more descriptive name like
`cluster_name=cluster-2020-10-20-dm-control-cpc-runs`. That way I (and others)
can tell which clusters were started on which day, and why they were started.

### Step 2: Spin up the cluster and set up logging

Once you have a customized cluster config, you can use `ray up` to spin up the
cluster:

```bash
ray up gcp_cluster_mine.yaml -y
```

This will do a few things:

1. Set up a _head node_ on GCP. This is a non-preemptible GCP instance that will
   coordinate sending of jobs to _worker nodes_. The head node will
   automatically be configured to have the project NFS volume mounted in the
   correct place, to have Docker installed, to have fresh drivers, etc. etc.
   This will take a few minutes.
2. Set up an initial worker node. Worker nodes are preemptible, and can be shut
   down at any time. The total number of worker nodes should automatically scale
   up and down as you run more jobs. Worker nodes will be set up in almost
   exactly the same way as head nodes, except they will connect to the running
   Ray instance on the head node instead of starting their own.
3. In addition to setting up the head and worker nodes, Ray will also start a
   _Docker container_ on each node using our standard project image. All
   commands that you submit to the cluster later on will be run within the
   version of this Docker container that exists on each machine.

Once Ray finishes doing all of that, it will give you a helpful list of commands
you can run to manage the machine. Of particular interest is the command that tails the logs on the head node, which will look like this:

```
ray exec /path/to/gcp_cluster_mine.yaml 'tail -n 100 -f /tmp/ray/session_*/logs/monitor*'
```

If you run that, it will periodically print out the number of running nodes, the
status of the nodes (spinning up, running already, not responding, etc.), and
the amount of each Ray-tracked resource that is currently in use. For instance,
on a large cluster, I get log messages that look like this:

```
2020-10-27 23:02:24,416 INFO autoscaler.py:455 -- Cluster status: 6/5 target nodes (0 pending) (1 failed to update)
 - MostDelayedHeartbeats: {'10.138.15.205': 0.9886248111724854, '10.138.15.204': 0.35214829444885254, '10.138.15.203': 0.3520946502685547, '10.138.15.201': 0.3520395755767822, '10.138.15.199': 0.35198497772216797}
 - NodeIdleSeconds: Min=0 Mean=0 Max=0
 - NumNodesConnected: 7
 - NumNodesUsed: 4.65
 - ResourceUsage: 84.0/300.0 CPU, 16.8/25.0 GPU, 0.0/7.0 GPUType:V100, 0.0 GiB/757.28 GiB memory, 0.0 GiB/225.29 GiB object_store_memory
 - TimeSinceLastHeartbeat: Min=0 Mean=0 Max=0
 ```

It can be very helpful to keep these logs up in the background while you're
working with the cluster. For instance, if the `ResourceUsage` is stuck at zero
despite the fact that you have submitted jobs, then it probably means the jobs
are failing as soon as they start. Likewise, if usage for some resources is
always stuck at 100%, then that resource might be a bottleneck that you should
increase next time you start another cluster.

### Step 3: Submitting jobs

A "job" for the Ray autoscaler is a Python script that gets run on the head node
of a cluster. For instance, take this minimal example of a Python script that connects to a local Ray server (using the default Redis port) and runs a trivial Ray task:

```python
#!/usr/bin/env python3
import ray
# connect to already-running Ray cluster on the head node
ray.init(address='localhost:6379')
@ray.remote
def say_hello():
    """Simple Ray task to run"""
    return "Hello from Ray!"
retval = ray.get(say_hello.remote())
print(f"Done! Job returned '{retval}'")
```

If you put this in a file named `test_job.py`, then you could submit it with:

```bash
ray submit --tmux /path/to/gcp_cluster_mine.yaml test_job.py
```

Ray will do a few things here:

1. Upload `test_job.py` to the head node of the cluster.
2. Start a new tmux session inside the running Docker container on the head node
   (due to the `--tmux` option).
3. Run `test_job.py` inside the tmux session (without `--tmux`, it would run it
   in a plain Bash shell instead of a Bash shell inside a tmux session).
4. While executing `test_job.py`, the Ray server on the head node will be able
   to distribute each Ray task to any machine in the Ray cluster (which could
   mean the head node, or one of the worker nodes). For instance,
   Ray might choose to service the `say_hello.remote()` call by starting a task
   on one of the worker nodes.

The `ray submit` command that you run on your local machine will not tell you
the output of `test_job.py`. To get the output, you need to "attach" to the head
node of the Ray cluster and inspect the relevant tmux session. Sessions are
numbered sequentially from zero, and when I submitted that job I had already
submitted 16 jobs previously, so I had to do this by connecting to tmux session
16 on the head node:

```bash
ray attach /path/to/gcp_cluster_mine.yaml
# now on the head node:
tmux a -t 16
```

For me, the `tmux` session looks like this:

```
WARNING: Logging before InitGoogleLogging() is written to STDERR
… [some more cruft here] …
Done! Job returned 'Hello from Ray!'
… [some warnings] …
root@ray-il-rep-sam-test-head-8c55e007:/#
```

Now you know the basics of how to submit jobs to a Ray cluster. For this
project, I've written a job script in `cloud/submit_pretrain_n_adapt.py` that
looks like this:

```python
#!/usr/bin/env python3
import os, sys
from il_representations.scripts.pretrain_n_adapt import main
if __name__ == '__main__':
    os.chdir(os.path.expanduser('~/il-rep/'))
    main([sys.argv[0], 'run', 'with', 'ray_init_kwargs.address=localhost:6379', *sys.argv[1:]])
```

All this does is run `pretrain_n_adapt.py run with …` followed by any Sacred
config arguments you supply to `ray submit`. You can use it like this:

```bash
ray submit --tmux /path/to/gcp_cluster_mine.yaml ./submit_pretrain_n_adapt.py \
    -- cfg_base_3seed_1cpu_pt2gpu_2envs cfg_bench_micro_sweep_magical \
    cfg_repl_none cfg_il_bc_nofreeze exp_ident=example_run_with_ray
```

(this is a real example that I was using to run plain BC on MAGICAL, without any
repL)

The `--` after the path to `submit_pretrain_n_adapt.py` marks the remaining
arguments to `ray submit` as arguments for `submit_pretrain_n_adapt.py` on the
head node, rather than arguments for `ray submit` on your local machine. You can
see more examples of `ray submit` usage in the `submit_expts_*.sh` scripts in
the `cloud/` directory (e.g. `cloud/submit_expts_2020_10_27.sh`).

Again, once you've started repL jobs, you can inspect their output in the same
way as before: attach to the head node of the cluster, and then attach to one of
the running tmux sessions.

### Step 4: Spinning down the cluster

Once you're done with the cluster, you can terminate it with:

```bash
ray down gcp_cluster_mine.yaml -y
```

Running the cluster can be pretty expensive, so it's a good idea to keep an eye
on the [list of active
instances](https://console.cloud.google.com/compute/instances?project=methodical-tea-257021)
and [billing
dashboard](https://console.cloud.google.com/billing/01BA8D-DF00D1-BBF6AF/reports;grouping=GROUP_BY_SKU;projects=methodical-tea-257021;credits=NONE?project=methodical-tea-257021)
while you are running experiments. In particular, the following points are
**important to keep in mind while running clusters:**

1. Don't forget to spin down the cluster once you're done with it.
2. When worker instances are preempted by GCP, they will simply be "stopped"
   instead of being fully terminated. That means they will keep incurring
   charges for disk and some other resources. Thus you need to **manually delete
   stopped instances in GCP** from time to time.

## Spinning up and manipulating shared NFS storage

To share experiment results within CHAI, we have a 1TB shared Google Filestore
volume. This gets exposed as an NFS filesystem that can be mounted on Ray worker
nodes, etc. These instructions explain how to set up a new NFS volume. Note that
**I (Sam) set up a shared NFS volume for all of our clusters to use at the
beginning of this project, so you likely won't need to follow these instructions
again.** As explained below, the config for the GCP cluster is actually baked
into this repo, so if you create a new cluster then should automatically connect
to that volume. These instructions are just here in case somebody else wants to
"reboot" the project after we are done with it.

Most of the scripts for setting up NFS are in `cloud/gcp-initial-setup`,
although there's also one utility and one config file in `cloud`. To set up a
new NFS volume, follow these steps:

1. Edit `cloud/nfs_config.sh` to have an appropriate NFS volume size
   (`NFS_CAP`), GCP zone (`ZONE`), a unique server name for your project
   (`SERVER_NAME`), a unique client instance name (`CLIENT_NAME`), and a
   sensible mount point for the volume on the client (`CLIENT_MOUNT_POINT`).
   This `nfs_config.sh` script is used by just about every part of the
   experiment cluster, and you should commit your changes to the file to git so
   that it can be accessed on other machines (e.g. Ray workers).
2. Run `cloud/gcp-initial-setup/nfs_create.sh` to create a new Google Filestore
   volume with the chosen configuration options. This script will also create an
   ordinary GCP client that we'll use to access files on the Google Filestore
   volume. Google Filestore volumes cannot be accessed from outside of Google
   Cloud by default, so we're using this client machine as a proxy to the
   firewalled Google Filestore machine.
    
   By default `nfs_create.sh` just prints and error and exits to prevent people from
   accidentally creating new NFS volumes (you don't need to create a new volume
   to start a cluster! Just use the existing one!). You'll have to delete those
   lines at the beginning to actually run it.
3. Run `cloud/gcp-initial-setup/nfs_generate_mount_cmd.sh` to generate a new
   mount script for the volume. This script will be stored in
   `cloud/ray-init-scripts/nfs_mount.sh` (which should be committed to git!). It
   will be used by Ray head nodes and workers on GCP to mount the Google
   Filestore volumes.
4. Run `cloud/gcp-initial-setup/nfs_populate.sh` to copy the demonstrations that
   I (Sam) put on svm over to Google Cloud. Demonstrations are stored in the NFS
   volume so they can be accessed by all Ray workers.
5. If you want to be able to access the Filestore volume from outside GCP,
   you'll have to run `cloud/mount_nfs_volume_as_sshfs.sh` on each machine from
   where you want to access the volume. This will create a new sshfs mount that
   proxies file requests to the Filestore volume via the GCP client instance
   that we created in step (2). For instance, I ran
   `bash mount_nfs_volume_as_sshfs.sh
   /scratch/sam/il-representations-gcp-volume` to mount an NFS volume on
   perceptron (likewise for svm and astar, IIRC).
