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

1. Copy the base cluster config to your own config file and customise it.
2. Spin up the cluster with the Ray autoscaler and set up some basic monitoring.
3. Submit jobs to the cluster/interact with the cluster.
4. Once the jobs are done, tear down the cluster. Importantly, this involves
   both an automated teardown step that you can run with one command, and a
   manual cleanup step where you remove stopped instances from the GCP cloud
   console).

Each of these steps is described separately in the following subsections.

### Step 1: Create your own cluster config


```bash
cd cloud  # this directory
cp gcp_cluster_template.yaml  gcp_cluster_mine.yaml
```


### Step 2: Spin up the cluster and set up logging

```bash
ray up gcp_cluster_mine.yaml -y
```

### Step 3: Submitting jobs


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

**TODO(sam)** finish this
