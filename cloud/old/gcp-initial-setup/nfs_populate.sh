#!/bin/bash

# Populate NFS server on GCP with data from svm/perceptron (including MuJoCo
# key, and all relevant demonstrations).

# This is Linux-only, because it depends on sshfs.

set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${THIS_DIR}/../nfs_config.sh"

if [ -e "${HOME}/google-cloud-sdk/" ]; then
    # for running under Docker + GCP with Ray autoscaler
    GCLOUD_PREFIX="${HOME}/google-cloud-sdk/bin/"
else
    GCLOUD_PREFIX=""
fi
TMP_DIR=_transfer_scratch
CHAI_USERNAME="$(whoami)"
CHAI_MACHINE="svm.bair.berkeley.edu"
COPY_CMD=""

echo "Will mount machine '$CHAI_MACHINE' on '$TMP_DIR'"

if [ -e "$TMP_DIR" ]; then
    echo "Mount point '$TMP_DIR' already exists. Perhaps you need to unmount"\
        "with 'fusermount -uz $TMP_DIR && rmdir $TMP_DIR'?"
    exit 1
fi

echo "Creating mountpoint"
mkdir -p "$TMP_DIR"
echo "Mounting"
sshfs "${CHAI_USERNAME}@${CHAI_MACHINE}:/" "$TMP_DIR" -o reconnect,follow_symlinks
echo "Copying files"
# we copy the user's mjkey.txt (whoever is mentioned in ${CHAI_USERNAME}) along
# with Sam's il-demos directory
gcloud compute scp --recurse --zone "${ZONE}"\
    "${TMP_DIR}/home/${CHAI_USERNAME}/.mujoco/mjkey.txt" \
    "${TMP_DIR}/scratch/sam/il-demos/" \
    "${CLIENT_NAME}:${CLIENT_MOUNT_POINT}"

echo "Unmounting filesystem from ${CHAI_MACHINE} and removing mount point"
fusermount -uz "$TMP_DIR" && rmdir "$TMP_DIR"
