#!/bin/bash

# Mount GCP Filestore volume to a local ssh mount by proxying through the client
# machine I set up on GCP. This is useful when (1) you already have the Google
# Cloud SDK installed and authenticated, and (2) you want to inspect the
# experiment data as a local filesystem.

set -e

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${THISDIR}/nfs_config.sh"

if [ $# -ne 1 ]; then
    echo "USAGE: $0 <mount-path>"
    exit 1
fi
mount_path="$1"
shift

echo "Getting SSH info from GCP"
# get the ssh command used by 'gcloud compute ssh', but strip out the -t flag
# (which allocates a tty) and the username/machine
gcp_ssh_cmd_raw="$(gcloud compute ssh --dry-run --zone "$ZONE" "$CLIENT_NAME")"
ssh_cmd="$(echo "$gcp_ssh_cmd_raw" | sed 's/ -t / /' | sed 's/ [^ ]\+$//')"
# now extract just the final user@machine part (separated by the last space;
# note GCP usernames cannot have spaces)
user_host="$(echo "$gcp_ssh_cmd_raw" | sed 's/^.* \([^ ]\+\)$/\1/')"

echo "Will mount '${user_host}:${CLIENT_MOUNT_POINT}' on '$mount_path' "\
"using ssh command '$ssh_cmd'"

mkdir -p "$mount_path"
sshfs -o reconnect,follow_symlinks -o ssh_command="$ssh_cmd" \
    "${user_host}:${CLIENT_MOUNT_POINT}" "$mount_path"

echo "Done, user \"fusermount -uz '$mount_path'\" to unmount."
