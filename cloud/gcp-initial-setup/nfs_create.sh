#!/bin/bash

# Create an NFS server on GCP.

set -e

echo "There should be an existing NFS server for our project. Check first "
echo "before running this, then comment out these lines in your local "
echo "checkout if you decide you actually need to run it."
exit 1


# Stuff below this line actually starts the server.

BRINGUP_MAX_RETRIES=10
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${THISDIR}/../nfs_config.sh"

echo "Creating server '$SERVER_NAME' with HD size $NFS_CAP in zone '$ZONE'"
gcloud beta filestore instances create "$SERVER_NAME" --zone="$ZONE" \
    --tier=BASIC_HDD --file-share=name="vol1",capacity="$NFS_CAP" \
    --network=name="default"

echo "Created. Server information:"
gcloud filestore instances describe "$SERVER_NAME" --zone="$ZONE"
server_ip="$(get_server_ip)"
if [ -z "$server_ip" ]; then
    echo "Count not get server IP from '$SERVER_NAME' (zone '$ZONE')"
    exit 1
else
    echo "Extracted server IP: ${server_ip}"
fi

echo "Creating client '$CLIENT_NAME' in zone '$ZONE'"
# g1-small gives us "0.5 of a core" (i.e. a core shared with another
# application), with capacity to burst to full CPU use very briefly (no idea how
# this differs from usual time slice allocation in, e.g., Linux).
gcloud compute instances create "${CLIENT_NAME}" --zone "$ZONE" \
    --image-project debian-cloud --image-family debian-10 \
    --machine-type g1-small --tags http-server,

echo "Waiting for client to come up"
for i in $(seq 1 $BRINGUP_MAX_RETRIES); do
    echo "Retry $i/${BRINGUP_MAX_RETRIES}"
    echo "echo test" | gcloud compute ssh "${CLIENT_NAME}" --zone "$ZONE"
    if [ "$?" = 0 ]; then
        success=1
        break
    fi
    echo "Connection failed, waiting..."
    sleep 5
done
if [ ! -z "$success" ]; then
    echo "Setting up mount on NFS client"
    gcloud compute ssh "${CLIENT_NAME}" --zone "$ZONE" <<EOF
sudo apt-get -y update && sudo apt-get -y install nfs-common
sudo mkdir -pv "${CLIENT_MOUNT_POINT}"
sudo mount "${server_ip}:/vol1" "${CLIENT_MOUNT_POINT}"
sudo chmod go+rw "${CLIENT_MOUNT_POINT}"
EOF
else
    echo "Client bringup failed after ${BRINGUP_MAX_RETRIES} attempts"
fi
