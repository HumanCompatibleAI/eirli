# Configuration values for NFS (this is used by other scripts)
# (If you want to edit this, mention it on #il-representations first. We should
# all be using the same NFS server if we want to see each other's files, and
# should be able to make do with just one shared client machine.)

NFS_CAP="1TB"
ZONE="us-west1-b"
SERVER_NAME="repl-nfs-server"
CLIENT_NAME="repl-nfs-client"
CLIENT_MOUNT_POINT="/data/il-representations/"

get_server_ip() {
    # get the IP address of NFS server at ${SERVER_NAME}
    gcloud filestore instances describe "$SERVER_NAME" \
           --zone="$ZONE" --format text \
    | grep '^networks\[0\].ipAddresses\[0\]:' \
    | tr -d ' ' \
    | cut -d : -f 2-
}
