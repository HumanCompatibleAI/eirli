#!/bin/bash

# Generate command for mounting the project NFS store. We bake the output into
# nfs_mount.sh so that we don't have to install the Google Cloud SDK on our
# Ray worker nodes (which would require some annoying credential distribution
# steps).

set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${THIS_DIR}/../nfs_config.sh"

echo "Inferring IP address of server '$SERVER_NAME' (zone '$ZONE')"
server_ip="$(get_server_ip)"
if [ -z "$server_ip" ]; then
    echo "Could not read server IP from 'gcloud filestore instances describe'. "\
        "Do the server ('$SERVER_NAME') and zone ('$ZONE') values make sense?"
    exit 1
fi

echo -e "Got IP '$server_ip'. Here is the generated script:\n\n"
mkdir -p "${THIS_DIR}/../ray-init-scripts/"
echo "vvvvvvvvvv (output file) vvvvvvvvvv"
tee "${THIS_DIR}/../ray-init-scripts/nfs_mount.sh" <<EOF
#!/bin/bash
# SCRIPT AUTOGENERATED BY $(basename "$0"); DO NOT EDIT MANUALLY

# This is a script to mount the il-representations project filesystem from
# server '$SERVER_NAME' (zone '$ZONE'). It will only work on GCP.

set -e

if mountpoint -q '${CLIENT_MOUNT_POINT}'; then
    echo "'${CLIENT_MOUNT_POINT}' is already a mountpoint; skipping remount"
    exit 0
fi

# Retries a command a configurable number of times with backoff.
#
# The retry count is given by ATTEMPTS (default 10), the initial backoff
# timeout is given by TIMEOUT in seconds (default 4.)
#
# Backoffs multiply the timeout by 2-3 (extra jitter is chosen randomly)
#
# Script from https://stackoverflow.com/a/8351489, modified for random backoff
# and to change defaults.
function with_backoff {
  local max_attempts=\${ATTEMPTS-12}
  local timeout=\${TIMEOUT-4}
  local attempt=1
  local exitCode=0

  while (( attempt < max_attempts ))
  do
    if "\$@"
    then
      return 0
    else
      exitCode="\$?"
    fi

    echo "Failure! Retrying in \$timeout.." 1>&2
    sleep "\$timeout"
    attempt="\$(( attempt + 1 ))"
    timeout="\$(( 2 * timeout + timeout * RANDOM / 32767 ))"
  done

  if [[ "\$exitCode" != 0 ]]
  then
    echo "Command failed after \$attempt attempts (\$*)" 1>&2
  fi

  return "\$exitCode"
}

if [ -z "\$(cat /proc/filesystems | grep 'nfsd\$')" ]; then
    echo "This machine does not seem to have NFS support. Will attempt to"\\
        "install NFS packages."
    with_backoff apt-get update -y && with_backoff apt-get install -y nfs-common
fi

mkdir -p '$CLIENT_MOUNT_POINT' \\
    && echo '$server_ip:/vol1' '$CLIENT_MOUNT_POINT' nfs defaults,_netdev 0 0 >> /etc/fstab \\
    && mount -a \\
    && chmod go+rw '$CLIENT_MOUNT_POINT'

echo "Done! Mount should be accessible on '$CLIENT_MOUNT_POINT'"
EOF
echo "^^^^^^^^^^ (output file) ^^^^^^^^^^"
