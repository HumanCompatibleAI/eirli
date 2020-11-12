#!/bin/bash

# Generate a deploy key for use with Github

set -e

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
KEYURL="https://github.com/HumanCompatibleAI/il-representations/settings/keys"

# make a new key with no passphrase
mkdir -p "${THISDIR}/../ray-init-scripts/"
key_path="${THISDIR}/../ray-init-scripts/ssh_deploy_key"
if [ ! -f "$key_path" ]; then
    # make a passwordless deploy key
    echo "Generating new deploy key in '$key_path'"
    ssh-keygen -C "ray-deploy-$(whoami)@$(hostname)" -t ed25519 -a 100 \
        -N "" -f "$key_path"
    echo "Your public deploy key is shown below. Please add it to $KEYURL"
    cat "${key_path}.pub"
else
    echo "Key already exists in $key_path. Make sure it's in $KEYURL"
fi
