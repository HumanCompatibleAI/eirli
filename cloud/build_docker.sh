#!/bin/bash

set -euo pipefail

# get dir containing this bash script (fall back to cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd || pwd)"
# dockerfile dir is level above this one
DOCKERFILE_DIR="$(dirname "$SCRIPT_DIR")"

# Argument defaults:
IMAGE_UID="$UID"
IMAGE_USERNAME="$USER"
DOCKER_HUB_USERNAME="humancompatibleai"
SHOULD_PUSH_TO_DOCKER_HUB=""

usage() {
    echo "Usage: $0 [-d <dockerhub_username>] [-p] [-n <username>] [-u <uid>] [-g <gid>]" 1>&2
    echo "  -d <username> Dockerhub username (default: $DOCKER_HUB_USERNAME)" 1>&2
    echo "  -p push to Dockerhub under username given with -d (default: false)" 1>&2
    echo "  -n <name> username in Docker image is <name> (default: $IMAGE_USERNAME)" 1>&2
    echo "  -u <uid> set UID/GID of user in Docker image to <uid> (default: $IMAGE_UID)" 1>&2
    echo "  -h print help" 1>&2
    echo "Note that this script must be run from the git repo, and the git repo must" 1>&2
    echo "be in a clean state (no uncommitted changes)." 1>&2
    exit 1
}

while getopts ":d:pn:u:h" o; do
    case "${o}" in
        d)
            DOCKER_HUB_USERNAME="${OPTARG}"
            ;;
        p)
            SHOULD_PUSH_TO_DOCKER_HUB="true"
            ;;
        n)
            IMAGE_USERNAME="${OPTARG}"
            ;;
        u)
            IMAGE_UID="${OPTARG}"
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# make sure there are no extra args
if [[ $# -gt $OPTIND ]]; then
    usage
fi

# check that there are no modifications to this repo
if [[ -n "$(git diff-index --name-only HEAD --)" ]]; then
    echo "There are uncommitted changes in $DOCKERFILE_DIR/src; commit them before building the Docker image" 1>&2
    exit 1
fi

# append a file to a .tar.gz file ($1), by reading from $2 and writing to $3
append_to_tar_gz() {
    # assert that we have 3 args
    if [[ $# -ne 3 ]]; then
        echo "append_to_tar_gz: expected 3 args, got $#"
        exit 1
    fi
    archive_path="$1"
    new_member_src_path="$2"
    new_member_dst_path="$3"

    # create a temporary directory
    tmp_dir="$(mktemp -d)"
    # extract the archive to the temporary directory
    tar -C "$tmp_dir" -xf "$archive_path"
    # copy the new member to the temporary directory
    cp "$new_member_src_path" "$tmp_dir/$new_member_dst_path"
    # rearchive the temporary directory to the archive path
    tar -C "$tmp_dir" -czf "$archive_path" .
    # delete the temporary directory
    rm -rf "$tmp_dir"
}

# generate a tarball of the current git repo for export to the Docker image
SRC_TARBALL="$SCRIPT_DIR/eirli.tar.gz" 
echo "Compressing source code at HEAD to $SRC_TARBALL" 1>&2
rm "$SRC_TARBALL" || true  # make anew
git archive --format=tar.gz --prefix=eirli/ -o "$SRC_TARBALL" HEAD
# compute path of tarball relative to DOCKERFILE_DIR
SRC_TARBALL_DOCKER_RELATIVE="$(realpath --relative-to="$DOCKERFILE_DIR" "$SRC_TARBALL")"

# get hash of git HEAD
git rev-parse --short HEAD > "$SCRIPT_DIR/git_hash.txt"
# append git_hash.txt to the tarball; path of the appended member should be eirli/git_hash.txt
append_to_tar_gz "$SRC_TARBALL" "$SCRIPT_DIR/git_hash.txt" "eirli/git_hash.txt"

IMAGE_NAME="$DOCKER_HUB_USERNAME/eirli-$IMAGE_USERNAME"
# tag is YYYY.MM.DD-rN, where N is the number of images with name $IMAGE_NAME
# that start with YYYY.MM.DD
YYYY_MM_DD="$(date '+%Y.%m.%d')"
NUM_DOCKER_TAGS="$(docker images --format '{{.Tag}}' "$IMAGE_NAME" | grep -c -E "^$YYYY_MM_DD-r[0-9]+$" || true)"
NEW_TAG="$YYYY_MM_DD-r$((NUM_DOCKER_TAGS + 1))"
# Build the Docker image in ../../, with args UID=$IMAGE_UID and
# USERNAME=$IMAGE_USERNAME. Name the image $IMAGE_NAME and tag it with both
# $NEW_TAG and :latest.
echo "Building Docker image $IMAGE_NAME:$NEW_TAG and $IMAGE_NAME:latest" 1>&2
docker build \
    --build-arg UID="$IMAGE_UID" \
    --build-arg USERNAME="$IMAGE_USERNAME" \
    --build-arg SRC_TARBALL="$SRC_TARBALL_DOCKER_RELATIVE" \
    -t "$IMAGE_NAME:$NEW_TAG" \
    -t "$IMAGE_NAME:latest" \
    "$DOCKERFILE_DIR"

# remove git_hash and eirli.tar.gz
rm "$SCRIPT_DIR/git_hash.txt" "$SRC_TARBALL" || true

if [[ -z "$SHOULD_PUSH_TO_DOCKER_HUB" ]]; then
    echo "Not pushing to Dockerhub (re-run with -p to push)" 1>&2
else
    # push both tags
    echo "Pushing Docker image $IMAGE_NAME:$NEW_TAG and $IMAGE_NAME:latest" 1>&2
    docker push "$IMAGE_NAME:$NEW_TAG"
    docker push "$IMAGE_NAME:latest"
fi

echo "Successfully built and pushed Docker image." 1>&2
echo "You can run it with either of the following tags:" 1>&2
echo "  $IMAGE_NAME:$NEW_TAG" 1>&2
echo "  $IMAGE_NAME:latest" 1>&2