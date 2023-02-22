#!/bin/bash

set -euxo pipefail

# get dir containing this bash script (fall back to cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd || pwd)"
# dockerfile dir is two levels above this one
DOCKERFILE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# check that there are no modifications to the src dir of this repo (by applying
# git diff-index just to $DOCKERFILE_DIR/src)
if [[ -n "$(git diff-index --name-only HEAD -- "$DOCKERFILE_DIR"/{src,requirements.txt,setup.py})" ]]; then
    echo "There are uncommitted changes in $DOCKERFILE_DIR/src; commit them before building the Docker image"
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
rm "$SRC_TARBALL" || true  # make anew
git archive --format=tar.gz --prefix=eirli/ -o "$SRC_TARBALL" HEAD
# compute path of tarball relative to DOCKERFILE_DIR
SRC_TARBALL_DOCKER_RELATIVE="$(realpath --relative-to="$DOCKERFILE_DIR" "$SRC_TARBALL")"

# get hash of git HEAD
git rev-parse --short HEAD > "$SCRIPT_DIR/git_hash.txt"
# append git_hash.txt to the tarball; path of the appended member should be eirli/git_hash.txt
append_to_tar_gz "$SRC_TARBALL" "$SCRIPT_DIR/git_hash.txt" "eirli/git_hash.txt"

# image name is always the same
IMAGE_UID=10002
IMAGE_USERNAME=sam
IMAGE_NAME="humancompatibleai/eirli-hofvarpnir-$IMAGE_USERNAME"
# tag is YYYY.MM.DD-rN, where N is the number of images with name $IMAGE_NAME
# that start with YYYY.MM.DD
YYYY_MM_DD="$(date '+%Y.%m.%d')"
NUM_DOCKER_TAGS="$(docker images --format '{{.Tag}}' "$IMAGE_NAME" | grep -c -E "^$YYYY_MM_DD-r[0-9]+$" || true)"
NEW_TAG="$YYYY_MM_DD-r$((NUM_DOCKER_TAGS + 1))"
# Build the Docker image in ../../, with args UID=$IMAGE_UID and
# USERNAME=$IMAGE_USERNAME. Name the image $IMAGE_NAME and tag it with both
# $NEW_TAG and :latest.
DOCKER_BUILDKIT=1 docker build \
    --build-arg UID="$IMAGE_UID" \
    --build-arg USERNAME="$IMAGE_USERNAME" \
    --build-arg SRC_TARBALL="$SRC_TARBALL_DOCKER_RELATIVE" \
    -t "$IMAGE_NAME:$NEW_TAG" \
    -t "$IMAGE_NAME:latest" \
    "$DOCKERFILE_DIR"
# push both tags
docker push "$IMAGE_NAME:$NEW_TAG"
docker push "$IMAGE_NAME:latest"

# remove git_hash and eirli.tar.gz
rm "$SCRIPT_DIR/git_hash.txt" "$SRC_TARBALL" || true

set +x
echo "Successfully built and pushed Docker image $IMAGE_NAME:$NEW_TAG and $IMAGE_NAME:latest"