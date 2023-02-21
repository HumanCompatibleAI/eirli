#!/bin/bash

set -euxo pipefail

# get dir containing this bash script (fall back to cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd || pwd)"
# dockerfile dir is two levels above this one
DOCKERFILE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# check that there are no modifications to the src dir of this repo (by applying
# git diff-index just to $DOCKERFILE_DIR/src)
if [[ -n "$(git diff-index --name-only HEAD -- "$DOCKERFILE_DIR/src")" ]]; then
    echo "There are uncommitted changes in $DOCKERFILE_DIR/src; commit them before building the Docker image"
    exit 1
fi

# generate a tarball of the current git repo for export to the Docker image
SRC_TARBALL="$SCRIPT_DIR/eirli.tar.gz" 
git archive --format=tar.gz --prefix=eirli/ -o "$SRC_TARBALL" HEAD

# image name is always the same
IMAGE_UID=10002
IMAGE_USERNAME=sam
IMAGE_NAME="humancompatibleai/eirli-hofvarpnir-$IMAGE_USERNAME"
# tag is YYYY.MM.DD-rN, where N is the number of images with name $IMAGE_NAME
# that start with YYYY.MM.DD
YYYY_MM_DD="$(date '+%Y.%m.%d')"
NUM_DOCKER_TAGS="$(docker images --format '{{.Tag}}' "$IMAGE_NAME" | grep -c -E "^$YYYY_MM_DD-r[0-9]+$" || true)"
NEW_TAG="$YYYY_MM_DD-r$((NUM_DOCKER_TAGS + 1))"
# Build the Docker image in ../../ at stage `hofvarpnir`, with args
# UID=$IMAGE_UID and USERNAME=$IMAGE_USERNAME. Name the image $IMAGE_NAME and
# tag it with both $NEW_TAG and :latest.
docker build \
    --target hofvarpnir \
    --build-arg UID="$IMAGE_UID" \
    --build-arg USERNAME="$IMAGE_USERNAME" \
    --build-arg SRC_TARBALL="$SRC_TARBALL" \
    -t "$IMAGE_NAME:$NEW_TAG" \
    -t "$IMAGE_NAME:latest" \
    "$DOCKERFILE_DIR"
# push both tags
docker push "$IMAGE_NAME:$NEW_TAG"
docker push "$IMAGE_NAME:latest"