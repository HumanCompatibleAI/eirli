#!/bin/bash

# A script to convert ../README.md (which appears on Github) to
# docs/README_former.rst (which appears on ReadTheDocs). You need pandoc to run
# this.

set -euf -o pipefail

pandoc -f markdown -t rst ../README.md -o source/README_former.rst --wrap=none
# one of the figures is a link to a file on docs/src; we replace this
sed -i -e 's-^.. figure:: docs/source/-.. figure:: -g' source/README_former.rst
