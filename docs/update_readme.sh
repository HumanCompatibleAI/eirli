#!/bin/bash

# A script to convert ../README.md (which appears on Github) to
# docs/README_former.rst (which appears on ReadTheDocs). You need pandoc to run
# this.

set -euf -o pipefail

pandoc -f markdown -t rst --wrap=none '../README.md' -o 'source/README_former.rst'
# One of the figures is a link to a file in `docs/source/`. We remove the
# `docs/source/` so that the link is relative to the Sphinx document root.
sed -i -e 's-^.. figure:: docs/source/-.. figure:: -g' 'source/README_former.rst'
