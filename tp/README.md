This directory contains vendored third-party libraries. Consult the GIT_VERSION
file in each subdirectory to see which version of the library our copy was
forked from. These files are **baked into the Dockerfile** like the rest of the
dependencies, so if you change them but then run under an old Docker image then
your changes will be ignored.