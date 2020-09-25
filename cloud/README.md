# Tools for spinning up GCP clusters

## Spinning up and manipulating shared NFS storage

To share experiment results within CHAI, we have a 1TB shared Google Filestore
volume. This gets exposed as an NFS filesystem that can be mounted on Ray worker
nodes, etc.

**TODO(sam):** eventually write a comprehensive README so that I'm not the only
one who knows how to spin this thing up.
