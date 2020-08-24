# IREE Docker Configuration

This directory contains the Dockerfiles that specify the container images used
for IREE. Images are uploaded to
[Google Container Registry (GCR)](https://cloud.google.com/container-registry).

To build an image, use `docker build`, e.g.:

```shell
docker build build_tools/docker/cmake
```
