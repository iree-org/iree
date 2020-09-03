# IREE Docker Configuration

This directory contains the Dockerfiles that specify the container images used
for IREE. Images are uploaded to
[Google Container Registry (GCR)](https://cloud.google.com/container-registry).

To build an image, use `docker build`, e.g.:

```shell
docker build build_tools/docker/cmake --tag cmake
```

To explore an image interactively, use `docker run`, e.g.

```shell
docker run --interactive --tty --rm cmake
```

You can find more information in the
[official Docker docs](https://docs.docker.com/get-started/overview/).

IREE images follow a consistent structure. The image defined by
`build_tools/docker/foo-bar/Dockerfile` is uploaded to GCR as
`gcr.io/iree-oss/foo-bar`. It may be tagged as `latest` or `prod`, e.g.
`gcr.io/iree-oss/foo-bar:latest`. Dockerfile image definitions should list their
dependencies based on these image names.

We use a helper python script to manage the Docker image deployment. It lists
all images and their dependencies. To build an image and all images it depends on:

```shell
python3 build_tools/docker/build_and_update_gcr.py --image cmake
```

To build multiple images

```shell
python3 build_tools/docker/build_and_update_gcr.py --image cmake --image bazel
```

There is also the special option `--image all` to build all registered images.

Pushing images to GCR requires the `Storage Admin` role in the `iree-oss` GCP
project. To push these images to GCR with the `latest` tag:

```shell
python3 build_tools/docker/build_and_update_gcr.py --image cmake --push
```

Kokoro builds use images tagged with `prod`, so to deploy an image:

```shell
python3 build_tools/docker/build_and_update_gcr.py --image cmake --push --tag=prod
```

We use
[multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/)
to limit duplication in our Dockerfiles and reduce the final image size. There
is still duplication in cases where it's difficult to determine the correct
files to copy.
