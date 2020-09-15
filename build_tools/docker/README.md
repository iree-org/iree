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


## Multi-stage Builds

We use
[multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/)
to limit duplication in our Dockerfiles and reduce the final image size. There
is still duplication in cases where it's difficult to determine the correct
files to copy.


## Dependencies Between Images

IREE images follow a consistent structure. The image defined by
`build_tools/docker/foo-bar/Dockerfile` is uploaded to GCR as
`gcr.io/iree-oss/foo-bar`. It may be tagged as `latest` or `prod`, e.g.
`gcr.io/iree-oss/foo-bar:latest`. Dockerfile image definitions should list their
dependencies based on these image names.

We use a helper python script to manage the Docker image deployment. It lists
all images and their dependencies and manages their canonical registry
location. When creating a new image, add it to this mapping. To build an image
and all images it depends on:

```shell
python3 build_tools/docker/manage_images.py --build --image cmake
```

To build multiple images

```shell
python3 build_tools/docker/manage_images.py --build --image cmake --image bazel
```

There is also the special option `--image all` to build all registered images.

Pushing images to GCR requires the `Storage Admin` role in the `iree-oss` GCP
project. To push these images to GCR with the `latest` tag:

```shell
python3 build_tools/docker/manage_images.py --image cmake --push
```

Kokoro build scripts and RBE configuration refer to images by their repository
digest. You can update references to the digest:

```shell
python3 build_tools/docker/manage_images.py --images all --tag latest --update_references
```

This requires that the tagged image have a repository digest, which means it was
pushed to or pulled from GCR.


## Deploying New Images

1. Modify the Dockerfiles as desired.
2. Update `manage_images.py` to include the new image and its dependencies.
3. Build and push the new image to GCR and update references to it:

  ```shell
  python3 build_tools/docker/manage_images.py --image "${IMAGE?}" --build --push --update_references
  ```

4. Commit changes and send a PR for review.
5. Merge your PR after is approved and all builds pass.
6. Kokoro builds preload images tagged with `prod` on VM creation, so after
   changing the images used, you should also update the images tagged as `prod`
   in GCR. Update your local reference to the `prod` tag to point at the new
   image:

  ```shell
  python3 build_tools/docker/manage_images.py --image "${IMAGE?}" --tag prod --build --update_references
  ```

  The build steps here should all be cache hits and no references should
  actually be changed. If they are, that indicates the images you've just built
  are different from the ones that are being referenced. Stop and fix this
  before proceeding. This relies on you keeping your local copy of the Docker
  images. If you didn't, you'll have to manually pull the missing images by
  their digest.

7. Push the new images with the `prod` tag to GCR.

   ```shell
  python3 build_tools/docker/manage_images.py --image "${IMAGE?}" --tag prod --push
  ```
