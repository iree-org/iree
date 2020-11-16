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
all images and their dependencies and manages their canonical registry location.
When creating a new image, add it to this mapping. To build an image and all
images it depends on:

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

## Adding or Updating an Image

If you have worked with the `docker` images before, it is prudent to follow the
steps in the "Debugging" section below before continuing.

### Part 1. Local Changes

1. Update the `Dockerfile` for the image that you want to modify or add.
2. Build the image, push the image to GCR and update all references to the image
   with the new GCR digest:

   ```shell
    python3 build_tools/docker/manage_images.py \
      --image "${IMAGE?}" --build \
      --tag latest \
      --push \
      --update_references
    ```

3. Test that the changes behave as expected locally and iterate on the steps
   above.

### Part 2. Submitting to GitHub

4. Commit the changes and send a PR for review. The CI will use the updated
   digest references to test the new images.

5. Merge your PR after is approved and all CI tests pass. **Please remember to
   complete the rest of the steps below**.

### Part 3. Updating the `:prod` tag

Kokoro builds preload images tagged with `prod` on VM creation, so after
changing the images used, you should also update the images tagged as `prod`
in GCR. This also makes development significantly easier for others who need to
modify the `docker` images.

6. On the `main` branch, build (but don't push) the images and locally tag them
   with the `:prod` tag:

   ```shell
    python3 build_tools/docker/manage_images.py \
      --image "${IMAGE?}" --build \
      --tag prod \
      --update_references
    ```

    This build should be entirely cache hits.
7. We include `--update_references` in the command above so that we can check
   that none of the images or references to them have been changed. Check that
   the following command produces no output before continuing:

   ```shell
   git status --porcelain
   ```

   If the output is not empty then you'll need to find the source of the
   discrepancy (e.g. a locally modified `Dockerfile`) and remove it, and repeat
   steps 5 and 6 before continuing. (This relies on you keeping your local copy
   of the Docker images. If you didn't, you'll have to manually pull the missing
   images by their digest).
8. Now that we've confirmed that none of the images were changed, we can push
   them to GCR with the `:prod` tag.

    ```shell
    python3 build_tools/docker/manage_images.py \
      --image "${IMAGE?}" \
      --tag prod \
      --push
    ```

## Debugging

Sometimes old versions of the `:latest` images can be stored locally and produce
unexpected behaviors. The following commands will download all of the prod
images and then update the images tagged with `:latest` on your machine (and on
GCR).

```shell
# Pull all :prod images
python3 build_tools/docker/manage_images.py --images all --pull --tag prod
# Update the :latest images to match the :prod images.
# If you have a clean workspace this _shouldn't_ require building anything as
# everything should be cache hits from the :prod images downloaded above, but if
# the :prod images are behind then that will not be the case and this may take
# several hours (depending on your machine).
python3 build_tools/docker/manage_images.py \
  --images all --build \
  --tag latest \
  --push \
  --update_references
```
