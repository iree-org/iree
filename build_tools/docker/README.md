# IREE Docker Configuration

This directory contains the Dockerfiles that specify the container images used
for IREE. Images are uploaded to
[Google Container Registry (GCR)](https://cloud.google.com/container-registry).

## Running Images Locally

To build an image, use `docker build`, e.g.:

```shell
docker build --file build_tools/docker/dockerfiles/base.Dockerfile --tag base .
```

To explore an image interactively, use `docker run`, e.g.

```shell
docker run --interactive --tty --rm base
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
`build_tools/docker/dockerfiles/foo-bar.Dockerfile` is uploaded to GCR as
`gcr.io/iree-oss/foo-bar`. It may be tagged as `latest` or `prod`, e.g.
`gcr.io/iree-oss/foo-bar:latest`. Dockerfile image definitions should list their
dependencies based on these image names.

We use a helper python script to manage the Docker image deployment. It lists
all images and their dependencies and manages their canonical registry location.
This script pushes images to GCR which requires the `Storage Admin` role in the
`iree-oss` GCP project.

When creating a new image, add it to the mapping in this script.

Run the script to

* Build an image and all images that transitively depend on it
* Push all built images to GCR
* Update all references to the image digest

For example, after changing the `base` image:

```shell
python3 build_tools/docker/manage_images.py --image base
```

For multiple images:

```shell
python3 build_tools/docker/manage_images.py --image base --image base-bleeding-edge
```

There is also the special option `--image all` for all registered images:

```shell
python3 build_tools/docker/manage_images.py --image all
```

## Adding or Updating an Image

### Part 1. Local Changes

1. Update the `Dockerfile` for the image that you want to modify or add. If
   you're adding a new image, or updating the dependencies between images, be
   sure to update `IMAGES_TO_DEPENDENCIES` in `manage_images.py` as well. If
   you are adding new images, it is best add them via `git add` before
   proceeding.

2. Build the image, push the image to GCR and update all references to the image
   with the new GCR digest:

    ```shell
    python3 build_tools/docker/manage_images.py --image "${IMAGE?}"
    ```

3. Test that the changes behave as expected locally and iterate on the steps
   above.

### Part 2. Submitting to GitHub

4. Commit the changes and send a PR for review. The CI will use the updated
   digest references to test the new images.

5. Merge your PR after is approved and all CI tests pass.
