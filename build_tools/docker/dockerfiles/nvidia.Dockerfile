# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using CMake and testing IREE with NVIDIA GPUs.

# To use the host GPUs, `docker run` must be called with the `--gpus all` flag.

# We use .deb files that we host because we have to pin the version and packages
# routinely dissapear from the Ubuntu apt repositories. The versions need to be
# compatible with the host driver (usually <= host driver version).
ARG NVIDIA_GL_DEB="libnvidia-gl-535_535.113.01-0ubuntu0.20.04.1_amd64.deb"
ARG NVIDIA_COMPUTE_DEB="libnvidia-compute-535_535.113.01-0ubuntu0.20.04.1_amd64.deb"
ARG NVIDIA_COMMON_DEB="libnvidia-common-535_535.113.01-0ubuntu0.20.04.1_all.deb"


FROM gcr.io/iree-oss/base@sha256:dc314b4fe30fc1315742512891357bffed4d1b62ffcb46258b1e0761c737b446 AS fetch-nvidia
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB

WORKDIR /fetch-nvidia
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMMON_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_GL_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMPUTE_DEB}"


# Set up the image and working directory by inheriting the base CMake
# configuration.
# Note that we don't start from NVIDIA's docker base:
# - nvidia/cuda (https://hub.docker.com/r/nvidia/cuda), or
# - nvidia/vulkan (https://hub.docker.com/r/nvidia/vulkan).
# This allows to share configuration with base CMake and better control the
# installed packages. But it does mean we need to carefully manage the MATCHING
# of the driver version between the host machine and the docker image.
FROM gcr.io/iree-oss/base@sha256:dc314b4fe30fc1315742512891357bffed4d1b62ffcb46258b1e0761c737b446 AS final
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB

COPY --from=fetch-nvidia \
  "/fetch-nvidia/${NVIDIA_COMMON_DEB}" \
  "/fetch-nvidia/${NVIDIA_GL_DEB}" \
  "/fetch-nvidia/${NVIDIA_COMPUTE_DEB}" \
  /tmp/

# The local .deb files have dependencies that requires apt-get update to locate.
RUN apt-get update \
  && apt-get -y install "/tmp/${NVIDIA_COMMON_DEB}" \
  "/tmp/${NVIDIA_GL_DEB}" \
  "/tmp/${NVIDIA_COMPUTE_DEB}"

# Install the CUDA SDK
RUN wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.1-535.86.10-1_amd64.deb \
  && dpkg --install cuda-repo-ubuntu2004-12-2-local_12.2.1-535.86.10-1_amd64.deb \
  && cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
  && apt-get update \
  && apt-get -y install cuda-toolkit-12-2

# Adding CUDA binaries to Path
ENV PATH=${PATH}:/usr/local/cuda/bin/
