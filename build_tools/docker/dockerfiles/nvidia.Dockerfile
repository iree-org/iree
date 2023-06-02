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
ARG NVIDIA_GL_DEB="libnvidia-gl-530_530.41.03-0ubuntu0.20.04.2_amd64.deb"
ARG NVIDIA_COMPUTE_DEB="libnvidia-compute-530_530.41.03-0ubuntu0.20.04.2_amd64.deb"
ARG NVIDIA_COMMON_DEB="libnvidia-common-530_530.41.03-0ubuntu0.20.04.2_all.deb"


FROM gcr.io/iree-oss/base@sha256:7dbb7e97e0baa6d4512822b5cd4f601d840a6f950f67c2df497a24cae64a0595 AS fetch-nvidia
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
FROM gcr.io/iree-oss/base@sha256:7dbb7e97e0baa6d4512822b5cd4f601d840a6f950f67c2df497a24cae64a0595 AS final
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB

COPY --from=fetch-nvidia \
  "/fetch-nvidia/${NVIDIA_COMMON_DEB}" \
  "/fetch-nvidia/${NVIDIA_GL_DEB}" \
  "/fetch-nvidia/${NVIDIA_COMPUTE_DEB}" \
  /tmp/

RUN apt-get install "/tmp/${NVIDIA_COMMON_DEB}" \
  "/tmp/${NVIDIA_GL_DEB}" \
  "/tmp/${NVIDIA_COMPUTE_DEB}"

# install cuda sdk
RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-debian11-11-6-local_11.6.0-510.39.01-1_amd64.deb \
  && dpkg --install cuda-repo-debian11-11-6-local_11.6.0-510.39.01-1_amd64.deb \
  && apt-key add /var/cuda-repo-debian11-11-6-local/7fa2af80.pub \
  && apt-get update \
  && apt-get -y install cuda-toolkit-11-6

# Adding CUDA binaries to Path
ENV PATH=${PATH}:/usr/local/cuda/bin/
