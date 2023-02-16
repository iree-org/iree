# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image with latest stuff for building IREE using CMake and testing IREE with
# NVIDIA GPUs.

# To use the host GPUs, `docker run` must be called with the `--gpus all` flag.

# We use .deb files that we host because we have to pin the version and packages
# routinely dissapear from the Ubuntu apt repositories.
ARG NVIDIA_GL_DEB="libnvidia-gl-515-server_515.86.01-0ubuntu0.22.04.1_amd64.deb"
ARG NVIDIA_COMPUTE_DEB="libnvidia-compute-515-server_515.86.01-0ubuntu0.22.04.1_amd64.deb"
ARG NVIDIA_COMMON_DEB="libnvidia-common-515-server_515.86.01-0ubuntu0.22.04.1_all.deb"
ARG NVIDIA_EGL_WAYLAND_DEB="libnvidia-egl-wayland1_1.1.9-1.1_amd64.deb"


FROM gcr.io/iree-oss/base-bleeding-edge@sha256:273559bafe04270a8ad5b21d31c6c056d2d966b5d06c41353da5063cc8a16616 AS fetch-nvidia
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB
ARG NVIDIA_EGL_WAYLAND_DEB

WORKDIR /fetch-nvidia
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMMON_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_GL_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMPUTE_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_EGL_WAYLAND_DEB}"


# Set up the image and working directory by inheriting the vulkan CMake
# configuration.
# Note that we don't start from NVIDIA's docker base:
# - nvidia/cuda (https://hub.docker.com/r/nvidia/cuda):
#     it's.. for CUDA.
# - nvidia/vulkan (https://hub.docker.com/r/nvidia/vulkan):
#      does not support Ubuntu 22.04.
# This allows to share configuration with base CMake.
FROM gcr.io/iree-oss/base-bleeding-edge@sha256:273559bafe04270a8ad5b21d31c6c056d2d966b5d06c41353da5063cc8a16616 AS final
ARG NVIDIA_COMMON_DEB
ARG NVIDIA_GL_DEB
ARG NVIDIA_COMPUTE_DEB
ARG NVIDIA_EGL_WAYLAND_DEB

COPY --from=fetch-nvidia \
  "/fetch-nvidia/${NVIDIA_COMMON_DEB}" \
  "/fetch-nvidia/${NVIDIA_GL_DEB}" \
  "/fetch-nvidia/${NVIDIA_COMPUTE_DEB}" \
  "/fetch-nvidia/${NVIDIA_EGL_WAYLAND_DEB}" \
  /tmp/

RUN apt-get install "/tmp/${NVIDIA_COMMON_DEB}" \
  "/tmp/${NVIDIA_GL_DEB}" \
  "/tmp/${NVIDIA_COMPUTE_DEB}" \
  "/tmp/${NVIDIA_EGL_WAYLAND_DEB}"

# install cuda sdk (11.7 is the lowest version supports Ubuntu 22.04)
RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb \
  && dpkg --install cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb \
  && cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
  && apt-get update \
  && apt-get -y install cuda-toolkit-11-7
