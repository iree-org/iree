# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# To use the host GPUs, `docker run` must be called with the `--gpus all` flag.
# The NVidia drivers need to *exactly* match between the host machine and the
# docker image.

FROM gcr.io/iree-oss/frontends@sha256:5817b2780534b32347b8214fd8205f4fbf9d15d35555ce024be07da59f4f9df4

# We use .deb files that we host because we have to pin the version exactly to
# match the host machine and packages routinely dissapear from the Ubuntu
# apt repositories.
ARG NVIDIA_GL_DEB="libnvidia-gl-535_535.113.01-0ubuntu0.20.04.1_amd64.deb"
ARG NVIDIA_COMPUTE_DEB="libnvidia-compute-535_535.113.01-0ubuntu0.20.04.1_amd64.deb"
ARG NVIDIA_COMMON_DEB="libnvidia-common-535_535.113.01-0ubuntu0.20.04.1_all.deb"

WORKDIR /install-nvidia
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMMON_DEB}" \
  && wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_GL_DEB}" \
  && wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_COMPUTE_DEB}" \
  && apt-get update \
  && apt-get -y install "./${NVIDIA_COMMON_DEB}" \
  "./${NVIDIA_GL_DEB}" \
  "./${NVIDIA_COMPUTE_DEB}" \
  && rm -rf /install-nvidia

WORKDIR /
