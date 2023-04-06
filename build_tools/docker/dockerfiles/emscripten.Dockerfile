# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE through Emscripten.

FROM gcr.io/iree-oss/base@sha256:24fb5467da30c7b4c0f4c191cdf6124bda63b172d3ae98906e53b3d55ed6ddcb

# See also
#   * https://github.com/emscripten-core/emsdk/blob/main/docker/Dockerfile
#   * https://hub.docker.com/r/emscripten/emsdk

ARG EMSDK_COMMIT=21611d2a507fad73385120d89e05a794666070ae
ARG SDK_VERSION=3.1.20

WORKDIR /

# Follow https://emscripten.org/docs/getting_started/downloads.html.
RUN git clone https://github.com/emscripten-core/emsdk \
    && cd emsdk && git checkout "${EMSDK_COMMIT}" && \
    ./emsdk install ${SDK_VERSION} && \
    ./emsdk activate ${SDK_VERSION}

# Set some environment variables for Emscripten to use.
ENV EMSDK=/emsdk
ENV EM_DATA=${EMSDK}/.data
ENV EM_CONFIG=${EMSDK}/.emscripten
ENV EM_CACHE=${EM_DATA}/cache
ENV EM_PORTS=${EM_DATA}/ports
# Emscripten writes into its cache location (outside of the CMake build
# directory).
# We can either
#   (A) Grant broad write permissions to the cache directory to be able to run
#       our scripts under different users.
#   (B) Mount a user home directory when using the image.
# Since (A) requires less configuration, we'll do that. If multiple tools would
# want a user directory (like Bazel), we should switch to (B).
# See https://github.com/emscripten-core/emsdk/issues/535
RUN mkdir -p "${EM_CACHE}" && chmod -R 777 "${EM_CACHE}"

# Normally we'd run `source emsdk_env.sh`, but that doesn't integrate with
# Docker's environment properties model. Instead, we directly extend the path
# to include the directories suggested by `emsdk activate`.
ENV PATH="${EMSDK}:${EMSDK}/node/14.18.2_64bit/bin:${EMSDK}/upstream/emscripten:$PATH"
