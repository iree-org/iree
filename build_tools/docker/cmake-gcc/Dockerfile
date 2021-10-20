# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using GCC with CMake. We're overriding the env
# variables set in the base image. Is this terrible layering? Yes, but don't
# blame me. I wasn't the one that made Dockerfile a really unextensible format.

FROM gcr.io/iree-oss/cmake@sha256:37ddc553262f28a4c290eb3ccda5e50ef1115209fbb62a44a85165e7b6f7d514 AS final

ENV CC /usr/bin/gcc-9
ENV CXX /usr/bin/g++-9

# Avoid apt-add-repository, which requires software-properties-common, which is
# a rabbit hole of python version compatibility issues. See
# https://mondwan.blogspot.com/2018/05/alternative-for-add-apt-repository-for.html
# We use gcc-9 because it's what manylinux had (at time of authorship) and
# we don't aim to support older versions. We need a more modern lld to handle
# --push-state flags
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main" >> /etc/apt/sources.list  \
  && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F \
  && apt-get update \
  && apt-get install -y gcc-9 g++-9 lld-9 \
  && rm /usr/bin/lld /usr/bin/ld.lld \
  && ln -s lld-9 /usr/bin/lld \
  && ln -s ld.lld-9 /usr/bin/ld.lld
