# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM ubuntu@sha256:fd25e706f3dea2a5ff705dbc3353cf37f08307798f3e360a13e9385840f73fb3 AS final

# Environment variables for IREE.
ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++

RUN apt-get update \
  && apt-get install -y \
    # For updating IREE's submodules.
    git \
    # Core IREE dependencies.
    clang \
    libsdl2-dev \
    libssl-dev
