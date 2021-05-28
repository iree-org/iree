# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image containing utilities useful for setting up docker images, but not
# needed in the final images. Intermediate stages can inherit from this image,
# but final stages should not.

# Ubuntu 18.04
FROM ubuntu@sha256:fd25e706f3dea2a5ff705dbc3353cf37f08307798f3e360a13e9385840f73fb3 AS final

RUN apt-get update \
  && apt-get install -y \
  git \
  unzip \
  wget \
  gnupg2
