# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE using bazel.

ARG BAZEL_VERSION=3.7.1

FROM gcr.io/iree-oss/util@sha256:c9691ba3cd4287cee61be42103888d7c6f02c44c7c7b7d30a3dca7fc73ee92cb AS install-bazel
WORKDIR /install-bazel
ARG BAZEL_VERSION

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Install Bazel.
# https://docs.bazel.build/versions/master/install-ubuntu.html
RUN wget -qO - https://bazel.build/bazel-release.pub.gpg | apt-key add - \
  && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | tee /etc/apt/sources.list.d/bazel.list \
  && apt-get update \
  && apt-get install -y "bazel=${BAZEL_VERSION?}"

FROM gcr.io/iree-oss/base@sha256:e5ad55549adc27d5d1927b7857ff308d9b75fa1be8a110ed848b24e3816127c5 AS final
ARG BAZEL_VERSION
COPY --from=install-bazel \
  /usr/bin/bazel* \
  /usr/bin/

# TF requires python and numpy at configure time...
# TODO(#1737): Remove this
RUN apt-get update \
    && apt-get install -y \
      python3 \
      python3-pip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy
