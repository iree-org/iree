# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is derived from a stock manylinux2014 image, based on CentOS 7.
# It does not derive from any of our other images and contains sufficient
# software to build release packages for that OS. Note that the upstream
# images are patched regularly with backports from RedHat and have relatively
# recent dev tooling and Python versions. Bump the base hash to get Python
# and dev tooling upgrades.
#
# This line of images is EOL on June 30, 2024. Prior to that, we should upgrade
# to a newer revision. Newer manylinux images are based on Debian.
#
# Refer to: https://github.com/pypa/manylinux
FROM quay.io/pypa/manylinux2014_x86_64@sha256:9b463efac479efbcab6dec77eca28c5cfa0c5ef64f13ac184eb7117dc1f8edda

USER root

######## Pre-requisite packages ########
# Add RHEL7 CUDA repo.
RUN yum-config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
RUN yum install -y \
  cuda-nvcc-11-6 cuda-cudart-devel-11-6 cuda-cupti-11-6 \
  java-11-openjdk-devel \
  ccache \
  capstone-devel libzstd-devel

######## Bazel ########
# Bazel requires Java.
ARG BAZEL_VERSION=5.1.0
RUN curl -fsSL \
  https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64 \
  -o /usr/local/bin/bazel \
  && chmod a+x /usr/local/bin/bazel \
  && /usr/local/bin/bazel --version

# See: https://github.com/bazelbuild/bazel/issues/10327
# Note also that many things that link fine on newer OS's seem to fail based
# on missing -lm, so just adding here.
ENV BAZEL_LINKOPTS ""
ENV BAZEL_LINKLIBS "-lstdc++ -lm"

######## TBB ########
# TBB is a dependency of Tracy and there is not a packaged source for a versoin
# that is compatible with the STL shipped on this OS. So we use a script to
# fetch/build/install exactly what is needed.
COPY install_tbb_manylinux2014.sh /usr/local/bin
RUN bash /usr/local/bin/install_tbb_manylinux2014.sh
