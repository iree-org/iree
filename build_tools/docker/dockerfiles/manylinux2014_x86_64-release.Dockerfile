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
FROM quay.io/pypa/manylinux2014_x86_64@sha256:1818cd784995512fd6865baf79bd34c8f426f356a98fdc53495cf0bcd9e6b790

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

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
WORKDIR /install-bazel
COPY build_tools/docker/context/install_bazel.sh .bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel

##############

# See: https://github.com/bazelbuild/bazel/issues/10327
# Note also that many things that link fine on newer OS's seem to fail based
# on missing -lm, so just adding here.
ENV BAZEL_LINKOPTS ""
ENV BAZEL_LINKLIBS "-lstdc++ -lm"

######## TBB ########
# TBB is a dependency of Tracy and there is not a packaged source for a versoin
# that is compatible with the STL shipped on this OS. So we use a script to
# fetch/build/install exactly what is needed.
COPY build_tools/docker/context/install_tbb_manylinux2014.sh /usr/local/bin
RUN bash /usr/local/bin/install_tbb_manylinux2014.sh

######## AMD ROCM #######
ARG ROCM_VERSION=5.2.1
ARG AMDGPU_VERSION=22.20.1

# Install the ROCm rpms
RUN yum clean all \
  && echo -e "[ROCm]\nname=ROCm\nbaseurl=https://repo.radeon.com/rocm/yum/${ROCM_VERSION}/main\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/rocm.repo \
  && echo -e "[amdgpu]\nname=amdgpu\nbaseurl=https://repo.radeon.com/amdgpu/${AMDGPU_VERSION}/rhel/7.9/main/x86_64\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/amdgpu.repo \
  && yum install -y rocm-dev

######## GIT CONFIGURATION ########
# Git started enforcing strict user checking, which thwarts version
# configuration scripts in a docker image where the tree was checked
# out by the host and mapped in. Disable the check.
# See: https://github.com/openxla/iree/issues/12046
# We use the wildcard option to disable the checks. This was added
# in git 2.35.3
RUN git config --global --add safe.directory '*'
