# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM ubuntu@sha256:fd25e706f3dea2a5ff705dbc3353cf37f08307798f3e360a13e9385840f73fb3

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

######## Basic stuff ########
# Default compiler environment variables for IREE.
# Matches the version of clang installed below.
ENV CC /usr/bin/clang-9
ENV CXX /usr/bin/clang++-9

RUN apt-get update \
  && apt-get install -y \
    # For updating IREE's submodules.
    git \
    # Install our minimum supported clang version.
    clang-9 \
    lld-9 \
    # IREE transitive dependencies
    libsdl2-dev \
    libssl-dev \
    # A much better CMake builder
    ninja-build \
    # For building child images. Best practices would tell us to use multi-stage
    # builds (https://docs.docker.com/develop/develop-images/multistage-build/)
    # but it turns out that Dockerfile is a thoroughly non-composable awful
    # format and that doesn't actually work that well. These deps are pretty
    # small.
    unzip \
    wget \
    gnupg2 \
    # Needed for installing Bazel, per https://bazel.build/install/ubuntu
    apt-transport-https \
    curl \
    gnupg \
    # Needed for building lld with Bazel (as currently configured)
    libxml2-dev \
    # Optional for tools like llvm-symbolizer, which we could build from
    # source but would rather just have available ahead of time
    llvm-dev \
    # Someone is welcome to tell me a better way to just install lld-9 as lld
    # (lld=9 doesn't work)
    && ln -s lld-9 /usr/bin/lld \
    && ln -s ld.lld-9 /usr/bin/ld.lld

######## CMake ########
WORKDIR /install-cmake

# These are separate args because there's no way to strip the patch version off
# to get the /usr/share path.
# See https://github.com/moby/moby/issues/41383
ARG CMAKE_MAJOR_VERSION=3
ARG CMAKE_MINOR_VERSION=21
ARG CMAKE_PATCH_VERSION=6

ENV CMAKE_VERSION="${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}"

# Install our CMake version, which may be ahead of apt-get's version.
RUN wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION?}/cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" \
    && chmod +x "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" \
    && "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" --skip-license --prefix=/usr/  \
    && rm -rf /install-cmake

##############

######## Bazel ########
WORKDIR /install-bazel
# Making a required Bazel version change? Most images derive from this one
# and will get it automatically. However these don't. Please update them as
# well:
#   manylinux2014_x86_64-release
ARG BAZEL_VERSION=5.1.0

# https://bazel.build/install/ubuntu
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg \
  | gpg --dearmor >bazel-archive-keyring.gpg \
  && mv bazel-archive-keyring.gpg /usr/share/keyrings \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
  | tee /etc/apt/sources.list.d/bazel.list \
  && apt-get update \
  && apt-get install -y "bazel=${BAZEL_VERSION?}" \
  && rm -rf /install-bazel

##############

######## Python ########
# Note that we use --ignore-installed when installing packages that may have
# been auto-installed by the OS package manager (i.e. PyYAML is often an
# implicit OS-level dep). This should not break so long as we do not
# subsequently reinstall it on the OS side. Failing to do this will yield a
# hard error with pip along the lines of:
#   Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we
#   cannot accurately determine which files belong to it which would lead to
#   only a partial uninstall.
WORKDIR /install-python

RUN apt-get update \
  && apt-get install -y \
    python3.7 \
    python3.7-dev \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 \
  && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-distutils \
    python3-venv \
    python3.7-venv \
  && python3 -m pip install --upgrade pip>=21.3 \
  && python3 -m pip install --upgrade setuptools \
  && python3 -m pip install --ignore-installed \
    numpy==1.19.4 absl-py==0.12.0 requests PyYAML==5.4.1 wheel==0.36.2 \
    pybind11==2.9.0

ENV PYTHON_BIN /usr/bin/python3

##############

######## IREE CUDA DEPS ########
COPY fetch_cuda_deps.sh /usr/local/bin
RUN /usr/local/bin/fetch_cuda_deps.sh /usr/local/iree_cuda_deps
ENV IREE_CUDA_DEPS_DIR="/usr/local/iree_cuda_deps"
##############

######## Vulkan ########
WORKDIR /install-vulkan
ARG VULKAN_SDK_VERSION=1.2.154.0

RUN wget -q \
  # This file disappeared from the canonical source:
  # "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION?}/linux/vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz"
  "https://storage.googleapis.com/iree-shared-files/vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz" \
  && mkdir -p /opt/vulkan-sdk \
  && tar -xzf "vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz" -C /opt/vulkan-sdk \
  && rm -rf /install-vulkan
WORKDIR /

ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"

ENV PATH="${VULKAN_SDK}/bin:$PATH"

# Symlink the Vulkan loader to a system library directory. This is needed to
# allow Vulkan applications to find the Vulkan loader. It also avoids using
# LD_LIBRARY_PATH, which is not supported well by Docker.
RUN ln -s "${VULKAN_SDK}/lib/libvulkan.so" /usr/lib/x86_64-linux-gnu/ \
  && ln -s "${VULKAN_SDK}/lib/libvulkan.so.1" /usr/lib/x86_64-linux-gnu/

##############


######## GCC ########
WORKDIR /

# Avoid apt-add-repository, which requires software-properties-common, which is
# a rabbit hole of python version compatibility issues. See
# https://mondwan.blogspot.com/2018/05/alternative-for-add-apt-repository-for.html
# We use gcc-9 because it's what manylinux had (at time of authorship) and
# we don't aim to support older versions. We need a more modern lld to handle
# --push-state flags
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main" >> /etc/apt/sources.list  \
  && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F \
  && apt-get update \
  && apt-get install -y gcc-9 g++-9

##############

WORKDIR /
