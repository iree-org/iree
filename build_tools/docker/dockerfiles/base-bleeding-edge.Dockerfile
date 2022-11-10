# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A Docker image that contains all the latest stuff. We mostly test our oldest
# supported versions, but it's good to test the cutting edge also.

# 22.04
FROM ubuntu@sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
ARG LLVM_VERSION=14

######## Basic stuff ########
# Default compiler environment variables for IREE.
# Matches the version of clang installed below.
ENV CC /usr/bin/clang-${LLVM_VERSION}
ENV CXX /usr/bin/clang++-${LLVM_VERSION}

RUN apt-get update \
  && apt-get install -y \
    # For updating IREE's submodules.
    git \
    # Install our maximum supported clang version.
    "clang-${LLVM_VERSION}" \
    "lld-${LLVM_VERSION}" \
    # IREE transitive dependencies
    libsdl2-dev \
    libssl-dev \
    # A much better CMake builder
    ninja-build \
    # Modern Vulkan versions now available via apt
    libvulkan-dev \
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
    # Someone is welcome to tell me a better way to just install a specific
    # version as just lld (lld=<version> doesn't work)
    && ln -s "lld-${LLVM_VERSION}" /usr/bin/lld \
    && ln -s "ld.lld-${LLVM_VERSION}" /usr/bin/ld.lld

######## CMake ########
WORKDIR /install-cmake

# These are separate args because there's no way to strip the patch version off
# to get the /usr/share path.
# See https://github.com/moby/moby/issues/41383
ARG CMAKE_MAJOR_VERSION=3
ARG CMAKE_MINOR_VERSION=24
ARG CMAKE_PATCH_VERSION=3

ENV CMAKE_VERSION="${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}"

# Install our CMake version, which may be ahead of apt-get's version.
RUN wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION?}/cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" \
    && chmod +x "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" \
    && "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" --skip-license --prefix=/usr/  \
    && rm -rf /install-cmake

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

ARG PYTHON_VERSION=3.11

COPY runtime/bindings/python/iree/runtime/build_requirements.txt ./
RUN apt-get update \
  && apt-get install -y \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-dev" \
  && update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${PYTHON_VERSION}" 1 \
  && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-distutils \
    python3-venv \
    "python${PYTHON_VERSION}-venv" \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install --upgrade setuptools \
  # Versions for things required to build IREE. We install the latest version
  # of packages in runtime/bindings/python/iree/runtime/build_requirements.txt
  # Unlike most places in the project, these dependencies are *not* pinned in
  # general. This adds non-determinism to the Docker image build, but reduces
  # maintenance burden. If a new build fails because of a new package version,
  # we should add a max version constraint in the build_requirements.txt file.
  && python3 -m pip install --ignore-installed -r build_requirements.txt \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3

##############

######## IREE CUDA DEPS ########
COPY build_tools/docker/context/fetch_cuda_deps.sh /usr/local/bin
RUN /usr/local/bin/fetch_cuda_deps.sh /usr/local/iree_cuda_deps
ENV IREE_CUDA_DEPS_DIR="/usr/local/iree_cuda_deps"
##############

### Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /
