# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running IREE's samples. Includes support for:
#   * CMake
#   * Vulkan (using SwiftShader)
#   * Python (including `venv` and common pip packages needed for Colab)

FROM gcr.io/iree-oss/swiftshader@sha256:c9be5cbc8467499ae71ec80f3af87b72e746e8903cd52c0be9bb5f7261acc521

# Install additional packages often used in notebooks.
# Installing these at the system level helps with caching, since venvs can
# set --system-site-packages to use the already installed versions.
#
# Note:
#   * Packages relating to TensorFlow are pinned to versions close to what
#     Colab includes in its hosted runtimes. We don't need to match all of
#     Colab's dependencies, but we should at least make an effort for the ones
#     our notebooks use.
#   * We explicitly do *not* install Jupyter notebook requirements since they
#     should be installed within venvs.
RUN python3 -m pip install --ignore-installed \
    numpy \
    matplotlib \
    bottleneck \
    tensorflow==2.12.0 \
    tensorflow_hub==0.13.0
