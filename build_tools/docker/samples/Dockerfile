# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running IREE's samples. Includes support for:
#   * CMake
#   * Vulkan (using SwiftShader)
#   * Python (including `venv` and common pip packages needed for Colab)

FROM gcr.io/iree-oss/swiftshader@sha256:d27bd8ffdffa7abcd02970408b2665b2a7b1e8238e5b3561672633bd3e52652e

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
    tensorflow==2.5.0 \
    tensorflow_hub==0.12.0
