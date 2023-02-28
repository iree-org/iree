# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running IREE's samples. Includes support for:
#   * CMake
#   * Vulkan (using SwiftShader)
#   * Python (including `venv` and common pip packages needed for Colab)

FROM gcr.io/iree-oss/swiftshader@sha256:05d59843bcd48352e4a14e96e9c6845b04d137e1132dc85f0a05fd7e53210263

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
    tensorflow==2.8.0 \
    tensorflow_hub==0.12.0

# NOTE: 2022-05-27: protobuf 4.21.0, released on May 25, 2022 is incompatible
# with prior releases. Specifically implicated are the above versions of
# tensorflow, which seem to include it without a version pin and therefore
# break out of the box. The next time the above versions are upgraded,
# try removing this line and then, within the docker image, run:
#   python3 -c "import tensorflow"
# If that fails with a stack trace, put this line back.
# On behalf of Google, we are sorry for the live at head philosophy
# and shoddy version management leaking into everything. We're victims too.
RUN python3 -m pip install protobuf==3.20.1 --force-reinstall
