# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running IREE's samples. Includes support for:
#   * CMake
#   * Vulkan (using SwiftShader)
#   * Python (including `venv` and other common pip packages)

FROM gcr.io/iree-oss/cmake-python-swiftshader@sha256:75ad1156a486aa4bf3f93d45f9a7e0f052345fa103171dfc742f8c92d4a7ea0f AS final

# update setuptools per https://github.com/pypa/setuptools/issues/1694#issuecomment-466010982
RUN apt-get update && apt-get install -y python3-venv python-setuptools && \
    python3 -m pip install --upgrade setuptools
