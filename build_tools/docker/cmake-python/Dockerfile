# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE and its Python bindings using CMake.

FROM gcr.io/iree-oss/cmake@sha256:57ee32f8b40b727b00054be9f3359f4980aca1556740aa70c3fcfb43eedc1a4f AS final
# Dependencies for the python bindings tests.
RUN apt-get update \
  && apt-get install -y \
    python3.7 \
    python3.7-dev \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 \
  && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-distutils \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install \
    numpy==1.19.4 absl-py==0.12.0 requests PyYAML==5.4.1 wheel==0.36.2 \
    pybind11==2.6.1

ENV PYTHON_BIN /usr/bin/python3
