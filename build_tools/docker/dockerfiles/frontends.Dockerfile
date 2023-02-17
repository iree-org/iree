# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/android@sha256:4540757ccf67e4cc594a16bdace31e23d193b04b9b54ecfcea07c3f50c6aeb6d

WORKDIR /pip-install

COPY integrations/tensorflow/test/requirements.txt ./

# Versions for things required to build IREE should match the minimum versions
# in integrations/tensorflow/test/requirements.txt. There
# doesn't appear to be a pip-native way to get the minimum versions, but this
# hack works for simple files, at least.
RUN sed -i 's/>=/==/' requirements.txt \
  && python3 -m pip install --upgrade -r requirements.txt \
  && rm -rf /pip-install

WORKDIR /
