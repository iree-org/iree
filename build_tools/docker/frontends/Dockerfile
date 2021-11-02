# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/android@sha256:58adb4131cfc7b08cd5767c577420f3479ca2f46bb67d9fac6c0797984627758

WORKDIR /install-kws

ARG KWS_COMMIT=168f27a070dcd4b0ce39a70f9a702608ff10eb44
ENV PYTHONPATH="/local-python:$PYTHONPATH"

RUN wget "https://github.com/google-research/google-research/tarball/${KWS_COMMIT?}" \
    -O google-research.tar.gz \
  && tar --extract --gzip --file=google-research.tar.gz \
    --wildcards */kws_streaming --strip-components=1 \
  && mkdir /local-python \
  && mv kws_streaming /local-python/kws_streaming \
  && rm -rf /install-kws

WORKDIR /

RUN python3 -m pip install --upgrade \
  # Matching versions of TF and Keras
  tf-nightly==2.7.0.dev20210806 \
  keras-nightly==2.7.0.dev2021080600 \
  # JAX.
  jax \
  jaxlib \
  flax \
  # KWS Dependency
  tensorflow-model-optimization==0.5.1.dev0
