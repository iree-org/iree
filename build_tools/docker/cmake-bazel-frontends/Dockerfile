# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/util@sha256:40846b4aea5886af3250399d6adfdb3e1195a8b0177706bb0375e812d62dc49c AS fetch-kws
WORKDIR /fetch-kws
ARG KWS_COMMIT=4bfbfe4ec19fe39cf9d1803e894d8bfb5fd56a45
RUN wget "https://github.com/google-research/google-research/tarball/${KWS_COMMIT?}" \
    -O google-research.tar.gz \
  && tar --extract --gzip --file=google-research.tar.gz \
    --wildcards */kws_streaming --strip-components=1

FROM gcr.io/iree-oss/cmake-python@sha256:51817f1a98f9ed9237577133b4c674b163280fd747c1745d6d0d93f0f2b01fb3 AS final

# Install bazel.
COPY --from=gcr.io/iree-oss/bazel@sha256:5e52c7d43b6fdff35d884b8b8b92b1b6e2151d675019edc92f09018e558e0f94 \
  /usr/bin/bazel* \
  /usr/bin/

# Install tensorflow.
RUN python3 -m pip install tf-nightly==2.6.0.dev20210527

# Install JAX.
RUN python3 -m pip install --upgrade jax jaxlib flax

# Install KWS Streaming Library.
RUN python3 -m pip install --upgrade tensorflow-model-optimization==0.5.1.dev0
COPY --from=fetch-kws /fetch-kws/kws_streaming /local-python/kws_streaming
ENV PYTHONPATH="/local-python:$PYTHONPATH"
