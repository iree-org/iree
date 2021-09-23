# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/util@sha256:40846b4aea5886af3250399d6adfdb3e1195a8b0177706bb0375e812d62dc49c AS fetch-kws
WORKDIR /fetch-kws
ARG KWS_COMMIT=168f27a070dcd4b0ce39a70f9a702608ff10eb44
RUN wget "https://github.com/google-research/google-research/tarball/${KWS_COMMIT?}" \
    -O google-research.tar.gz \
  && tar --extract --gzip --file=google-research.tar.gz \
    --wildcards */kws_streaming --strip-components=1

FROM gcr.io/iree-oss/cmake-python@sha256:055eba8441ada5e0241f102241747e8d61196dca7f04316a0bfd68d2769883dc AS final

# Install bazel.
COPY --from=gcr.io/iree-oss/bazel@sha256:31c3acef0aeb2976b63b8cb946bda1de9bc8b5319b7e079e507afad2a62d13e7 \
  /usr/bin/bazel* \
  /usr/bin/

# Install TensorFlow and make sure the version of Keras we get matches.
RUN python3 -m pip install keras-nightly==2.7.0.dev2021080600 tf-nightly==2.7.0.dev20210806

# Install JAX.
RUN python3 -m pip install --upgrade jax jaxlib flax

# Install KWS Streaming Library.
RUN python3 -m pip install --upgrade tensorflow-model-optimization==0.5.1.dev0
COPY --from=fetch-kws /fetch-kws/kws_streaming /local-python/kws_streaming
ENV PYTHONPATH="/local-python:$PYTHONPATH"
