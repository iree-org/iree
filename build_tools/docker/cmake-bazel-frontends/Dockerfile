# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/util@sha256:c9691ba3cd4287cee61be42103888d7c6f02c44c7c7b7d30a3dca7fc73ee92cb AS fetch-kws
WORKDIR /fetch-kws
ARG KWS_COMMIT=168f27a070dcd4b0ce39a70f9a702608ff10eb44
RUN wget "https://github.com/google-research/google-research/tarball/${KWS_COMMIT?}" \
    -O google-research.tar.gz \
  && tar --extract --gzip --file=google-research.tar.gz \
    --wildcards */kws_streaming --strip-components=1

FROM gcr.io/iree-oss/cmake@sha256:37ddc553262f28a4c290eb3ccda5e50ef1115209fbb62a44a85165e7b6f7d514 AS final

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
