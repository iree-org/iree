# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/android@sha256:e67ce11d45ea0bf35ab5a42554aa9076a6be084c05a7024b17c4924e18ba5fb6

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
  tensorflow==2.7.1 \
  keras==2.7.0 \
  # JAX.
  jax \
  jaxlib \
  flax \
  # KWS Dependency
  tensorflow-model-optimization==0.5.1.dev0

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
