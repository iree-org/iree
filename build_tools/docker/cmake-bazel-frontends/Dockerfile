# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM gcr.io/iree-oss/cmake-python@sha256:c1f05139fa478107bdd5889d6b3ca9f8a4bcf19e77e3e8058e2084367a1aa23d AS final

# Install bazel.
COPY --from=gcr.io/iree-oss/bazel@sha256:5e52c7d43b6fdff35d884b8b8b92b1b6e2151d675019edc92f09018e558e0f94 \
  /usr/bin/bazel* \
  /usr/bin/

# Install tensorflow.
RUN python3 -m pip install tf-nightly==2.6.0.dev20210412

# Install JAX.
RUN python3 -m pip install --upgrade jax jaxlib flax

# Install KWS Streaming Library.
RUN python3 -m pip install --upgrade tensorflow-model-optimization==0.5.1.dev0
ARG KWS_COMMIT=4bfbfe4ec19fe39cf9d1803e894d8bfb5fd56a45
RUN git clone https://github.com/google-research/google-research
RUN cd google-research && git checkout "${KWS_COMMIT?}" && cd ..
ENV PYTHONPATH="/google-research:$PYTHONPATH"
