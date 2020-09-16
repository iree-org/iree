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

# An image for building IREE using bazel.

ARG BAZEL_VERSION=3.3.1
# Change to a new version if upgrading Bazel.
ARG NEW_BAZEL_VERSION=3.3.1

FROM gcr.io/iree-oss/util AS install-bazel
WORKDIR /install-bazel
ARG BAZEL_VERSION
ARG NEW_BAZEL_VERSION

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Install Bazel.
# https://docs.bazel.build/versions/master/install-ubuntu.html
RUN wget -qO - https://bazel.build/bazel-release.pub.gpg | apt-key add - \
  && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | tee /etc/apt/sources.list.d/bazel.list \
  && apt-get update \
  # Install Bazel pinned at the version we want. Optionally install an
  # additional version of Bazel to ease upgrades (modify NEW_BAZEL_VERSION
  # above). Bazel does some shenanigans to select the correct version based on
  # your .bazelversion file. When upgrading, we therefore need to have both the
  # old and new version. When the versions are the same this second installation
  # is effectively a noop.
  && apt-get install -y "bazel=${BAZEL_VERSION?}" "bazel-${NEW_BAZEL_VERSION?}"

FROM gcr.io/iree-oss/base AS final
ARG BAZEL_VERSION
ARG NEW_BAZEL_VERSION
COPY --from=install-bazel \
  /usr/bin/bazel \
  "/usr/bin/bazel-${BAZEL_VERSION}" \
  "/usr/bin/bazel-${NEW_BAZEL_VERSION}" \
  /usr/bin/bazel-real \
  /usr/bin/

# TF requires python2 numpy at configure time...
# TODO(#1737): Remove this
RUN apt-get update \
    && apt-get install -y python-pip \
    && python -m pip install --upgrade pip \
    && python -m pip install numpy
