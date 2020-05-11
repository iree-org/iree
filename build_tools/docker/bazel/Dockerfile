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

# Builds an image for running IREE on bazel.

# Set up the image and working directory.
FROM l.gcr.io/google/bazel:2.1.0
WORKDIR /usr/src/git

# Set environment variables.
ENV CXX clang++
ENV CC clang
ENV PYTHON_BIN /usr/bin/python3

# Install the newest version of git. This is necessary because the checkout
# action (https://github.com/actions/checkout) requires `git >= 2.18` for this
# action to work with git submodules.
RUN add-apt-repository ppa:git-core/ppa
RUN apt-get update
RUN apt-get install -y git

# Install core IREE dependencies.
RUN apt-get install -y clang libsdl2-dev

# Install python2 numpy. Temporary fix for issue #1737:
# https://github.com/google/iree/issues/1737
RUN apt-get install -y python-pip
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy
