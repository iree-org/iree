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

# An image for building IREE in RBE's remmote execution environments.
# The parent image requires gcloud authorization to download

# Build using:
# gcloud auth configure-docker
# docker build --tag gcr.io/iree-oss/rbe-toolchain build_tools/docker/rbe_toolchain/

FROM gcr.io/cloud-marketplace/google/rbe-ubuntu16-04@sha256:ac36d37616b044ee77813fc7cd36607a6dc43c65357f3e2ca39f3ad723e426f6

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy

# Install dependencies for python3.6-dev
RUN apt-get install -y software-properties-common
# Remove the symlink to python3 from /opt/python3.6/bin/python3.6 and symlink it
# to python3.5 to run a single command before resymlinking it to
# /opt/python3.6/bin/python3.6.
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.5 /usr/bin/python3
RUN add-apt-repository ppa:deadsnakes/ppa
RUN rm /usr/bin/python3
RUN ln -s /opt/python3.6/bin/python3.6 /usr/bin/python3

# Install python3.6-dev
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev
