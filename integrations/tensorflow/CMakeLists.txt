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

# TensorFlow builds through bazel, and IREE maintains all of its TensorFlow
# dependent code under this directory tree. The CMake support is limited to
# compiler binaries and python bindings.
#
# Bazel is a beast that likes to be the center of the universe. There is some
# fragility in delegating to it in this fashion.
#
# If this directory is included, then building TensorFlow is assumed (the
# config option happens at the higher level).

iree_add_bazel_invocation(
  INVOCATION_TARGET
    integrations_iree_tensorflow_importers
  BAZEL_TARGETS
    //integrations/tensorflow/compiler:iree-tf-import
  EXECUTABLE_PATHS
    integrations/tensorflow/compiler/iree-tf-import
)

if(${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(bindings/python)
endif()
