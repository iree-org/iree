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

set(_bazel_targets)
set(_executable_paths)

set(IREE_TF_TOOLS_ROOT
    "${CMAKE_SOURCE_DIR}/integrations/tensorflow/bazel-bin/iree_tf_compiler"
    CACHE STRING "Root directory for IREE TensorFlow integration binaries")


if(${IREE_BUILD_TENSORFLOW_COMPILER})
  add_executable(iree_tf_compiler_iree-tf-import IMPORTED GLOBAL)
  set_property(TARGET iree_tf_compiler_iree-tf-import
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/iree-tf-import"
  )
endif()

if(${IREE_BUILD_TFLITE_COMPILER})
  add_executable(iree_tf_compiler_iree-import-tflite IMPORTED GLOBAL)
  set_property(TARGET iree_tf_compiler_iree-import-tflite
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/iree-import-tflite"
  )
endif()

if(${IREE_BUILD_XLA_COMPILER})
  add_executable(iree_tf_compiler_iree-import-xla IMPORTED GLOBAL)
  set_property(TARGET iree_tf_compiler_iree-import-xla
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/iree-import-xla"
  )
endif()

if(${IREE_BUILD_TESTS})
  add_executable(iree_tf_compiler_iree-tf-opt IMPORTED GLOBAL)
  set_property(TARGET iree_tf_compiler_iree-tf-opt
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/iree-tf-opt"
  )

  add_executable(iree_tf_compiler_iree-opt-tflite IMPORTED GLOBAL)
  set_property(TARGET iree_tf_compiler_iree-opt-tflite
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/iree-opt-tflite"
  )
endif()

if(${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(bindings/python)
endif()

if(${IREE_BUILD_TESTS} AND ${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(e2e)
endif()
