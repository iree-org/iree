# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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


if(${IREE_BUILD_TENSORFLOW_COMPILER} OR ${IREE_BUILD_TENSORFLOW_ALL})
  add_executable(iree_tf_compiler_iree-import-tf IMPORTED GLOBAL)
  set_property(TARGET iree_tf_compiler_iree-import-tf
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/iree-import-tf"
  )
endif()

if(${IREE_BUILD_TFLITE_COMPILER} OR ${IREE_BUILD_TENSORFLOW_ALL})
  add_executable(iree_tf_compiler_iree-import-tflite IMPORTED GLOBAL)
  set_property(TARGET iree_tf_compiler_iree-import-tflite
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/iree-import-tflite"
  )
endif()

if(${IREE_BUILD_XLA_COMPILER} OR ${IREE_BUILD_TENSORFLOW_ALL})
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
  if(${IREE_ENABLE_NEW_INTEGRATION_TESTS})
    add_subdirectory(test)
  else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/e2e/")
    include(iree_e2e_cartesian_product_test_suite)
    add_subdirectory(e2e)
  endif()
endif()
