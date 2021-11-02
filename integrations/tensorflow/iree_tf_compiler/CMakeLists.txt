# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_TF_TOOLS_ROOT
    "${CMAKE_SOURCE_DIR}/integrations/tensorflow/bazel-bin/iree_tf_compiler"
    CACHE STRING "Root directory for IREE TensorFlow integration binaries")

iree_package_name(_PACKAGE_NAME)
iree_package_ns(_PACKAGE_NS)

function(configure_tf_binary BINARY_NAME)
  set(_NAME "${_PACKAGE_NAME}_${BINARY_NAME}")
  add_executable("${_NAME}" IMPORTED GLOBAL)
  set_property(TARGET ${_NAME}
    PROPERTY IMPORTED_LOCATION
        "${IREE_TF_TOOLS_ROOT}/${BINARY_NAME}"
  )
  add_executable(${_PACKAGE_NS}::${BINARY_NAME} ALIAS ${_NAME})
  add_executable(${BINARY_NAME} ALIAS ${_NAME})
endfunction()

if(${IREE_BUILD_TENSORFLOW_COMPILER})
  configure_tf_binary("iree-import-tf")
endif()

if(${IREE_BUILD_TFLITE_COMPILER})
  configure_tf_binary("iree-import-tflite")
endif()

if(${IREE_BUILD_XLA_COMPILER})
  configure_tf_binary("iree-import-xla")
endif()

if(${IREE_BUILD_TESTS})
  configure_tf_binary("iree-tf-opt")
  configure_tf_binary("iree-opt-tflite")
endif()
