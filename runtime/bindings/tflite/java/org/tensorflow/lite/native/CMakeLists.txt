# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

set(_NAME "iree-tflite-bindings")
add_library(${_NAME} SHARED "")

target_sources(${_NAME}
  PRIVATE
    "interpreter_jni.cc"
    "tensor_jni.cc"
    "tensorflow_lite_jni.cc"
)

target_link_libraries(${_NAME}
  PUBLIC
    iree::base
    iree::runtime::bindings::tflite::shim
)

target_link_options(${_NAME}
  INTERFACE
    "-landroid"
    "-llog"
)
