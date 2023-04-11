# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ANDROID)
  return()
endif()

set(NATIVE_APP_GLUE_DIR "${ANDROID_NDK}/sources/android/native_app_glue")

iree_cc_library(
  NAME
    android_native_app_glue
  SRCS
    "${NATIVE_APP_GLUE_DIR}/android_native_app_glue.c"
)

# Export ANativeActivity_onCreate().
# See https://github.com/android-ndk/ndk/issues/381.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

iree_cc_library(
  NAME
    iree_run_module_app
  SRCS
    src/main.cc
  INCLUDES
    ${NATIVE_APP_GLUE_DIR}
  DEPS
    ::android_native_app_glue
    iree::base
    iree::modules::hal
    iree::tooling::device_util
    iree::tooling::vm_util
    iree::vm
  LINKOPTS
    "-landroid"
    "-llog"
  SHARED
)
