# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")

iree_cc_library(
  NAME
    hook_cpu_events_linux
  SHARED
  SRCS
    hook_cpu_events_linux.cc
    perf_event_linux.cc
    stats.cc
  DEPS
    iree::base::core_headers
    iree::hal::local::executable_library
)

endif()  # CMAKE_SYSTEM_NAME STREQUAL "Linux"
