# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_EXPERIMENTAL_CPU_UKERNEL_PLUGINS)
  return()
endif()

add_subdirectory(test)

include(iree_experimental_system_plugin.cmake)

iree_experimental_system_plugin(
  NAME
    builtin_ukernel_system_plugin
  SRCS
    plugin.c
  DEPS
    iree::builtins::ukernel
)
