# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(NOT IREE_HAL_DRIVER_REMOTE)
  return()
endif()

# Transport options controlling which factories are compiled into the
# registration module. TCP is enabled by default; SHM is opt-in.
option(IREE_NET_TRANSPORT_TCP "Enable TCP transport for remote HAL" ON)
option(IREE_NET_TRANSPORT_SHM "Enable SHM transport for remote HAL" OFF)

set(_DEFINES "IREE_HAVE_HAL_REMOTE_CLIENT_DRIVER_MODULE=1")
set(_DEPS
  iree::base
  iree::hal
  iree::hal::remote::client
  iree::net::transport_factory
)

if(IREE_NET_TRANSPORT_TCP)
  list(APPEND _DEFINES "IREE_HAVE_NET_TCP_TRANSPORT=1")
  list(APPEND _DEPS iree::net::carrier::tcp::factory)
endif()

if(IREE_NET_TRANSPORT_SHM)
  list(APPEND _DEFINES "IREE_HAVE_NET_SHM_TRANSPORT=1")
  list(APPEND _DEPS iree::net::carrier::shm::factory)
endif()

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  DEPS
    ${_DEPS}
  DEFINES
    ${_DEFINES}
  PUBLIC
)
