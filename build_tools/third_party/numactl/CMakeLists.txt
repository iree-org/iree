# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(NUMA_ROOT "${IREE_ROOT_DIR}/third_party/numactl/")

external_cc_library(
  PACKAGE
    numa
  NAME
    libnuma
  ROOT
    ${NUMA_ROOT}
  HDRS
    "config.h"
    "numa.h"
    "numaif.h"
    "numaint.h"
    "util.h"
    "sysfs.h"
    "rtnetlink.h"
    "affinity.h"
  SRCS
    "sysfs.c"
    "rtnetlink.c"
    "affinity.c"
    "syscall.c"
    "libnuma.c"
    "numactl.c"
  PUBLIC
)

# Define our own binary target for the CLI, for host builds only.
# depending on the build configuration, so we prefer to just set our own name.
if(NOT CMAKE_CROSSCOMPILING)
  add_executable(iree-numa
    "${NUMA_ROOT}/libnuma.c"
    "${NUMA_ROOT}/numactl.c"
    "${NUMA_ROOT}/syscall.c"
    "${NUMA_ROOT}/affinity.c"
    "${NUMA_ROOT}/sysfs.c"
    "${NUMA_ROOT}/rtnetlink.c"
  )
  target_include_directories(iree-numa SYSTEM
    PUBLIC
      "${NUMA_ROOT}"
  )
endif()