# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(NUMA_ROOT "${IREE_ROOT_DIR}/third_party/numactl/")

message("Configure numactl")

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
    "shm.h"
    "sysfs.h"
    "rtnetlink.h"
    "affinity.h"
  SRCS
    "sysfs.c"
    "rtnetlink.c"
    "affinity.c"
    "syscall.c"
    "distance.c"
    "shm.c"
    "util.c"
    "libnuma.c"
  PUBLIC
)