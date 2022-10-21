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
    "affinity.h"
    "sysfs.h"
    "rtnetlink.h"
    "shm.h"
  SRCS
    "affinity.c"
    "util.c"
    "syscall.c"
    "sysfs.c"
    "rtnetlink.c"
    "shm.c"
    "libnuma.c"
    "numactl.c"
  PUBLIC
)