# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(NCCL_SOURCE_DIR
  "${IREE_SOURCE_DIR}/third_party/nccl/"
)

external_cc_library(
  PACKAGE
    nccl
  NAME
    headers
  ROOT
    "${NCCL_SOURCE_DIR}"
  INCLUDES
    "${NCCL_SOURCE_DIR}"
  PUBLIC
)
