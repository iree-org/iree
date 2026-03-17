# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(RDMA_CORE_HEADERS_ROOT "${IREE_ROOT_DIR}/third_party/rdma-core-headers/")

external_cc_library(
  PACKAGE
    rdma_core
  NAME
    headers
  ROOT
    ${RDMA_CORE_HEADERS_ROOT}
  SYSTEM_INCLUDES
    ${RDMA_CORE_HEADERS_ROOT}/include/
  PUBLIC
)

iree_install_targets(
  TARGETS
    rdma_core_headers
  COMPONENT
    IREEBundledLibraries
  EXPORT_SET
    Runtime
)
