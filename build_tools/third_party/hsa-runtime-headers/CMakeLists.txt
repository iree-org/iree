# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(HSA_RUNTIME_HEADERS_ROOT "${IREE_ROOT_DIR}/third_party/hsa-runtime-headers/")

external_cc_library(
  PACKAGE
    hsa_runtime
  NAME
    headers
  ROOT
    ${HSA_RUNTIME_HEADERS_ROOT}
  SYSTEM_INCLUDES
    ${HSA_RUNTIME_HEADERS_ROOT}/include/
  PUBLIC
)

iree_install_targets(
  TARGETS
    hsa_runtime_headers
  COMPONENT
    IREEBundledLibraries
  EXPORT_SET
    Runtime
)
