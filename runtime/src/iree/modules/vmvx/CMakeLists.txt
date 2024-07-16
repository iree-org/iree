# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_VMVX_OPTIONAL_COPTS)
set(_VMVX_OPTIONAL_DEPS)

iree_cc_library(
  NAME
    vmvx
  COPTS
    ${_VMVX_OPTIONAL_COPTS}
  HDRS
    "module.h"
  TEXTUAL_HDRS
    "exports.inl"
  SRCS
    "elementwise.c"
    "module.c"
  DEFINES
    "IREE_HAVE_VMVX_MODULE"
  DEPS
    iree::base
    iree::builtins::ukernel
    iree::base::internal::cpu
    iree::vm
    ${_VMVX_OPTIONAL_DEPS}
  PUBLIC
)
