# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    cc_wrappers
  SRCS
    "context_wrapper.cc"
    "function_wrapper.cc"
    "instance_wrapper.cc"
    "module_wrapper.cc"
  HDRS
    "context_wrapper.h"
    "function_wrapper.h"
    "instance_wrapper.h"
    "module_wrapper.h"
  DEPS
    iree::base::api
    iree::base::internal::flags
    iree::base::logging
    iree::base::status
    iree::hal::api
    iree::hal::drivers
    iree::modules::hal
    iree::modules::strings::strings_module
    iree::modules::tensorlist::native_module
    iree::vm
    iree::vm::cc
    iree::vm::bytecode_module
)
