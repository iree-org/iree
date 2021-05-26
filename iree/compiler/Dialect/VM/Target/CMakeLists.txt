# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(IREE_ENABLE_EMITC)
  set(IREE_VM_CONDITIONAL_TARGETS
    iree::compiler::Dialect::VM::Target::C
  )
endif()

iree_cc_library(
  NAME
    CallingConventionUtils
  HDRS
    "CallingConventionUtils.h"
  SRCS
    "CallingConventionUtils.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRSupport
    iree::compiler::Dialect::VM::IR
  PUBLIC
)

iree_cc_library(
  NAME
    init_targets
  HDRS
    "init_targets.h"
  DEPS
    iree::compiler::Dialect::VM::Target::Bytecode
    ${IREE_VM_CONDITIONAL_TARGETS}
  PUBLIC
)
