# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_ENABLE_EMITC})
  iree_add_all_subdirs()

  iree_cc_library(
    NAME
      VMToEmitC
    HDRS
      "ConvertVMToEmitC.h"
      "DropExcludedExports.h"
      "EmitCTypeConverter.h"
    SRCS
      "ConvertVMToEmitC.cpp"
      "DropExcludedExports.cpp"
    DEPS
      MLIREmitC
      MLIRIR
      MLIRPass
      MLIRStandard
      MLIRTransforms
      iree::compiler::Dialect::IREE::Conversion::PreserveCompilerHints
      iree::compiler::Dialect::IREE::IR
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::IR
      iree::compiler::Dialect::VM::Utils::CallingConvention
    PUBLIC
  )
endif()
