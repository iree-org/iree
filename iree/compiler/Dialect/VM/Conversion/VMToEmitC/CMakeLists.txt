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
      "EmitCTypeConverter.h"
      "DropExcludedExports.h"
    SRCS
      "ConvertVMToEmitC.cpp"
      "DropExcludedExports.cpp"
    DEPS
      MLIRIR
      MLIRPass
      MLIREmitC
      MLIRTransforms
      iree::compiler::Dialect::IREE::Conversion::PreserveCompilerHints
      iree::compiler::Dialect::IREE::IR
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::IR
    INCLUDES
      "${PROJECT_SOURCE_DIR}/third_party/mlir-emitc/include"
      "${PROJECT_BINARY_DIR}/third_party/mlir-emitc/include"
    PUBLIC
  )
endif()
