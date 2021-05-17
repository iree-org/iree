# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


if(${IREE_ENABLE_EMITC})
  iree_add_all_subdirs()

  iree_cc_library(
    NAME
      C
    HDRS
      "CModuleTarget.h"
      "TranslationFlags.h"
    SRCS
      "CModuleTarget.cpp"
      "TranslationFlags.cpp"
      "TranslationRegistration.cpp"
    DEPS
      LLVMSupport
      MLIRTargetCpp
      MLIRIR
      MLIRPass
      MLIRSupport
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::IR
      iree::compiler::Dialect::VM::Conversion::VMToEmitC
      iree::compiler::Dialect::VM::Target::CallingConventionUtils
      iree::compiler::Dialect::VM::Target::ConstantEncodingUtils
    INCLUDES
      "${PROJECT_SOURCE_DIR}/third_party/mlir-emitc/include"
    PUBLIC
  )
endif()
