# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


if(${IREE_ENABLE_EMITC})
  iree_add_all_subdirs()

  iree_cc_library(
    NAME
      TranslateToCpp
    HDRS
      CppEmitter.h
    SRCS
      TranslateToCpp.cpp
    DEPS
      MLIREmitC
      MLIRIR
      MLIRSCF
      MLIRStandard
  )

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
      ::TranslateToCpp
      LLVMSupport
      MLIRIR
      MLIRPass
      MLIRSupport
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::IR
      iree::compiler::Dialect::VM::Conversion::VMToEmitC
    PUBLIC
  )
endif()
