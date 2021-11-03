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
    IREEVM
  HDRS
    "IREEVM.h"
  SRCS
    "IREEVM.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTranslation
    iree::compiler::Bindings::Native::Transforms
    iree::compiler::Bindings::TFLite::Transforms
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::Conversion::HALToVM
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::Stream::Transforms
    iree::compiler::Dialect::Util::Transforms
    iree::compiler::Dialect::VM::Conversion
    iree::compiler::Dialect::VM::Conversion::StandardToVM
    iree::compiler::Dialect::VM::Target::Bytecode
    iree::compiler::Dialect::VM::Transforms
    iree::compiler::InputConversion::MHLO
    iree::compiler::InputConversion::TOSA
    iree::compiler::Utils
    ${IREE_VM_CONDITIONAL_TARGETS}
  PUBLIC
)
