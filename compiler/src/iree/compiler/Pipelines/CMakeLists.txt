# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Enable input dialects based on options.
set(IREE_INPUT_DEPS "")
if(IREE_INPUT_MHLO)
  list(APPEND IREE_INPUT_DEPS iree::compiler::InputConversion::MHLO)
  list(APPEND IREE_INPUT_DEPS iree::compiler::InputConversion::StableHLO)
endif()
if(IREE_INPUT_TORCH)
  list(APPEND IREE_INPUT_DEPS iree::compiler::InputConversion::TMTensor)
endif()
if(IREE_INPUT_TOSA)
  list(APPEND IREE_INPUT_DEPS iree::compiler::InputConversion::TOSA)
endif()

iree_cc_library(
  NAME
    Options
  HDRS
    "Options.h"
  SRCS
    "Options.cpp"
  DEPS
    ${IREE_INPUT_DEPS}
    iree::compiler::Utils
  PUBLIC
)

iree_cc_library(
  NAME
    Pipelines
  HDRS
    "Pipelines.h"
  SRCS
    "Pipelines.cpp"
  DEPS
    ${IREE_INPUT_DEPS}
    ::Options
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSupport
    iree::compiler::Bindings::Native::Transforms
    iree::compiler::Bindings::TFLite::Transforms
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::Conversion::HALToVM
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::Stream::Transforms
    iree::compiler::Dialect::Util::Transforms
    iree::compiler::Dialect::VM::Conversion
    iree::compiler::Dialect::VM::Conversion::StandardToVM
    iree::compiler::Dialect::VM::Target::Bytecode
    iree::compiler::Dialect::VM::Transforms
    iree::compiler::Modules::HAL::Inline::Transforms
    iree::compiler::Modules::HAL::Loader::Transforms
    iree::compiler::InputConversion::Common
    iree::compiler::InputConversion::Common::AutoInputConversionPipeline
    iree::compiler::Preprocessing::Passes
    iree::compiler::Utils
  PUBLIC
)
