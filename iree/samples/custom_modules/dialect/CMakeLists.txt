# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

iree_add_all_subdirs()

file(GLOB _GLOB_X_TD LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.td)
iree_cc_library(
  NAME
    dialect
  HDRS
    "conversion_patterns.h"
    "custom_dialect.h"
    "custom_ops.h.inc"
  SRCS
    "conversion_patterns.cc"
    "custom_dialect.cc"
    "custom_ops.cc.inc"
  DEPS
    ::custom_imports
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTransforms
    iree::compiler::Dialect::HAL::Conversion
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::VM::Conversion
  PUBLIC
)

iree_tablegen_library(
  NAME
    custom_ops_gen
  TD_FILE
    "custom_ops.td"
  OUTS
    -gen-op-decls custom_ops.h.inc
    -gen-op-defs custom_ops.cc.inc
)

iree_cc_embed_data(
  NAME
    custom_imports
  SRCS
    "custom.imports.mlir"
  CC_FILE_OUTPUT
    "custom.imports.cc"
  H_FILE_OUTPUT
    "custom.imports.h"
  CPP_NAMESPACE
    "mlir::iree_compiler::IREE::Custom"
  FLATTEN
  PUBLIC
)

iree_cc_library(
  NAME
    init_dialect
  HDRS
    "init_dialect.h"
  DEPS
    ::dialect
    MLIRIR
  PUBLIC
)

iree_cc_binary(
  NAME
    custom-opt
  OUT
    custom-opt
  SRCS
    "custom_opt.cc"
  DEPS
    ::init_dialect
    LLVMSupport
    MLIRIR
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
    iree::compiler::Conversion::HLOToLinalg
    iree::compiler::Conversion::init_conversions
    iree::compiler::Dialect::Flow::Analysis
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::Conversion::FlowToHAL
    iree::compiler::Dialect::HAL::Conversion::HALToVM
    iree::compiler::Dialect::HAL::Conversion::Passes
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::IREE::Transforms
    iree::compiler::Dialect::Shape::Conversion
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Transforms
    iree::compiler::Dialect::VM::Analysis
    iree::compiler::Dialect::VM::Conversion::StandardToVM
    iree::compiler::Dialect::VM::IR
    iree::compiler::Dialect::VM::Transforms
    iree::compiler::Dialect::VMLA::Conversion::HLOToVMLA
    iree::compiler::Dialect::VMLA::Conversion::VMLAToVM
    iree::compiler::Dialect::VMLA::Transforms
    iree::compiler::Dialect::Vulkan::IR
    iree::compiler::Translation::IREEVM
    iree::tools::init_compiler_modules
    iree::tools::init_iree_passes_and_dialects
    iree::tools::init_mlir_passes_and_dialects
    iree::tools::init_targets
    iree::tools::init_xla_dialects
    tensorflow::mlir_hlo
)

iree_cc_binary(
  NAME
    custom-translate
  OUT
    custom-translate
  SRCS
    "custom_translate.cc"
  DEPS
    ::init_dialect
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSCFTransforms
    MLIRSupport
    MLIRTranslation
    iree::compiler::Conversion::init_conversions
    iree::compiler::Dialect::VM::Target::Bytecode
    iree::compiler::Dialect::VM::Target::init_targets
    iree::compiler::Translation::IREEVM
    iree::tools::init_compiler_modules
    iree::tools::init_iree_passes_and_dialects
    iree::tools::init_mlir_passes_and_dialects
    iree::tools::init_targets
    iree::tools::init_translations
    iree::tools::init_xla_dialects
)
