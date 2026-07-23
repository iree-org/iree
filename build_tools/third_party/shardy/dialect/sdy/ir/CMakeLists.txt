# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Shardy SDY dialect IR build
# TableGen and library definitions

set(SDY_IR_SOURCE_DIR "${SHARDY_SOURCE_DIR}/shardy/dialect/sdy/ir")

# TableGen include path
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include_directories(${SDY_IR_SOURCE_DIR})

set(LLVM_TARGET_DEFINITIONS ${SDY_IR_SOURCE_DIR}/dialect.td)
mlir_tablegen(dialect.h.inc -gen-dialect-decls)
mlir_tablegen(dialect.cc.inc -gen-dialect-defs)
add_public_tablegen_target(SdyDialectIncGen)
add_dependencies(mlir-headers SdyDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ${SDY_IR_SOURCE_DIR}/ops.td)
mlir_tablegen(ops.h.inc -gen-op-decls)
mlir_tablegen(ops.cc.inc -gen-op-defs)
add_public_tablegen_target(SdyOpsIncGen)
add_dependencies(mlir-headers SdyOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ${SDY_IR_SOURCE_DIR}/attrs.td)
mlir_tablegen(attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(attrs.cc.inc -gen-attrdef-defs)
add_public_tablegen_target(SdyAttrsIncGen)
add_dependencies(mlir-headers SdyAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS ${SDY_IR_SOURCE_DIR}/enums.td)
mlir_tablegen(enums.h.inc -gen-enum-decls)
mlir_tablegen(enums.cc.inc -gen-enum-defs)
add_public_tablegen_target(SdyEnumsIncGen)
add_dependencies(mlir-headers SdyEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS ${SDY_IR_SOURCE_DIR}/op_interface.td)
mlir_tablegen(op_interface.h.inc -gen-op-interface-decls)
mlir_tablegen(op_interface.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(SdyOpInterfaceIncGen)
add_dependencies(mlir-headers SdyOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS ${SDY_IR_SOURCE_DIR}/canonicalization.td)
mlir_tablegen(canonicalization.cc.inc -gen-rewriters)
add_public_tablegen_target(SdyCanonicalizationIncGen)
add_dependencies(mlir-headers SdyCanonicalizationIncGen)

set(LLVM_TARGET_DEFINITIONS ${SDY_IR_SOURCE_DIR}/bytecode.td)
mlir_tablegen(bytecode.cc.inc -gen-bytecode -bytecode-dialect=sdy)
add_public_tablegen_target(SdyBytecodeIncGen)
add_dependencies(mlir-headers SdyBytecodeIncGen)

add_mlir_dialect_library(SdyDialect
  PARTIAL_SOURCES_INTENDED
  ${SDY_IR_SOURCE_DIR}/bytecode.cc
  ${SDY_IR_SOURCE_DIR}/canonicalization.cc
  ${SDY_IR_SOURCE_DIR}/compatibility.cc
  ${SDY_IR_SOURCE_DIR}/dialect.cc
  ${SDY_IR_SOURCE_DIR}/extensions/stablehlo_extensions.cc
  ${SDY_IR_SOURCE_DIR}/parsers.cc
  ${SDY_IR_SOURCE_DIR}/printers.cc
  ${SDY_IR_SOURCE_DIR}/utils.cc
  ${SDY_IR_SOURCE_DIR}/verifiers.cc

  DEPENDS
  SdyDialectIncGen
  SdyOpsIncGen
  SdyAttrsIncGen
  SdyEnumsIncGen
  SdyOpInterfaceIncGen
  SdyCanonicalizationIncGen
  SdyBytecodeIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  ShardyCommon
  MLIRBytecodeOpInterface
  MLIRFuncDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRShapeDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  StablehloAssemblyFormat
  StablehloOps
  StablehloTypeInference
)

target_include_directories(SdyDialect PUBLIC
  $<BUILD_INTERFACE:${SHARDY_SOURCE_DIR}>
  $<BUILD_INTERFACE:${SHARDY_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# Register library
add_library(SdyDialectRegister
  ${SDY_IR_SOURCE_DIR}/register.cc
)

target_include_directories(SdyDialectRegister PUBLIC
  $<BUILD_INTERFACE:${SHARDY_SOURCE_DIR}>
  $<BUILD_INTERFACE:${SHARDY_BINARY_DIR}>
)

target_link_libraries(SdyDialectRegister PUBLIC
  SdyDialect
  MLIRFuncDialect
  MLIRFuncAllExtensions
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRTensorDialect
  StablehloOps
)
