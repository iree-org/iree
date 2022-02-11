# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of various special logic throughout.

# TODO(#6353)Tools has thread dependency in gtest and benchmark; they should
# be separated into the runtime tools and compiler tools.
if(NOT ${IREE_ENABLE_THREADING})
  return()
endif()

add_subdirectory(android)
add_subdirectory(test)
add_subdirectory(utils)

# Enable compiler targets based on options.
set(IREE_COMPILER_TARGETS "")
set(IREE_COMPILER_TARGET_COPTS "")
if(IREE_TARGET_BACKEND_DYLIB_LLVM_AOT OR IREE_TARGET_BACKEND_WASM_LLVM_AOT)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::LLVM)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_LLVMAOT_TARGET")
endif()
if(IREE_TARGET_BACKEND_METAL_SPIRV)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::MetalSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_METALSPIRV_TARGET")
endif()
if(IREE_TARGET_BACKEND_VMVX)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VMVX)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VMVX_TARGET")
endif()
if(IREE_TARGET_BACKEND_VULKAN_SPIRV)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VulkanSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VULKANSPIRV_TARGET")
endif()
if(IREE_TARGET_BACKEND_WEBGPU)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::WebGPU)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_WEBGPU_TARGET")
endif()
if(IREE_TARGET_BACKEND_CUDA)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::CUDA)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_CUDA_TARGET")
endif()
if(IREE_TARGET_BACKEND_ROCM)
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::ROCM)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_ROCM_TARGET")
endif()

if(IREE_ENABLE_EMITC)
  set(IREE_EMITC_CONDITIONAL_DEP
    MLIREmitC
  )
endif()

iree_cc_binary(
  NAME
    iree-benchmark-module
  SRCS
    "iree-benchmark-module-main.cc"
  DEPS
    benchmark
    iree::base
    iree::base::cc
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::hal::drivers
    iree::modules::hal
    iree::tools::utils::vm_util
    iree::vm
    iree::vm::bytecode_module
    iree::vm::cc
)

iree_cc_binary(
  NAME
    iree-benchmark-trace
  SRCS
    "iree-benchmark-trace-main.c"
  DEPS
    iree::base
    iree::base::internal::atomic_slist
    iree::base::internal::file_path
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::hal::drivers
    iree::modules::hal
    iree::testing::benchmark
    iree::tools::utils::trace_replay
    iree::tools::utils::yaml_util
    iree::vm
    yaml
)

iree_cc_binary(
  NAME
    iree-check-module
  SRCS
    "iree-check-module-main.cc"
  DEPS
    iree::base
    iree::base::cc
    iree::base::core_headers
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::logging
    iree::base::tracing
    iree::hal
    iree::hal::drivers
    iree::modules::check
    iree::modules::hal
    iree::testing::gtest
    iree::tools::utils::vm_util
    iree::vm
    iree::vm::bytecode_module
  TESTONLY
)

iree_cc_binary(
  NAME
    iree-dump-module
  SRCS
    "iree-dump-module-main.cc"
  DEPS
    flatcc::runtime
    iree::base
    iree::base::internal::file_io
    iree::base::internal::flatcc::debugging
    iree::schemas::bytecode_module_def_c_fbs
    iree::tools::utils::vm_util
)

iree_cc_binary(
  NAME
    iree-run-module
  SRCS
    "iree-run-module-main.cc"
  DEPS
    iree::base
    iree::base::cc
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::tracing
    iree::hal::drivers
    iree::modules::hal
    iree::tools::utils::vm_util
    iree::vm
    iree::vm::bytecode_module
    iree::vm::cc
)

iree_cc_binary(
  NAME
    iree-run-trace
  SRCS
    "iree-run-trace-main.c"
  DEPS
    iree::base
    iree::base::internal::file_path
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::hal::drivers
    iree::modules::hal
    iree::tools::utils::trace_replay
    iree::tools::utils::yaml_util
    iree::vm
    yaml
)

iree_cc_binary(
  NAME
    iree-e2e-matmul-test
  SRCS
    "iree-e2e-matmul-test.c"
  DEPS
    iree::base
    iree::base::internal::file_path
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::hal::drivers
    iree::modules::hal
    iree::tools::utils::trace_replay
    iree::tools::utils::yaml_util
    iree::vm
    yaml
)

if(${IREE_BUILD_COMPILER})
  iree_cc_binary(
    NAME
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
      "${IREE_SOURCE_DIR}/iree/compiler/Dialect/Util/Tools/StructAttrGen.cpp"
      "${IREE_SOURCE_DIR}/iree/compiler/Dialect/VM/Tools/VMOpEncoderGen.cpp"
      "${IREE_SOURCE_DIR}/iree/compiler/Dialect/VM/Tools/VMOpTableGen.cpp"
    DEPS
      LLVMSupport
      LLVMTableGen
      MLIRSupport
      MLIRTableGen
      iree::compiler::Utils
    HOSTONLY
  )

  iree_cc_library(
    NAME
      init_iree_passes_and_dialects
    HDRS
      "init_iree_dialects.h"
      "init_iree_passes.h"
    DEPS
      IREEInputDialect
      IREEPyDMDialect
      MLIRIR
      iree::compiler::Bindings::Native::Transforms
      iree::compiler::Bindings::TFLite::Transforms
      iree::compiler::Codegen::Dialect::IREECodegenDialect
      iree::compiler::Codegen::Interfaces::Interfaces
      iree::compiler::Dialect::Flow::IR
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::IR::HALDialect
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::Modules::VMVX::IR::VMVXDialect
      iree::compiler::Dialect::Modules::VMVX::Transforms
      iree::compiler::Dialect::Stream::IR
      iree::compiler::Dialect::Stream::Transforms
      iree::compiler::Dialect::Util::IR
      iree::compiler::Dialect::Util::Transforms
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::IR
      iree::compiler::Dialect::VM::Transforms
      iree::compiler::Dialect::Vulkan::IR
      iree::compiler::InputConversion::Common
      iree::compiler::InputConversion::MHLO
      iree::compiler::InputConversion::TOSA
      iree::compiler::ConstEval
      iree::compiler::Translation::HALExecutable
      iree::compiler::Translation::IREEVM
    PUBLIC
  )

  iree_cc_library(
    NAME
      init_mlir_passes_and_dialects
    HDRS
      "init_mlir_dialects.h"
      "init_mlir_passes.h"
    DEPS
      MLIRAffine
      MLIRAffineTransforms
      MLIRArmNeon
      MLIRArmNeon2dToIntr
      MLIRBufferization
      MLIRControlFlow
      MLIRGPUOps
      MLIRGPUToSPIRV
      MLIRIR
      MLIRLLVMIR
      MLIRLinalg
      MLIRLinalgToLLVM
      MLIRLinalgToSPIRV
      MLIRLinalgTransforms
      MLIRQuant
      MLIRQuantTransforms
      MLIRSCF
      MLIRSCFToGPU
      MLIRSCFTransforms
      MLIRSPIRV
      MLIRSPIRVTransforms
      MLIRShape
      MLIRStandard
      MLIRStandardToSPIRV
      MLIRTensorInferTypeOpInterfaceImpl
      MLIRTosa
      MLIRTosaTransforms
      MLIRTransforms
      MLIRVector
      ${IREE_EMITC_CONDITIONAL_DEP}
    PUBLIC
  )

  iree_cc_library(
    NAME
      init_passes_and_dialects
    HDRS
      "init_dialects.h"
      "init_passes.h"
    DEPS
      ::init_compiler_modules
      ::init_iree_passes_and_dialects
      ::init_mlir_passes_and_dialects
      ::init_xla_dialects
      iree::compiler::Codegen::Codegen
      iree::compiler::Dialect::HAL::Conversion::Passes
    PUBLIC
  )

  iree_cc_library(
    NAME
      init_compiler_modules
    HDRS
      "init_compiler_modules.h"
    DEPS
      iree::compiler::Dialect::Modules::Check::IR::CheckDialect
  )

  iree_cc_library(
    NAME
      init_targets
    HDRS
      "init_targets.h"
    SRCS
      "init_targets.cc"
    DEPS
      ${IREE_COMPILER_TARGETS}
    COPTS
      ${IREE_COMPILER_TARGET_COPTS}
    PUBLIC
  )

  iree_cc_library(
    NAME
      init_xla_dialects
    HDRS
      "init_xla_dialects.h"
    DEPS
      tensorflow::external_mhlo_includes
      ChloDialect
      MhloDialect
    PUBLIC
  )

  iree_cc_library(
    NAME
      init_translations
    HDRS
      "init_translations.h"
    DEPS
      iree::compiler::Translation::HALExecutable
      iree::compiler::Translation::IREEVM
      MLIRSPIRVTranslateRegistration
  )

  iree_cc_library(
    NAME
      iree_opt_main
    SRCS
      "iree-opt-main.cc"
    DEPS
      ::init_passes_and_dialects
      ::init_targets
      LLVMSupport
      MLIRIR
      MLIROptLib
      MLIRSupport
    PUBLIC
  )

  iree_cc_library(
    NAME
      iree_translate_lib
    HDRS
      "iree_translate_lib.h"
    SRCS
      "iree_translate_lib.cc"
    DEPS
      ::init_compiler_modules
      ::init_iree_passes_and_dialects
      ::init_mlir_passes_and_dialects
      ::init_targets
      ::init_translations
      ::init_xla_dialects
      LLVMSupport
      MLIRIR
      MLIRArmNeonToLLVMIRTranslation
      MLIRLLVMToLLVMIRTranslation
      MLIRSCFTransforms
      MLIRPass
      MLIRSupport
      MLIRTargetLLVMIRExport
      MLIRTranslation
      iree::compiler::Codegen::Codegen
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Target::init_targets
      iree::compiler::Translation::HALExecutable
      iree::compiler::Translation::IREEVM
    PUBLIC
  )

  iree_cc_binary(
    NAME
      iree-translate
    SRCS
      "iree-translate-main.cc"
    DEPS
      ::iree_translate_lib
    DATA
      lld
    HOSTONLY
  )

  iree_cc_binary(
    NAME
      iree-opt
    DEPS
      ::iree_opt_main
    DATA
      lld
    HOSTONLY
  )

  iree_cc_binary(
    NAME
      iree-mlir-lsp-server
    SRCS
      "iree-mlir-lsp-server.cc"
    DEPS
      ::init_passes_and_dialects
      MLIRIR
      MLIRLspServerLib
      MLIRSupport
    PUBLIC
  )

  iree_cc_binary(
    NAME
      iree-run-mlir
    SRCS
      "iree-run-mlir-main.cc"
    DEPS
      ::init_passes_and_dialects
      ::init_targets
      LLVMSupport
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSupport
      MLIRTargetLLVMIRExport
      iree::base
      iree::base::cc
      iree::base::internal::flags
      iree::base::logging
      iree::base::tracing
      iree::compiler::Dialect::HAL::Target
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Target::init_targets
      iree::compiler::Translation::IREEVM
      iree::hal
      iree::hal::drivers
      iree::modules::hal
      iree::tools::utils::vm_util
      iree::vm
      iree::vm::bytecode_module
      iree::vm::cc
    DATA
      lld
    HOSTONLY
  )

  # Ensure FileCheck gets built. Tests don't have dependencies in CMake because
  # they aren't targets. So until we fix that, we just force this to get built.
  # Limiting this to when IREE_BUILD_TESTS is set prevents the installation
  # below, which we use for cross-platform testing.
  set_target_properties(FileCheck PROPERTIES EXCLUDE_FROM_ALL OFF)

  # lld install - required by the compiler to link codegen executables.
  install(
    TARGETS lld
    COMPONENT Compiler
    RUNTIME DESTINATION bin
  )

  # Bundle the FileCheck binary from LLVM into our tests/bin directory so
  # installed FileCheck tests are hermetic.
  install(
    TARGETS FileCheck
    DESTINATION "tests/bin"
    COMPONENT Tests
  )
endif(${IREE_BUILD_COMPILER})
