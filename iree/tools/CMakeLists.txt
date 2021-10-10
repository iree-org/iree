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
if("${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}" OR "${IREE_TARGET_BACKEND_WASM-LLVM-AOT}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::LLVM)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_LLVMAOT_TARGET")
endif()
if("${IREE_TARGET_BACKEND_METAL-SPIRV}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::MetalSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_METALSPIRV_TARGET")
endif()
if("${IREE_TARGET_BACKEND_VMVX}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VMVX)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VMVX_TARGET")
endif()
if("${IREE_TARGET_BACKEND_VULKAN-SPIRV}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VulkanSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VULKANSPIRV_TARGET")
endif()
if("${IREE_TARGET_BACKEND_CUDA}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::CUDA)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_CUDA_TARGET")
endif()
if("${IREE_TARGET_BACKEND_ROCM}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::ROCM)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_ROCM_TARGET")
endif()

if(IREE_ENABLE_EMITC)
  set(IREE_EMITC_CONDITIONAL_DEP
    MLIREmitC
  )
  set(IREE_TRANSLATE_CONDITIONAL_DEPS
    MLIREmitC
    emitc::TranslateToCpp
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
    HOSTONLY
  )

  iree_cc_library(
    NAME
      init_iree_passes_and_dialects
    HDRS
      "init_iree_dialects.h"
      "init_iree_passes.h"
    DEPS
      IREEDialectsIREEPyDMDialect
      IREEDialectsIREEDialect
      MLIRIR
      iree::compiler::Bindings::Native::Transforms
      iree::compiler::Bindings::TFLite::Transforms
      iree::compiler::Dialect::Flow::IR
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::IR::HALDialect
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::LinalgExt::IR
      iree::compiler::Dialect::LinalgExt::IR::TiledOpInterface
      iree::compiler::Dialect::LinalgExt::Transforms
      iree::compiler::Dialect::Modules::VMVX::IR::VMVXDialect
      iree::compiler::Dialect::Modules::VMVX::Transforms
      iree::compiler::Dialect::Shape::IR
      iree::compiler::Dialect::Shape::Transforms
      iree::compiler::Dialect::Stream::IR
      iree::compiler::Dialect::Util::IR
      iree::compiler::Dialect::Util::Transforms
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::IR
      iree::compiler::Dialect::VM::Transforms
      iree::compiler::Dialect::Vulkan::IR
      iree::compiler::InputConversion::Common
      iree::compiler::InputConversion::MHLO
      iree::compiler::InputConversion::TOSA
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
      MLIRGPUOps
      MLIRGPUToSPIRV
      MLIRIR
      MLIRLLVMIR
      MLIRLinalg
      MLIRLinalgToLLVM
      MLIRLinalgToSPIRV
      MLIRLinalgTransforms
      MLIRQuant
      MLIRSCF
      MLIRSCFToGPU
      MLIRSCFTransforms
      MLIRSPIRV
      MLIRSPIRVTransforms
      MLIRShape
      MLIRStandard
      MLIRStandardToSPIRV
      MLIRTosa
      MLIRTosaTransforms
      MLIRTransforms
      MLIRVector
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
      tensorflow::mlir_hlo
    PUBLIC
  )

  iree_cc_library(
    NAME
      init_translations
    HDRS
      "init_translations.h"
    DEPS
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
      ${IREE_EMITC_CONDITIONAL_DEP}
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
      MLIRLLVMToLLVMIRTranslation
      MLIRSCFTransforms
      MLIRPass
      MLIRSupport
      MLIRTargetLLVMIRExport
      MLIRTranslation
      iree::compiler::Codegen::Codegen
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Target::init_targets
      iree::compiler::Translation::IREEVM
      ${IREE_TRANSLATE_CONDITIONAL_DEPS}
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
      ${IREE_EMITC_CONDITIONAL_DEP}
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

  # IreeFileCheck
  # We could encapsulate this in iree_sh_binary, but we only have one use of it
  # in the project and this file is already not covered by Bazel->CMake

  # Symlink the shell script into the build tree as the expected executable name
  add_custom_command(
    OUTPUT
      IreeFileCheck
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/IreeFileCheck.sh
      IreeFileCheck
  )

  # CMake always needs to custom targets to drive custom commands. We have to
  # give it a different name.
  add_custom_target(iree_tools_IreeFileCheck_Driver ALL DEPENDS IreeFileCheck)

  # But custom target doesn't have the TARGET_FILE property, which we need, so
  # wrap it yet again in an executable. This is marked as an imported executable
  # because otherwise CMake would try to compile it as C++.
  add_executable(iree_tools_IreeFileCheck IMPORTED GLOBAL)
  set_property(
    TARGET
    iree_tools_IreeFileCheck
    PROPERTY IMPORTED_LOCATION
    "${CMAKE_CURRENT_BINARY_DIR}/IreeFileCheck"
  )
  add_dependencies(iree_tools_IreeFileCheck iree_tools_IreeFileCheck_Driver)
  # Finally do our normal aliasing tricks to make this accessible as a qualified
  # target name throughout the project. Accessing this as an unqualified target
  # is not supported as the name would conflict with the file name, which CMake
  # does not support (https://gitlab.kitware.com/cmake/cmake/-/issues/18627).
  add_executable(iree::tools::IreeFileCheck ALIAS iree_tools_IreeFileCheck)

  add_custom_target(BundledLLVMFileCheck ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
    DEPENDS FileCheck
  )

  # lld install - required by the compiler to link codegen executables.
  install(
    TARGETS lld
    COMPONENT Compiler
    RUNTIME DESTINATION bin
  )

  # General test dependency binary/scripts installed.
  install(
    PROGRAMS
      IreeFileCheck.bat
      run_lit.bat
      run_lit.ps1
      run_lit.sh
    DESTINATION "tests/bin"
    COMPONENT Tests
  )

  # Rename IreeFileCheck.sh -> IreeFileCheck in lieu of symlink (as is used in
  # the build tree).
  install(
    PROGRAMS
      IreeFileCheck.sh
    DESTINATION "tests/bin"
    RENAME "IreeFileCheck"
    COMPONENT Tests
  )

  # Bundle the FileCheck binary from LLVM into our tests/bin directory so
  # installed FileCheck tests are hermetic.
  install(
    TARGETS FileCheck
    DESTINATION "tests/bin"
    COMPONENT Tests
  )
endif(${IREE_BUILD_COMPILER})
