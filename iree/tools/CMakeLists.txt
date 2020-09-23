# Copyright 2019 Google LLC
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

# bazel_to_cmake: DO NOT EDIT (Special logic is used throughout this file)

add_subdirectory(test)

# Enable compiler targets based on options.
set(IREE_COMPILER_TARGETS "")
set(IREE_COMPILER_TARGET_COPTS "")
# TODO(#2645): Move LLVMAOT target under DYLIB-LLVM-AOT flag
if(${IREE_TARGET_BACKEND_LLVM-IR})
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::LLVM::AOT::LLVMAOT)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_LLVMAOT_TARGET")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::LLVM::IR::LLVMIR)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_LLVMIR_TARGET")
endif()
if(${IREE_TARGET_BACKEND_VMLA})
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VMLA)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VMLA_TARGET")
endif()
if(${IREE_TARGET_BACKEND_VULKAN-SPIRV})
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VulkanSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VULKANSPIRV_TARGET")
endif()

if(IREE_ENABLE_EMITC)
  set(IREE_OPT_CONDITIONAL_DEPS
    MLIREmitC
  )
  set(IREE_TRANSLATE_CONDITIONAL_DEPS
    MLIREmitC
    MLIRTargetCpp
  )
endif()

iree_cc_binary(
  NAME
    iree-benchmark-module
  OUT
    iree-benchmark-module
  SRCS
    "iree-benchmark-module-main.cc"
  DEPS
    ::vm_util
    absl::flags
    absl::strings
    benchmark
    iree::base::init
    iree::base::file_io
    iree::base::status
    iree::base::tracing
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
    ${IREE_HAL_DRIVER_MODULES}
  TESTONLY
)

iree_cc_binary(
  NAME
    iree-dump-module
  OUT
    iree-dump-module
  SRCS
    "iree-dump-module-main.cc"
  DEPS
    flatbuffers
    iree::base::file_io
    iree::base::flatbuffer_util
    iree::base::init
    iree::schemas::bytecode_module_def_cc_fbs
)

iree_cc_binary(
  NAME
    iree-run-module
  OUT
    iree-run-module
  SRCS
    "iree-run-module-main.cc"
  DEPS
    ::vm_util
    absl::flags
    absl::strings
    iree::base::file_io
    iree::base::init
    iree::base::status
    iree::base::tracing
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
    ${IREE_HAL_DRIVER_MODULES}
)

if(${IREE_ENABLE_MLIR})
  iree_cc_binary(
    NAME
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
    DEPS
      MLIRSupport
      MLIRTableGen
      iree::compiler::Dialect::IREE::Tools
      iree::compiler::Dialect::VM::Tools
    LINKOPTS
      "-lpthread"
    HOSTONLY
  )
endif()

if(${IREE_BUILD_COMPILER})
  iree_cc_library(
    NAME
      init_iree_passes_and_dialects
    HDRS
      "init_iree_dialects.h"
      "init_iree_passes.h"
    DEPS
      MLIRIR
      iree::compiler::Dialect::Flow::Analysis
      iree::compiler::Dialect::Flow::IR
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::IR::HALDialect
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::IREE::IR
      iree::compiler::Dialect::IREE::Transforms
      iree::compiler::Dialect::Sequence::IR
      iree::compiler::Dialect::Shape::Conversion
      iree::compiler::Dialect::Shape::IR
      iree::compiler::Dialect::Shape::Transforms
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::IR
      iree::compiler::Dialect::VM::Transforms
      iree::compiler::Dialect::VMLA::IR::VMLADialect
      iree::compiler::Dialect::VMLA::Transforms
      iree::compiler::Dialect::Vulkan::IR
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
      MLIRAffineOps
      MLIRAffineTransforms
      MLIRGPU
      MLIRGPUToSPIRVTransforms
      MLIRIR
      MLIRLLVMIR
      MLIRLinalgOps
      MLIRLinalgToLLVM
      MLIRLinalgToSPIRVTransforms
      MLIRLinalgTransforms
      MLIRQuant
      MLIRSCF
      MLIRSCFToGPU
      MLIRSCFTransforms
      MLIRSDBM
      MLIRSPIRV
      MLIRSPIRVTransforms
      MLIRShape
      MLIRStandardOps
      MLIRStandardToSPIRVTransforms
      MLIRTransforms
      MLIRVector
    PUBLIC
  )

  iree_cc_library(
    NAME
      init_compiler_modules
    HDRS
      "init_compiler_modules.h"
    DEPS
      iree::compiler::Dialect::Modules::Check::IR::CheckDialect
      iree::compiler::Dialect::Modules::Strings::IR::Dialect
      iree::compiler::Dialect::Modules::TensorList::IR::TensorListDialect
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
      MLIRSPIRVSerialization
  )

  iree_cc_library(
    NAME
      iree_opt_main
    SRCS
      "iree-opt-main.cc"
    DEPS
      ::init_compiler_modules
      ::init_iree_passes_and_dialects
      ::init_mlir_passes_and_dialects
      ::init_targets
      ::init_xla_dialects
      LLVMSupport
      MLIRIR
      MLIROptLib
      MLIRParser
      MLIRPass
      MLIRTransformUtils
      MLIRSupport
      iree::compiler::Conversion::init_conversions
      iree::compiler::Conversion::HLOToLinalg
      iree::compiler::Dialect::Flow::Analysis
      iree::compiler::Dialect::Flow::IR
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::Conversion::FlowToHAL
      iree::compiler::Dialect::HAL::Conversion::HALToVM
      iree::compiler::Dialect::HAL::Conversion::Passes
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::IREE::IR
      iree::compiler::Dialect::IREE::Transforms
      iree::compiler::Dialect::Sequence::IR
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
      tensorflow::mlir_hlo
      ${IREE_OPT_CONDITIONAL_DEPS}
    PUBLIC
  )

  iree_cc_library(
    NAME
      iree_translate_main
    SRCS
      "iree-translate-main.cc"
    DEPS
      ::init_compiler_modules
      ::init_iree_passes_and_dialects
      ::init_mlir_passes_and_dialects
      ::init_targets
      ::init_translations
      ::init_xla_dialects
      LLVMSupport
      MLIRIR
      MLIRSCFTransforms
      MLIRPass
      MLIRSupport
      MLIRTranslation
      iree::compiler::Conversion::init_conversions
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Target::init_targets
      iree::compiler::Translation::IREEVM
      ${IREE_TRANSLATE_CONDITIONAL_DEPS}
    PUBLIC
  )

  iree_cc_binary(
    NAME
      iree-opt
    OUT
      iree-opt
    DEPS
      ::iree_opt_main
    HOSTONLY
  )

  iree_cc_binary(
    NAME
      iree-run-mlir
    OUT
      iree-run-mlir
    SRCS
      "iree-run-mlir-main.cc"
    DEPS
      ::init_compiler_modules
      ::init_iree_passes_and_dialects
      ::init_mlir_passes_and_dialects
      ::init_targets
      ::init_xla_dialects
      ::vm_util
      LLVMSupport
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSupport
      absl::flags
      absl::span
      absl::strings
      iree::base::api
      iree::base::init
      iree::base::source_location
      iree::base::status
      iree::base::tracing
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::IREE::Transforms
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Target::init_targets
      iree::compiler::Dialect::VM::Transforms
      iree::compiler::Translation::IREEVM
      iree::hal::api
      iree::modules::hal
      iree::vm
      iree::vm::bytecode_module
      ${IREE_HAL_DRIVER_MODULES}
    HOSTONLY
  )
endif(${IREE_BUILD_COMPILER})

# If cross-compiling, we need to declare iree-translate under host configuration
# unconditionally because we need to run it on host to generate VM modules
# for tests.
if(${IREE_BUILD_COMPILER} OR CMAKE_CROSSCOMPILING)
  iree_cc_binary(
    NAME
      iree-translate
    OUT
      iree-translate
    DEPS
      ::iree_translate_main
    HOSTONLY
  )
endif()

iree_cc_library(
  NAME
    vm_util
  HDRS
    "vm_util.h"
  SRCS
    "vm_util.cc"
  DEPS
    absl::span
    absl::strings
    iree::base::file_io
    iree::base::signature_mangle
    iree::base::status
    iree::hal::api
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
    iree::vm::ref_cc
  PUBLIC
)

iree_cc_test(
  NAME
    vm_util_test
  SRCS
    "vm_util_test.cc"
  DEPS
    ::vm_util
    iree::base::api
    iree::hal::api
    iree::hal::vmla::vmla_driver_module
    iree::modules::hal
    iree::testing::gtest
    iree::testing::gtest_main
    iree::vm
)

if(${IREE_ENABLE_MLIR})
  add_custom_target(IreeFileCheck ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/IreeFileCheck.sh IreeFileCheck
  )
  if(${IREE_MLIR_DEP_MODE} STREQUAL "BUNDLED")
    add_custom_target(BundledLLVMFileCheck ALL
      COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
      DEPENDS FileCheck
    )
  endif()
endif()
