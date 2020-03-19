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

add_subdirectory(test)

# TODO: iree_select_target_platform macros.
# TODO: skip targets disabled by options.
iree_select_compiler_opts(IREE_HAL_DRIVER_MODULES
  ALL
    "iree::hal::vmla::vmla_driver_module"
    "iree::hal::vulkan::vulkan_driver_module"
  CLANG_OR_GCC
    "iree::hal::llvmjit::llvmjit_driver_module"
)

iree_select_compiler_opts(IREE_COMPILER_TARGET_BACKENDS
  ALL
    "iree::compiler::Dialect::HAL::Target::VMLA"
    "iree::compiler::Dialect::HAL::Target::VulkanSPIRV"
  CLANG_OR_GCC
    "iree::compiler::Dialect::HAL::Target::LLVM"
)

iree_cc_binary(
  NAME
    iree-benchmark-module
  OUT
    iree-benchmark-module
  SRCS
    "benchmark_module_main.cc"
  DEPS
    ::vm_util
    absl::flags
    absl::strings
    benchmark
    iree::base::api_util
    iree::base::file_io
    iree::base::localfile
    iree::base::source_location
    iree::base::status
    iree::modules::hal
    iree::testing::benchmark_main
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
    "dump_module_main.cc"
  DEPS
    flatbuffers
    iree::base::file_io
    iree::base::flatbuffer_util
    iree::base::init
    iree::base::localfile
    iree::schemas::bytecode_module_def_cc_fbs
)

iree_cc_library(
  NAME
    PassesAndDialects
  HDRS
    "init_dialects.h"
    "init_passes.h"
  DEPS
    MLIRAffineOps
    MLIRAnalysis
    MLIRFxpMathOps
    MLIRGPU
    MLIRGPUtoSPIRVTransforms
    MLIRIR
    MLIRLLVMIR
    MLIRLinalgOps
    MLIRLinalgToLLVM
    MLIRLinalgToSPIRVTransforms
    MLIRLinalgTransforms
    MLIRLoopOps
    MLIRLoopOpsTransforms
    MLIRLoopsToGPU
    MLIRQuantOps
    MLIRQuantizerTransforms
    MLIRSDBM
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRStandardOps
    MLIRStandardToSPIRVTransforms
    MLIRTransforms
    MLIRVector
  PUBLIC
)

iree_cc_binary(
  NAME
    iree-run-module
  OUT
    iree-run-module
  SRCS
    "run_module_main.cc"
  DEPS
    ::vm_util
    absl::flags
    absl::strings
    iree::base::api_util
    iree::base::file_io
    iree::base::init
    iree::base::localfile
    iree::base::source_location
    iree::base::status
    iree::modules::hal
    iree::vm::bytecode_module
    ${IREE_HAL_DRIVER_MODULES}
)

if(${IREE_ENABLE_LLVM})
  iree_cc_binary(
    NAME
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
    OUT
      iree-tblgen
    DEPS
      LLVMMLIRTableGen
      MLIRSupport
      MLIRTableGen
      iree::compiler::Dialect::VM::Tools
    LINKOPTS
      "-lpthread"
  )
endif()

if(${IREE_BUILD_COMPILER})
  iree_cc_library(
    NAME
      iree_opt_main
    SRCS
      "opt_main.cc"
    DEPS
      ::PassesAndDialects
      LLVMSupport
      MLIROptLib
      MLIRPass
      MLIRSupport
      iree::compiler::Dialect::Flow::Analysis
      iree::compiler::Dialect::Flow::IR
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::Conversion::FlowToHAL
      iree::compiler::Dialect::HAL::Conversion::HALToVM
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::IREE::IR
      iree::compiler::Dialect::IREE::Transforms
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
      iree::compiler::Translation::SPIRV::LinalgToSPIRV
      iree::compiler::Translation::SPIRV::XLAToSPIRV
      iree::compiler::Translation::CodegenPasses
      tensorflow::mlir_xla
      ${IREE_COMPILER_TARGET_BACKENDS}
    PUBLIC
  )

  iree_cc_binary(
    NAME
      iree-opt
    OUT
      iree-opt
    DEPS
      ::iree_opt_main
  )

  iree_cc_binary(
    NAME
      iree-run-mlir
    OUT
      iree-run-mlir
    SRCS
      "run_mlir_main.cc"
    DEPS
      ::PassesAndDialects
      ::vm_util
      LLVMSupport
      MLIRIR
      MLIRLoopOpsTransforms
      MLIRParser
      MLIRPass
      MLIRSupport
      absl::flags
      absl::span
      absl::strings
      iree::base::api
      iree::base::api_util
      iree::base::init
      iree::base::source_location
      iree::base::status
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::IREE::Transforms
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Transforms
      iree::compiler::Translation::IREEVM
      iree::hal::api
      iree::modules::hal
      iree::vm
      iree::vm::bytecode_module
      iree::vm::value
      ${IREE_COMPILER_TARGET_BACKENDS}
      ${IREE_HAL_DRIVER_MODULES}
  )

  iree_cc_library(
    NAME
      iree_translate_main
    SRCS
      "translate_main.cc"
    DEPS
      ::PassesAndDialects
      LLVMSupport
      MLIRIR
      MLIRLoopOpsTransforms
      MLIRPass
      MLIRSupport
      MLIRTranslateClParser
      MLIRTranslation
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Translation::IREEVM
      iree::compiler::Translation::SPIRV::XLAToSPIRV
      tensorflow::mlir_xla
      ${IREE_COMPILER_TARGET_BACKENDS}
    ALWAYSLINK
    PUBLIC
  )

  iree_cc_binary(
    NAME
      iree-translate
    OUT
      iree-translate
    DEPS
      ::iree_translate_main
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
    iree::base::api_util
    iree::base::buffer_string_util
    iree::base::shape
    iree::base::shaped_buffer
    iree::base::shaped_buffer_string_util
    iree::base::signature_mangle
    iree::base::status
    iree::hal::api
    iree::modules::hal
    iree::vm::bytecode_module
    iree::vm::module
    iree::vm::variant_list
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
    iree::base::status_matchers
    iree::hal::api
    iree::hal::vmla::vmla_driver_module
    iree::modules::hal
    iree::testing::gtest_main
    iree::vm::value
    iree::vm::variant_list
)

add_custom_target(IreeFileCheck ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/IreeFileCheck.sh IreeFileCheck
)
add_custom_target(LLVMFileCheck ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
  DEPENDS FileCheck
)
