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

iree_cc_binary(
  NAME
    iree-dump-module
  OUT
    iree-dump-module
  SRCS
    "dump_module_main.cc"
  DEPS
    iree::base::file_io
    iree::base::flatbuffer_util
    iree::base::init
    iree::schemas::bytecode_module_def_cc_fbs
    flatbuffers
)
add_executable(iree-dump-module ALIAS iree_tools_iree-dump-module)

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
    iree::base::source_location
    iree::base::status
    iree::modules::hal
    iree::vm::bytecode_module
    iree::testing::benchmark_main
    # TODO(marbre): Add PLATFORM_VULKAN_DEPS
    iree::hal::interpreter::interpreter_driver_module
    # TODO(b/142004903): enable when Dawn HAL implementation is functional
    # iree::hal::dawn::dawn_driver_module
    iree::hal::vmla::vmla_driver_module
    iree::hal::vulkan::vulkan_driver_module
    iree::hal::llvmjit::llvmjit_driver_module
)
add_executable(iree-benchmark-module ALIAS iree_tools_iree-benchmark-module)

if (${IREE_ENABLE_LLVM})
  iree_cc_binary(
    NAME
      iree-tblgen
    OUT
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
    DEPS
      iree::compiler::Dialect::VM::Tools
      LLVMMLIRTableGen
      MLIRSupport
      LLVMTableGen
    LINKOPTS
      "-lpthread"
  )
  add_executable(iree-tblgen ALIAS iree_tools_iree-tblgen)
endif()

iree_cc_library(
  NAME
    vm_util
  HDRS
    "vm_util.h"
  SRCS
    "vm_util.cc"
  DEPS
    iree::base::api_util
    iree::base::buffer_string_util
    iree::base::shape
    iree::base::shaped_buffer
    iree::base::shaped_buffer_string_util
    iree::base::signature_mangle
    iree::base::status
    iree::hal::api
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
    iree::vm::module
    iree::vm::variant_list
    absl::span
)

iree_cc_test(
  NAME
    vm_util_test
  SRCS
    "vm_util_test.cc"
  DEPS
    iree::tools::vm_util
    iree::base::api
    iree::hal::api
    iree::hal::interpreter::interpreter_driver_module
    iree::modules::hal
    iree::testing::gtest_main
    iree::vm
    iree::vm::value
    iree::vm::variant_list
)

if(${IREE_BUILD_COMPILER})

  # Logic to identify targets_to_link adopted from MLIR:
  # Manually expand the target library, since our MLIR libraries
  # aren't plugged into the LLVM dependency tracking. If we don't
  # do this then we can't insert the CodeGen library after ourselves
  llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
  # Prepend LLVM in front of every target, this is how the library
  # are named with CMake
  set(targets_to_link)
  foreach(t ${TARGET_LIBS})
    list(APPEND targets_to_link "LLVM${t}")
  endforeach(t)

  iree_cc_library(
    NAME
      iree_opt_library
    DEPS
      iree::compiler::Dialect::IREE::IR
      iree::compiler::Dialect::IREE::Transforms
      iree::compiler::Dialect::Flow::Analysis
      iree::compiler::Dialect::Flow::Conversion
      iree::compiler::Dialect::Flow::IR
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::Conversion
      iree::compiler::Dialect::HAL::Conversion::FlowToHAL
      iree::compiler::Dialect::HAL::Conversion::HALToVM
      iree::compiler::Dialect::HAL::IR
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::Shape::IR
      iree::compiler::Dialect::Shape::Transforms
      iree::compiler::Dialect::VM::Analysis
      iree::compiler::Dialect::VM::Conversion
      iree::compiler::Dialect::VM::Conversion::StandardToVM
      iree::compiler::Dialect::VM::IR
      iree::compiler::Dialect::VM::Transforms
      iree::compiler::Translation::Interpreter::Transforms
      iree::compiler::Translation::IREEVM
      iree::compiler::Translation::SPIRV::LinalgToSPIRV
      iree::compiler::Translation::SPIRV::XLAToSPIRV
      iree::compiler::Translation::XLAToLinalg
      LLVMSupport
      MLIRAffineOps
      MLIRLinalgOps
      MLIRTestIR
      MLIRTestPass
      MLIRTestTransforms
      MLIRGPUtoCUDATransforms
      MLIRGPUtoNVVMTransforms
      MLIRGPUtoROCDLTransforms
      MLIRGPUtoVulkanTransforms
      MLIRQuantizerTransforms
      MLIRLinalgToSPIRVTransforms
      MLIRLinalgToLLVM
      MLIRLinalgUtils
      MLIRLoopOpsTransforms
      MLIROptLib
      MLIRStandardOps
      MLIRSPIRV
      MLIRSPIRVSerialization
      MLIRSPIRVTransforms
      MLIRSPIRVTestPasses
      MLIRTranslation
      tensorflow::mlir_xla
      # TODO(marbre): Move the HAL::Target deps to TARGET_COMPILER_BACKENDS
      iree::compiler::Dialect::HAL::Target::LegacyInterpreter
      iree::compiler::Dialect::HAL::Target::VMLA
      iree::compiler::Dialect::HAL::Target::VulkanSPIRV
      iree::compiler::Dialect::HAL::Target::LLVM
      ${targets_to_link}
    ALWAYSLINK
  )

  iree_cc_binary(
    NAME
      iree-opt
    OUT
      iree-opt
    DEPS
      iree::tools::iree_opt_library
      MLIRMlirOptMain
  )
  add_executable(iree-opt ALIAS iree_tools_iree-opt)

  iree_cc_binary(
    NAME
      iree-run-mlir
    OUT
      iree-run-mlir
    SRCS
      "run_mlir_main.cc"
    DEPS
      ::vm_util
      absl::flags
      absl::strings
      absl::span
      iree::base::api
      iree::base::api_util
      iree::base::shaped_buffer
      iree::base::shaped_buffer_string_util
      iree::base::signature_mangle
      iree::base::source_location
      iree::compiler::Dialect::Flow::Conversion
      iree::compiler::Dialect::Flow::Transforms
      iree::compiler::Dialect::HAL::Conversion
      iree::compiler::Dialect::HAL::Transforms
      iree::compiler::Dialect::IREE::Transforms
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Transforms
      iree::hal::api
      iree::modules::hal
      iree::vm
      iree::vm::bytecode_module
      iree::vm::value
      LLVMSupport
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSPIRV
      MLIRSPIRVSerialization
      MLIRSPIRVTransforms
      MLIRSupport
      iree::base::init
      iree::base::status
      iree::compiler::Translation::IREEVM
      iree::hal::buffer_view_string_util
      iree::hal::driver_registry
      # TODO(marbre): Add PLATFORM_VULKAN_DEPS
      # TODO(marbre): Move the HAL::Target deps to TARGET_COMPILER_BACKENDS
      iree::compiler::Dialect::HAL::Target::LegacyInterpreter
      iree::compiler::Dialect::HAL::Target::VMLA
      iree::compiler::Dialect::HAL::Target::VulkanSPIRV
      iree::compiler::Dialect::HAL::Target::LLVM
      iree::hal::interpreter::interpreter_driver_module
      # TODO(b/142004903): enable when Dawn HAL implementation is functional
      # iree::hal::dawn::dawn_driver_module
      iree::hal::vmla::vmla_driver_module
      iree::hal::vulkan::vulkan_driver_module
      iree::hal::llvmjit::llvmjit_driver_module
  )
  add_executable(iree-run-mlir ALIAS iree_tools_iree-run-mlir)

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
      iree::base::source_location
      iree::base::status
      iree::modules::hal
      iree::vm::bytecode_module
      # TODO(marbre): Add PLATFORM_VULKAN_DEPS
      iree::hal::interpreter::interpreter_driver_module
      # TODO(b/142004903): enable when Dawn HAL implementation is functional
      # iree::hal::dawn::dawn_driver_module
      iree::hal::vmla::vmla_driver_module
      iree::hal::vulkan::vulkan_driver_module
      iree::hal::llvmjit::llvmjit_driver_module
  )
  add_executable(iree-run-module ALIAS iree_tools_iree-run-module)

  iree_cc_library(
    NAME
      iree_translate_library
    SRCS
      "translate_main.cc"
    DEPS
      iree::compiler::Dialect::Flow::Conversion
      iree::compiler::Dialect::HAL::Conversion
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Translation::IREEVM
      iree::compiler::Translation::SPIRV::XLAToSPIRV
      LLVMSupport
      MLIRAllDialects
      MLIRAnalysis
      MLIREDSC
      MLIRParser
      MLIRIR
      MLIRPass
      MLIRSPIRVSerialization
      MLIRSPIRVTransforms
      MLIRSupport
      MLIRTransforms
      MLIRTranslateClParser
      MLIRTranslation
      tensorflow::mlir_xla
      # TODO(marbre): Move the HAL::Target deps to TARGET_COMPILER_BACKENDS
      iree::compiler::Dialect::HAL::Target::LegacyInterpreter
      iree::compiler::Dialect::HAL::Target::VMLA
      iree::compiler::Dialect::HAL::Target::VulkanSPIRV
      iree::compiler::Dialect::HAL::Target::LLVM
    ALWAYSLINK
  )

  iree_cc_binary(
    NAME
      iree-translate
    OUT
      iree-translate
    DEPS
      iree::tools::iree_translate_library
  )
  add_executable(iree-translate ALIAS iree_tools_iree-translate)

  if(${IREE_BUILD_TESTS})
    add_custom_target(IreeFileCheck ALL
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/IreeFileCheck.sh IreeFileCheck
    )
    add_custom_target(LLVMFileCheck ALL
      COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
      DEPENDS FileCheck
    )
  endif()

endif()
