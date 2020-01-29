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

if(${IREE_BUILD_COMPILER})

  iree_cc_library(
    NAME
      iree_opt_library
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-opt/mlir-opt.cpp"
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
      iree::compiler::Translation::SPIRV::XLAToSPIRV
      iree::compiler::Translation::SPIRV::LinalgToSPIRV
      iree::compiler::Translation::XLAToLinalg
      iree::compiler::Translation::XLAToLinalg::IREELinalgTensorToBuffer
      LLVMSupport
      MLIRAffineOps
      MLIRLinalgOps
      MLIRLinalgUtils
      MLIRStandardOps
      MLIRSPIRV
      MLIRSPIRVSerialization
      MLIRSPIRVTransforms
      MLIRTranslation
      tensorflow::mlir_xla
    ALWAYSLINK
  )

  iree_cc_binary(
    NAME
      iree-opt
    OUT
      iree-opt
    DEPS
      iree::tools::iree_opt_library
      MLIROptMain
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
      absl::flags
      absl::strings
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
      iree::compiler::Dialect::HAL::Target::VulkanSPIRV
      iree::hal::vulkan::vulkan_driver_module
  )
  add_executable(iree-run-mlir ALIAS iree_tools_iree-run-mlir)

  iree_cc_library(
    NAME
      iree_translate_library
    SRCS
      "translate_main.cc"
    DEPS
      iree::compiler::Dialect::Flow::Conversion
      iree::compiler::Dialect::HAL::Conversion
      iree::compiler::Dialect::HAL::Target::LegacyInterpreter
      iree::compiler::Dialect::HAL::Target::VulkanSPIRV
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Translation::IREEVM
      iree::compiler::Translation::SPIRV::XLAToSPIRV
      LLVMSupport
      MLIRAffineOps
      MLIRAnalysis
      MLIREDSC
      MLIRParser
      MLIRIR
      MLIRPass
      MLIRSPIRV
      MLIRSPIRVSerialization
      MLIRSPIRVTransforms
      MLIRSupport
      MLIRStandardOps
      MLIRTransforms
      MLIRTranslateClParser
      MLIRTranslation
      MLIRVectorOps
      tensorflow::mlir_xla
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
