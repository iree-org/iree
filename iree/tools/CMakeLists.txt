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

  # Additional libraries containing statically registered functions/flags, which
  # should always be linked in to binaries.
  set(_ALWAYSLINK_LIBS
    MLIRAffineOps
    MLIRAnalysis
    MLIREDSC
    MLIRParser
    MLIRPass
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms
    MLIRStandardOps
    MLIRTransforms
    MLIRTranslation
    MLIRSupport
    MLIRVectorOps
    tensorflow::mlir_xla
  )

  foreach(LIB ${_ALWAYSLINK_LIBS})
    # Aliased targets are always in-project, so we control them and can set
    # ALWAYSLINK on them directly.
    # TODO(scotttodd): add the ALWAYSLINK property to MLIR libraries elsewhere
    get_target_property(_ALIASED_TARGET ${LIB} ALIASED_TARGET)
    if(NOT _ALIASED_TARGET)
      set_property(TARGET ${LIB} PROPERTY ALWAYSLINK 1)
    endif()
  endforeach(LIB)

# TODO(mabre): Get this working
#  iree_cc_binary(
#    NAME
#      iree-run-mlir
#    SRCS
#      "run_mlir_main.cc"
#    DEPS
#      absl::flags
#      absl::strings
#      # TODO(marbre): Fix problems with property "ALWAYSLINK".
#      #iree::base::source_location
#      iree::rt
#      LLVMSupport
#      MLIRIR
#      MLIRParser
#      MLIRSupport
#      iree::base::init
#      iree::base::status
#      iree::compiler::Translation::Interpreter
#      iree::compiler::Translation::SPIRV::LinalgToSPIRV
#      iree::compiler::Translation::SPIRV::XLAToSPIRV
#      iree::compiler::Translation::IREEVM
#      iree::hal::buffer_view_string_util
#      iree::hal::driver_registry
#      # TODO(marbre)_ Add deps
#      # PLATFORM_VULKAN_DEPS
#      iree::hal::interpreter::interpreter_driver_module
#      iree::hal::vulkan::vulkan_driver_module
#    )

  iree_cc_library(
    NAME
      iree_translate_library
    SRCS
      "translate_main.cc"
    DEPS
      ${_ALWAYSLINK_LIBS}
      iree::compiler::Dialect::Flow::Conversion
      iree::compiler::Dialect::HAL::Conversion
      iree::compiler::Dialect::HAL::Target::LegacyInterpreter
      iree::compiler::Dialect::HAL::Target::VulkanSPIRV
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Translation::IREEVM
      iree::compiler::Translation::SPIRV::XLAToSPIRV
      LLVMSupport
      MLIRTranslateClParser
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

endif()
