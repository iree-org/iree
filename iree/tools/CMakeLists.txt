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

add_subdirectory(android)
add_subdirectory(test)
add_subdirectory(utils)

# Enable compiler targets based on options.
set(IREE_COMPILER_TARGETS "")
set(IREE_COMPILER_TARGET_COPTS "")
if("${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::LLVM)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_LLVMAOT_TARGET")
endif()
if("${IREE_TARGET_BACKEND_METAL-SPIRV}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::MetalSPIRV)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_METALSPIRV_TARGET")
endif()
if("${IREE_TARGET_BACKEND_VMLA}")
  list(APPEND IREE_COMPILER_TARGETS iree::compiler::Dialect::HAL::Target::VMLA)
  list(APPEND IREE_COMPILER_TARGET_COPTS "-DIREE_HAVE_VMLA_TARGET")
endif()
if("${IREE_TARGET_BACKEND_VULKAN-SPIRV}")
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
  set(IREE_EMITC_INCLUDES
    "${PROJECT_SOURCE_DIR}/third_party/mlir-emitc/include"
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
    absl::flags
    absl::flags_parse
    absl::flags_usage
    absl::strings
    benchmark
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::status
    iree::base::tracing
    iree::hal::drivers
    iree::modules::hal
    iree::tools::utils::vm_util
    iree::vm
    iree::vm::bytecode_module
  TESTONLY
)

iree_cc_binary(
  NAME
    iree-check-module
  OUT
    iree-check-module
  SRCS
    "iree-check-module-main.cc"
  DEPS
    iree::modules::check::native_module
    absl::flags
    absl::strings
    iree::base::api
    iree::base::core_headers
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::status
    iree::base::tracing
    iree::hal::drivers
    iree::modules::hal
    iree::testing::gtest
    iree::tools::utils::vm_util
    iree::vm::bytecode_module
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
    flatcc::runtime
    iree::base::internal::file_io
    iree::schemas::bytecode_module_def_c_fbs
)

iree_cc_binary(
  NAME
    iree-run-module
  OUT
    iree-run-module
  SRCS
    "iree-run-module-main.cc"
  DEPS
    absl::flags
    absl::strings
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::status
    iree::base::tracing
    iree::hal::drivers
    iree::modules::hal
    iree::tools::utils::vm_util
    iree::vm
    iree::vm::bytecode_module
)

if(${IREE_BUILD_COMPILER})
  iree_cc_binary(
    NAME
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
      "${IREE_SOURCE_DIR}/iree/compiler/Dialect/IREE/Tools/StructAttrGen.cpp"
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
      MLIRIR
      iree::compiler::Bindings::TFLite::Transforms
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
      MLIRAffine
      MLIRAffineTransforms
      MLIRGPU
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
      MLIRSDBM
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
      iree::compiler::Conversion::init_conversions
      iree::compiler::Conversion::HLOToLinalg
      iree::compiler::Dialect::HAL::Conversion::Passes
    INCLUDES
      "${IREE_EMITC_INCLUDES}"
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
    INCLUDES
      "${IREE_EMITC_INCLUDES}"
    PUBLIC
  )

  iree_cc_binary(
    NAME
      iree-translate
    OUT
      iree-translate
    DEPS
      ::iree_translate_main
    HOSTONLY
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
      ::init_passes_and_dialects
      ::init_targets
      LLVMSupport
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRSupport
      absl::flags
      absl::span
      absl::strings
      iree::base::api
      iree::base::internal::flags
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
      iree::hal::drivers
      iree::modules::hal
      iree::tools::utils::vm_util
      iree::vm
      iree::vm::bytecode_module
    HOSTONLY
  )

  add_custom_target(IreeFileCheck ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/IreeFileCheck.sh IreeFileCheck
  )
  if(${IREE_MLIR_DEP_MODE} STREQUAL "BUNDLED")
    add_custom_target(BundledLLVMFileCheck ALL
      COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
      DEPENDS FileCheck
    )
  endif()

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
