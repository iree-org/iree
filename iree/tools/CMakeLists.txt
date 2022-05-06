# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of various special logic throughout.

# TODO(#6353)Tools has thread dependency in gtest and benchmark; they should
# be separated into the runtime tools and compiler tools.
if(NOT IREE_ENABLE_THREADING)
  return()
endif()

# Write some important CMake options to a file for convenient use from scripts.
configure_file(build_config_template.txt.in build_config.txt)

# Depending on which target backends are enabled, we may or may not have built
# LLD.
if(IREE_LLD_TARGET)
  # lld install - required by the compiler to link codegen executables.
  install(
    TARGETS lld
    COMPONENT Compiler
    RUNTIME DESTINATION bin
  )
endif()

add_subdirectory(android)
add_subdirectory(test)
add_subdirectory(utils)

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
    "iree-dump-module-main.c"
  DEPS
    flatcc::runtime
    iree::base
    iree::base::internal::file_io
    iree::base::internal::flatcc::debugging
    iree::schemas::bytecode_module_def_c_fbs
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
    iree::tools::utils::cpu_features
    iree::tools::utils::trace_replay
    iree::tools::utils::yaml_util
    iree::vm
    yaml
)

if(IREE_BUILD_COMPILER)
  iree_cc_binary(
    NAME
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
      "${IREE_SOURCE_DIR}/compiler/src/iree/compiler/Dialect/Util/Tools/StructAttrGen.cpp"
      "${IREE_SOURCE_DIR}/compiler/src/iree/compiler/Dialect/VM/Tools/VMOpEncoderGen.cpp"
      "${IREE_SOURCE_DIR}/compiler/src/iree/compiler/Dialect/VM/Tools/VMOpTableGen.cpp"
    DEPS
      LLVMSupport
      LLVMTableGen
      MLIRSupport
      MLIRTableGen
      iree::compiler::Utils
    HOSTONLY
  )

  iree_cc_binary(
    NAME
      iree-translate
    SRCS
      "iree-translate-main.cc"
    DEPS
      iree::compiler::Tools::iree_translate_lib
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
  )

  iree_cc_binary(
    NAME
      iree-compile
    SRCS
      "iree-compile-main.cc"
    DEPS
      iree::compiler::Tools::iree_translate_lib
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
  )

  iree_cc_library(
    NAME
      iree_opt_main
    SRCS
      "iree-opt-main.cc"
    DEPS
      LLVMSupport
      MLIRIR
      MLIROptLib
      MLIRSupport
      iree::compiler::Tools::init_passes_and_dialects
      iree::compiler::Tools::init_targets
    PUBLIC
  )

  iree_cc_binary(
    NAME
      iree-opt
    DEPS
      ::iree_opt_main
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
  )

  iree_cc_binary(
    NAME
      iree-mlir-lsp-server
    SRCS
      "iree-mlir-lsp-server.cc"
    DEPS
      MLIRIR
      MLIRLspServerLib
      MLIRSupport
      iree::compiler::Tools::init_passes_and_dialects
    PUBLIC
    EXCLUDE_FROM_ALL
  )

  iree_cc_binary(
    NAME
      iree-run-mlir
    SRCS
      "iree-run-mlir-main.cc"
    DEPS
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
      iree::compiler::Tools::init_passes_and_dialects
      iree::compiler::Tools::init_targets
      iree::compiler::Translation::IREEVM
      iree::hal
      iree::hal::drivers
      iree::modules::hal
      iree::tools::utils::vm_util
      iree::vm
      iree::vm::bytecode_module
      iree::vm::cc
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
  )

  # Ensure FileCheck gets built. Tests don't have dependencies in CMake because
  # they aren't targets. So until we fix that, we just force this to get built.
  # Limiting this to when IREE_BUILD_TESTS is set prevents the installation
  # below, which we use for cross-platform testing.
  set_target_properties(FileCheck PROPERTIES EXCLUDE_FROM_ALL OFF)

  # Bundle the FileCheck binary from LLVM into our tests/bin directory so
  # installed FileCheck tests are hermetic.
  install(
    TARGETS FileCheck
    DESTINATION "tests/bin"
    COMPONENT Tests
  )
endif(IREE_BUILD_COMPILER)
