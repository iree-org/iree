# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tools IREE provides for compiling, executing, and benchmarking programs, as
# well as other utilities.
#
# Only binary targets and their associated main files should go in this
# directory. Library targets and header files should be placed in the
# appropriate subtree, e.g. `compiler/src/iree/compiler/Tools/`.
#
# Programs with a dependency on the compiler are not designed to run on device
# platforms (e.g. Android), so they are tagged "hostonly".
#
# This file does not use bazel_to_cmake because of special logic throughout.

# TODO(#6353): Tools has thread dependencies in gtest, benchmark, yaml, etc.
# This should be split between runtime/compiler with optional threading support.
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

iree_cc_binary(
  NAME
    iree-benchmark-module
  SRCS
    "iree-benchmark-module-main.cc"
  DEPS
    benchmark
    iree::base
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::modules::hal::types
    iree::tooling::context_util
    iree::tooling::device_util
    iree::tooling::vm_util
    iree::vm
)

iree_cc_binary(
  NAME
    iree-benchmark-trace
  SRCS
    "iree-benchmark-trace-main.c"
  DEPS
    iree::base
    iree::base::internal::atomic_slist
    iree::base::internal::flags
    iree::base::internal::path
    iree::base::tracing
    iree::hal
    iree::modules::hal
    iree::testing::benchmark
    iree::tooling::device_util
    iree::tooling::trace_replay
    iree::tooling::yaml_util
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
    iree::base::core_headers
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::modules::check
    iree::testing::gtest
    iree::tooling::context_util
    iree::tooling::device_util
    iree::tooling::vm_util
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
    iree::vm::bytecode_module
)

iree_cc_binary(
  NAME
    iree-run-module
  SRCS
    "iree-run-module-main.cc"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::modules::hal::types
    iree::tooling::comparison
    iree::tooling::context_util
    iree::tooling::device_util
    iree::tooling::vm_util
    iree::vm
)

iree_cc_binary(
  NAME
    iree-run-trace
  SRCS
    "iree-run-trace-main.c"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::base::internal::path
    iree::base::tracing
    iree::hal
    iree::modules::hal
    iree::tooling::device_util
    iree::tooling::trace_replay
    iree::tooling::yaml_util
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
    iree::base::internal::cpu
    iree::base::internal::flags
    iree::base::internal::path
    iree::base::tracing
    iree::hal
    iree::modules::hal
    iree::tooling::device_util
    iree::tooling::trace_replay
    iree::tooling::yaml_util
    iree::vm
    yaml
)

if(IREE_BUILD_COMPILER)
  iree_cc_binary(
    NAME
      iree-tblgen
    SRCS
      "${IREE_ROOT_DIR}/third_party/llvm-project/mlir/tools/mlir-tblgen/mlir-tblgen.cpp"
      "${IREE_SOURCE_DIR}/compiler/src/iree/compiler/Dialect/VM/Tools/VMOpEncoderGen.cpp"
      "${IREE_SOURCE_DIR}/compiler/src/iree/compiler/Dialect/VM/Tools/VMOpTableGen.cpp"
    DEPS
      LLVMSupport
      LLVMTableGen
      MLIRSupport
      MLIRTableGen
      MLIRTblgenLib
      iree::compiler::Utils
    HOSTONLY
  )

  iree_cc_binary(
    NAME
      iree-compile
    SRCS
      "iree-compile-main.cc"
    DEPS
      iree::compiler::API2::Headers
      iree::compiler::API2::Impl
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
    SETUP_INSTALL_RPATH
  )

  iree_cc_binary(
    NAME
      iree-opt
    SRCS
      "iree-opt-main.cc"
    DEPS
      iree::compiler::API2::Headers
      iree::compiler::API2::Impl
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
    SETUP_INSTALL_RPATH
  )

  iree_cc_binary(
    NAME
      iree-mlir-lsp-server
    SRCS
      "iree-mlir-lsp-server.cc"
    DEPS
      iree::compiler::API2::Headers
      iree::compiler::API2::Impl
    SETUP_INSTALL_RPATH
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
      iree::base::internal::flags
      iree::base::tracing
      iree::compiler::ConstEval
      iree::compiler::Dialect::HAL::Target
      iree::compiler::Dialect::VM::Target::Bytecode
      iree::compiler::Dialect::VM::Target::init_targets
      iree::compiler::Pipelines
      iree::compiler::Tools::init_passes_and_dialects
      iree::compiler::Tools::init_targets
      iree::hal
      iree::modules::hal::types
      iree::tooling::context_util
      iree::tooling::device_util
      iree::tooling::vm_util
      iree::vm
      iree::vm::bytecode_module
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
  )

  # Ensure FileCheck and associated binaries get built. Tests don't have
  # dependencies in CMake because they aren't targets. So until we fix that, we
  # just force this to get built.
  # Limiting this to when IREE_BUILD_TESTS is set prevents the installation
  # below, which we use for cross-platform testing.
  set_target_properties(FileCheck PROPERTIES EXCLUDE_FROM_ALL OFF)
  set_target_properties(not PROPERTIES EXCLUDE_FROM_ALL OFF)

  # Bundle the FileCheck and associated binaries from LLVM into our tests/bin
  # directory so installed FileCheck tests are hermetic.
  install(
    TARGETS FileCheck
    DESTINATION "tests/bin"
    COMPONENT Tests
  )
  install(
    TARGETS not
    DESTINATION "tests/bin"
    COMPONENT Tests
  )
endif(IREE_BUILD_COMPILER)
