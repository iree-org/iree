# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Special case package namespace for tools/ -> "" (empty string)
# For example, tools/iree-compile -> iree-compile (no namespace)
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Tools IREE provides for compiling, executing, and benchmarking programs, as
# well as other utilities.
#
# Only binary targets and their associated main files should go in this
# directory. Library targets and header files should be placed in the
# appropriate subtree, e.g. `compiler/src/iree/compiler/Tools/`.
#
# Compiler tools are designed to run on host platforms (Linux, macOS, Windows),
# so they are only built when IREE_BUILD_COMPILER is set and are tagged
# "hostonly". When cross-compiling (or generally wanting to use already-built
# compiler tools), set the IREE_HOST_BIN_DIR CMake option.
#
# This file does not use bazel_to_cmake because of special logic throughout.

# Write some important CMake options to a file for convenient use from scripts.
configure_file(build_config_template.txt.in build_config.txt)

# For sub-directories, we want targets fully qualified relative to the
# root. But after, we want those in *this* directory to be unqualified
# (i.e. 'iree-compile').
set(IREE_PACKAGE_ROOT_PREFIX "iree/tools/")
iree_add_all_subdirs()
set(IREE_PACKAGE_ROOT_PREFIX "")

# TODO(scotttodd): Should this be checking IREE_BUILD_COMPILER?
#   Maybe we should disallow setting both at the same time, since it's
#   ambigious which should be used
if(IREE_HOST_BIN_DIR AND NOT IREE_BUILD_COMPILER)
  iree_import_binary(NAME iree-tblgen)
  iree_import_binary(NAME iree-compile)
  iree_import_binary(NAME iree-opt)
  iree_import_binary(NAME iree-run-mlir)
endif()

# TODO(#6353): Tools has thread dependencies in gtest, benchmark, yaml, etc.
# This should be split between runtime/compiler with optional threading support.
if(NOT IREE_ENABLE_THREADING)
  return()
endif()

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
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::internal::path
    iree::base::tracing
    iree::hal
    iree::modules::hal
    iree::testing::benchmark
    iree::tooling::device_util
    iree::tooling::trace_replay
    iree::tooling::vm_util
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
    iree::vm::bytecode::module
  TESTONLY
)

iree_cc_binary(
  NAME
    iree-cpuinfo
  SRCS
    "iree-cpuinfo.c"
  DEPS
    iree::base
    iree::base::internal::cpu
    iree::schemas::cpu_data
)

iree_cc_binary(
  NAME
    iree-dump-instruments
  SRCS
    "iree-dump-instruments-main.c"
  DEPS
    flatcc::runtime
    iree::base
    iree::base::internal::file_io
    iree::base::internal::flatcc::parsing
    iree::schemas::instruments
    iree::schemas::instruments::dispatch_def_c_fbs
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
    iree::vm::bytecode::module
)

# Only enable fatelf tool when we're compiling it in.
# Currently it requires that the host and target both support embedded ELFs as
# the ELF implementation is only compiled when the target supports it.
if(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
iree_cc_binary(
  NAME
    iree-fatelf
  SRCS
    "iree-fatelf.c"
  DEPS
    iree::base
    iree::base::internal::file_io
    iree::base::internal::path
    iree::hal::local::elf::elf_module
)
endif()  # IREE_HAL_EXECUTABLE_*_EMBEDDED_ELF

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
    iree::tooling::instrument_util
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
    iree::tooling::vm_util
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
  # If a target backend that requires LLD to link codegen executables is
  # enabled, install the target.
  if(IREE_LLD_TARGET)
    install(
      TARGETS lld
      COMPONENT Compiler
      RUNTIME DESTINATION bin
    )
  endif()

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
      iree::compiler::bindings::c::headers
      iree::compiler::API::Impl
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
      iree::compiler::bindings::c::headers
      iree::compiler::API::Impl
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
      iree::compiler::bindings::c::headers
      iree::compiler::API::Impl
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
      iree::vm::bytecode::module
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
elseif(NOT IREE_HOST_BIN_DIR)
  message(STATUS
      "*Not* building or importing IREE's compiler tools.\n   "
      "Set IREE_BUILD_COMPILER to build them or IREE_HOST_BIN_DIR to "
      "import them.")
endif()
