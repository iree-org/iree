# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
# CAPI library using the LLVM build system.
#
# WARNING WILL ROBINSON!
# This does not look like the rest of IREE. It is directly using the upstream
# LLVM build system in order to create bundled compiler API binaries that
# are consistent with LLVM. Consult upstream CMake macros if you don't
# understand what this does.
################################################################################

include(AddLLVM)
include(AddMLIR)

# LLD may not have been enabled at the global level (typically because we
# are not building for a CPU backend). If not, don't add API support for it.
set(_OPTIONAL_SOURCES)
set(_OPTIONAL_LINK_LIBS)
if(TARGET lld)
  list(APPEND _OPTIONAL_SOURCES Lld.cpp)
  list(APPEND _OPTIONAL_LINK_LIBS
    lldCommon
    lldCOFF
    lldELF
    lldMachO
    lldMinGW
    lldWasm
  )
endif()

add_mlir_public_c_api_library(IREECompilerCAPILib
  Compiler.cpp
  Tools.cpp
  ${_OPTIONAL_SOURCES}
  PARTIAL_SOURCES_INTENDED
  # TODO: If installing, complains about IREEVM not being in any export set.
  DISABLE_INSTALL
  LINK_COMPONENTS
    Support
  LINK_LIBS PUBLIC
    MLIRIR
    iree::compiler::ConstEval
    iree::compiler::InputConversion::MHLO::MHLO
    iree::compiler::InputConversion::TOSA::TOSA
    iree::compiler::Dialect::VM::IR::IR
    iree::compiler::Dialect::VM::Target::Bytecode::Bytecode
    iree::compiler::Pipelines

    # Passes and dialects.
    iree::compiler::Tools::init_llvmir_translations
    iree::compiler::Tools::init_passes_and_dialects

    # All HAL Targets.
    iree::compiler::Tools::init_targets

    # Tools.
    iree::compiler::Tools::iree_compile_lib

    ${_OPTIONAL_LINK_LIBS}
)

if(IREE_BUILD_TORCH_MLIR_SUPPORT)
  target_compile_definitions(obj.IREECompilerCAPILib
    PUBLIC
    IREE_HAVE_TORCH_MLIR_DIALECTS
  )
endif()

add_subdirectory(test)

################################################################################
# Language specific bindings.
# These have an ordering dependency on the backing CAPI being defined.
################################################################################

if(IREE_BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
