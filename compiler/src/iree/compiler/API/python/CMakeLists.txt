# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under npcomp.
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=iree.compiler.")

set(_PYTHON_BUILD_PREFIX "${IREE_BINARY_DIR}/compiler/bindings/python")
set(_PYTHON_INSTALL_PREFIX "python_packages/iree_compiler")

# HACK: This should not be necessary, but add_mlir_python_extension is
# accidentally closing over an errant include_directories from up-tree, so
# when built in-tree it is somehow working based on that. This will need
# to be fixed to capture the correct include directories in that macro.
include_directories(
  "${IREE_SOURCE_DIR}/compiler/src"
  "${IREE_SOURCE_DIR}/llvm-external-projects/iree-dialects/include"
  "${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include"
  "${IREE_SOURCE_DIR}/third_party/mlir-hlo/include"
)

# On Unixes, disable the creation of versioned/symlinked `.so` files. With
# this set, we just generate libIREECompilerAggregateCAPI.so vs making that
# a symlink to a versioned file right next to it. When packaging for Python,
# symlinks are duplicated, so this is pretty important. It is usually set at
# a toolchain level for dedicated Python builds, but is nice to override since
# it makes the build directory suitable for directly packaging.
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(IREECompilerAPIPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  SOURCES
    transforms/ireec.py
)
declare_mlir_python_sources(IREECompilerAPIPythonExtensions)

declare_mlir_python_sources(IREECompilerAPIPythonTools
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  SOURCES
    __init__.py
    tf.py
    tflite.py
    xla.py
  SOURCES_GLOB
    tools/*.py
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(IREECompilerAPIPythonExtensions.IREECTransforms
  MODULE_NAME _ireecTransforms
  ADD_TO_PARENT IREECompilerAPIPythonExtensions
  SOURCES
    IREECTransforms.cpp
  EMBED_CAPI_LINK_LIBS
    IREECompilerCAPILib
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
################################################################################

set(_SOURCE_COMPONENTS
  # Local sources.
  IREECompilerAPIPythonSources
  IREECompilerAPIPythonExtensions
  IREECompilerAPIPythonTools

  MLIRPythonSources.Core

  # Core dialects (constrained to IREE input dialects).
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.pdl
  MLIRPythonSources.Dialects.shape
  MLIRPythonSources.Dialects.structured_transform
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.tosa
  MLIRPythonSources.Dialects.transform
  MLIRPythonSources.Dialects.vector

  # mhlo project.
  MLIRHLOPythonSources
  MLIRHLOPythonExtensions

  # iree-dialects project.
  IREEDialectsPythonSources
  IREEDialectsPythonExtensions
)

add_mlir_python_common_capi_library(IREECompilerAggregateCAPI
  INSTALL_COMPONENT IREECompilerPythonModules
  INSTALL_DESTINATION ${_PYTHON_INSTALL_PREFIX}/iree/compiler/_mlir_libs
  OUTPUT_DIRECTORY "${_PYTHON_BUILD_PREFIX}/iree/compiler/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
)

add_mlir_python_modules(IREECompilerPythonModules
  ROOT_PREFIX "${_PYTHON_BUILD_PREFIX}/iree/compiler"
  INSTALL_PREFIX "${_PYTHON_INSTALL_PREFIX}/iree/compiler"
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
    IREECompilerAggregateCAPI
  )


################################################################################
# Tools linked against the shared CAPI library
################################################################################

function(add_iree_compiler_busybox_tool target)
  cmake_parse_arguments(ARG
    ""
    "OUTPUT_NAME"
    "SRCS"
    ${ARGN})

  add_executable(
    ${target}
    ${ARG_SRCS}
  )
  target_link_libraries(${target} IREECompilerAggregateCAPI)
  set_target_properties(${target}
    PROPERTIES
      OUTPUT_NAME "${ARG_OUTPUT_NAME}"
      RUNTIME_OUTPUT_DIRECTORY "${_PYTHON_BUILD_PREFIX}/iree/compiler/_mlir_libs"
  )
  mlir_python_setup_extension_rpath(${target})
  add_dependencies(IREECompilerPythonModules ${target})
  install(TARGETS ${target}
    DESTINATION "${_PYTHON_INSTALL_PREFIX}/iree/compiler/_mlir_libs"
  )
endfunction()

add_iree_compiler_busybox_tool(
  IREECompilerIREECompileTool
  OUTPUT_NAME iree-compile
  SRCS
    IREECompileTool.c
)

if(TARGET lld)
  add_iree_compiler_busybox_tool(
    IREECompilerLldTool
    OUTPUT_NAME iree-lld
    SRCS
      LldTool.c
  )
endif()

################################################################################
# Subdirectories
################################################################################

add_subdirectory(test)
