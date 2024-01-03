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
  "${IREE_SOURCE_DIR}/compiler/bindings/c"
  "${IREE_SOURCE_DIR}/llvm-external-projects/iree-dialects/include"
  "${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include"
  "${IREE_SOURCE_DIR}/third_party/stablehlo/include"
  "${IREE_SOURCE_DIR}/third_party/stablehlo"
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

declare_mlir_python_sources(IREEPythonSources)
declare_mlir_python_sources(IREEPythonSources.Dialects
  ADD_TO_PARENT IREEPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/FlowBinding.td
  GEN_ENUM_BINDINGS
  SOURCES dialects/flow.py
  DIALECT_NAME flow
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/HALBinding.td
  GEN_ENUM_BINDINGS
  SOURCES dialects/hal.py
  DIALECT_NAME hal
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/StreamBinding.td
  GEN_ENUM_BINDINGS
  SOURCES dialects/stream.py
  DIALECT_NAME stream
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/UtilBinding.td
  GEN_ENUM_BINDINGS
  SOURCES dialects/util.py
  DIALECT_NAME util
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/VMBinding.td
  GEN_ENUM_BINDINGS
  SOURCES dialects/vm.py
  DIALECT_NAME vm
)

declare_mlir_python_sources(IREECompilerAPIPythonCore
  ADD_TO_PARENT IREEPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  SOURCES
    _package_test.py
    api/__init__.py
    api/ctypes_dl.py
)

# Note that some tools rely on optional features but we unconditionally
# include them because they are referenced from console scripts and
# other package metadata. They will detect mis-configuration and error
# accordingly at runtime.
declare_mlir_python_sources(IREECompilerAPIPythonTools
  ADD_TO_PARENT IREEPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  SOURCES
    __init__.py
    tf.py
    tflite.py
    tools/__init__.py
    tools/binaries.py
    tools/core.py
    tools/debugging.py
    tools/tf.py
    tools/tflite.py
    tools/import_onnx/__main__.py
    tools/ir_tool/__main__.py
    tools/scripts/ireec/__main__.py
)

# The Python bindings are monolithic and we don't have a good way for the
# torch plugin to contribute Python sources, so we just gate it here
# versus having more complicated indirection. May want to rethink this
# if others need it.
if(IREE_INPUT_TORCH)

  declare_mlir_python_sources(IREEPythonSources.Torch.Importers
    ADD_TO_PARENT IREEPythonSources
    ROOT_DIR "${IREE_SOURCE_DIR}/third_party/torch-mlir/python/torch_mlir"
    SOURCES
      extras/onnx_importer.py
  )

endif()

################################################################################
# Extensions
################################################################################

declare_mlir_python_sources(IREECompilerPythonExtensions)

declare_mlir_python_extension(IREECompilerPythonExtensions.Registration
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT IREECompilerPythonExtensions
  SOURCES
    IREECompilerRegistration.cpp
  EMBED_CAPI_LINK_LIBS
    iree_compiler_API_SharedImpl
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
################################################################################

set(_SOURCE_COMPONENTS
  # Local sources.
  IREECompilerAPIPythonTools
  IREECompilerPythonExtensions
  IREEPythonSources

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
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.shape
  MLIRPythonSources.Dialects.structured_transform
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.tosa
  MLIRPythonSources.Dialects.transform
  MLIRPythonSources.Dialects.vector

  # iree-dialects project.
  IREEDialectsPythonSources
  IREEDialectsPythonExtensions
)

add_mlir_python_modules(IREECompilerPythonModules
  ROOT_PREFIX "${_PYTHON_BUILD_PREFIX}/iree/compiler"
  INSTALL_PREFIX "${_PYTHON_INSTALL_PREFIX}/iree/compiler"
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
    iree_compiler_API_SharedImpl
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
  target_link_libraries(${target}
    iree_compiler_bindings_c_headers
    iree_compiler_API_Impl)
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
# libIREECompiler.so dylib tree
# We copy the compiler shared library and its related files to both the build
# and install tree for consistency. Note that on Windows, this is load bearing
# since the DLL must be colocated in the same directory. On Unix, CMake's
# automatic RPATH handling in the build tree will cause binaries to resolve
# against the project level library, so this is just cosmetic, but it does
# have the side effect of making the Python build tree relocatable and making
# logic that expects to find the compiler libraries in a consistent place
# simple. Since a symlink is used on Unix and it is not the versioned dylib, 
# the realpath will still be the build-tree wide shared library. This also means
# that any tools or files that must be colocated with the dylib will come from
# the project-wide location.
################################################################################

# Copy compiler dylib files into the python _mlir_libs tree so that the
# binaries can find them with a relative rpath.
get_property(_dylib_relpaths GLOBAL PROPERTY IREE_COMPILER_DYLIB_RELPATHS)
set(_dylib_copy_commands)
set(_dylib_target_files)
set(_dylib_src_files)
foreach (_dylib_relpath ${_dylib_relpaths})
  set(_dylib_srcpath "${IREE_COMPILER_DYLIB_DIR}/${_dylib_relpath}")
  set(_dylib_outputpath "${_PYTHON_BUILD_PREFIX}/iree/compiler/_mlir_libs/${_dylib_relpath}")
  cmake_path(GET _dylib_outputpath PARENT_PATH _parent_dir)
  file(MAKE_DIRECTORY "${_parent_dir}")
  list(APPEND _dylib_copy_commands
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${_dylib_srcpath}" "${_dylib_outputpath}"
  )
  list(APPEND _dylib_target_files "${_dylib_outputpath}")
  list(APPEND _dylib_src_files "${_dylib_srcpath}")
  cmake_path(GET _dylib_relpath PARENT_PATH _dylib_install_destination)
  install(FILES "${_dylib_srcpath}"
    DESTINATION "${_PYTHON_INSTALL_PREFIX}/iree/compiler/_mlir_libs/${_dylib_install_destination}"
  )
endforeach()

add_custom_command(
  OUTPUT ${_dylib_target_files}
  DEPENDS ${_dylib_src_files}
  POST_BUILD
    ${_dylib_copy_commands}
)

add_custom_target(IREECompilerPythonDylibFiles
  DEPENDS ${_dylib_target_files}
)

add_dependencies(IREECompilerPythonModules IREECompilerPythonDylibFiles)

################################################################################
# Subdirectories
################################################################################

add_subdirectory(test)
