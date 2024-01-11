# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
if (NOT IREE_TARGET_BACKEND_ROCM)
  return()
endif()

# Check if HIP is installed on system.
# HIP is required to compile ukernels.
if (NOT IREE_ROCM_PATH)
  set(IREE_ROCM_PATH "/opt/rocm")
endif()
set (IREE_ROCM_VERSION "${IREE_ROCM_PATH}/.info/version")
if (NOT EXISTS ${IREE_ROCM_VERSION})
  message(STATUS
          "hip runtime cannot be found in ${IREE_ROCM_PATH}.
          Please try setting IREE_ROCM_PATH to rocm directory.
          Ukernel will not be compiled.")
  return()
endif()


iree_add_all_subdirs()

set(_platform_lib_reldir "iree_platform_libs/rocm")
set(_device_bc_path "${IREE_COMPILER_DYLIB_DIR}/iree_platform_libs/rocm")
set (_amd_ukernel_libs)
set (_amd_ukernel_targets)
function(iree_rocm_bitcode_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;ROCM_ARCH"
    "SRCS;COPTS"
    ${ARGN}
  )

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}_${_RULE_ROCM_ARCH}.bc")
  endif()

  set(_ROCM_ARCH "${_RULE_ROCM_ARCH}")
  set (OPT_FLAG "-O0")
  if (_ROCM_ARCH MATCHES "GFX9")
    set (OPT_FLAG "-O3")
  endif()
  set(_COPTS
    "-x" "hip"
    # Target architecture.
    "--offload-arch=${_ROCM_ARCH}"

    # Suppress warnings about missing path to rocm lib,
    # and benign warning about ROCM version.
    "-nogpulib"
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    # Avoid error caused by clang bundling.
    "--offload-device-only"
    # Only enable necessary optimizations S.T we can use -O3.
    "-Xclang" "-disable-llvm-optzns"
    "${OPT_FLAG}"

    # Object file only in bitcode format:
    "-c"
    "-emit-llvm"
  )

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    set(_BITCODE_FILE "${_RULE_NAME}_${_SRC}_${_ROCM_ARCH}.bc")
    list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
    add_custom_command(
      OUTPUT
        "${_BITCODE_FILE}"
      COMMAND
        "${IREE_CLANG_BINARY}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${IREE_CLANG_BINARY}"
        "${_SRC}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  add_custom_command(
    OUTPUT
      "${_OUT}"
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
      "-o"
      "${_OUT}"
    DEPENDS
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
    COMMENT
      "Linking bitcode to ${_OUT}"
    VERBATIM
  )
  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}_${_ROCM_ARCH}"
    DEPENDS "${_OUT}"
  )
  set(_amd_ukernel_libs ${_amd_ukernel_libs} ${_OUT} PARENT_SCOPE)
  set(_amd_ukernel_targets ${_amd_ukernel_targets} "${_PACKAGE_NAME}_${_RULE_NAME}_${_ROCM_ARCH}" PARENT_SCOPE)
endfunction()

# TODO: Decide what to build by default. No real constaints here
#       except compile-time cost, so just picked out the popular ones.
set(_ukernel_supported_chips "gfx90a" "gfx940" "gfx1030" "gfx1100")
foreach(_amd_chip ${_ukernel_supported_chips})
  iree_rocm_bitcode_library(
    NAME
      rocm_argmax_ukernel
    ROCM_ARCH
      ${_amd_chip}
    SRCS
      "argmax_ukernel.c"
  )
endforeach()

# Copy UKernel into platform dir.
set(_all_ukernel_bc_copy_commands)
set(_all_ukernel_bc_files)
set(_ukernel_lib_srcdir ${CMAKE_CURRENT_BINARY_DIR})
foreach(_amd_ukernel_name ${_amd_ukernel_libs})
  # Copy to lib/ tree.
  set(_ukernel_bc_srcpath "${_ukernel_lib_srcdir}/${_amd_ukernel_name}")
  set(_ukernel_bc_relpath "${_platform_lib_reldir}/${_amd_ukernel_name}")
  list(APPEND _all_ukernel_bc_files "${IREE_COMPILER_DYLIB_DIR}/${_ukernel_bc_relpath}")
  list(APPEND _all_ukernel_bc_deps "${_ukernel_bc_path}")
  list(APPEND _all_ukernel_bc_copy_commands
    COMMAND ${CMAKE_COMMAND} -E copy
      "${_ukernel_bc_srcpath}"
      "${IREE_COMPILER_DYLIB_DIR}/${_ukernel_bc_relpath}"
  )
  # Note this bc file as being part of the bundle that must be included with
  # the compiler dylib.
  set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_DYLIB_RELPATHS "${_ukernel_bc_relpath}")
endforeach()

# Generate a custom target with all file level dependencies and commands to
# copy to our build tree locations.
# Our GenDeviceLibs target depends on all of the defined device lib targets.
add_custom_command(
  OUTPUT ${_all_ukernel_bc_files}
  DEPENDS ${_amd_ukernel_targets}
  POST_BUILD
    ${_all_ukernel_bc_copy_commands}
)

add_custom_target(iree_builtin_ROCM_UkernelDeviceLibs
  DEPENDS
    ${_all_ukernel_bc_files}
)

# Ensure that the device libs are built when the compiler dylib is built.
set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_DYLIB_DEPENDS
  iree_builtin_ROCM_UkernelDeviceLibs)

# Install.
install(FILES ${_all_ukernel_bc_files}
  DESTINATION "${IREE_COMPILER_DYLIB_INSTALL_PREFIX}/${_platform_lib_reldir}")
