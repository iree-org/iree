# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_bitcode_library()
#
# Builds an LLVM bitcode library from an input file via clang
#
# Parameters:
# NAME: Name of target (see Note).
# SRCS: Source files. Headers go here as well, as in iree_cc_library. There is
#       no concept of public headers (HDRS) here.
# COPTS: additional flags to pass to clang.
# OUT: Output file name (defaults to NAME.bc).
function(iree_bitcode_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;ARCH"
    "INTERNAL_HDRS;SRCS;COPTS"
    ${ARGN}
  )

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  # Produce an empty file if the compiler wouldn't use bitcode for this arch anyway.
  iree_compiler_targeting_iree_arch(_IREE_COMPILER_TARGETING_THIS_ARCH "${_RULE_ARCH}")
  if (NOT _IREE_COMPILER_TARGETING_THIS_ARCH)
    iree_make_empty_file("${_OUT}")
    return()
  endif()

  iree_arch_to_llvm_arch(_LLVM_ARCH "${_RULE_ARCH}")

  set(_COPTS
    # Target architecture.
    "-target" "${_LLVM_ARCH}"

    # C17 with no system deps.
    "-std=c17"
    "-nostdinc"
    "-ffreestanding"

    # Optimized and unstamped.
    "-O3"
    "-DNDEBUG"
    "-fno-ident"
    "-fdiscard-value-names"

    # Set the size of wchar_t to 4 bytes (instead of 2 bytes).
    # This must match what the runtime is built with.
    "-fno-short-wchar"

    # Enable inline asm.
    "-fasm"

    # Object file only in bitcode format:
    "-c"
    "-emit-llvm"

    # Force the library into standalone mode (not depending on build-directory
    # configuration).
    "-DIREE_DEVICE_STANDALONE=1"
  )

  list(APPEND _COPTS "-isystem" "${IREE_CLANG_BUILTIN_HEADERS_PATH}")
  list(APPEND _COPTS "-I" "${IREE_SOURCE_DIR}/runtime/src")
  list(APPEND _COPTS "-I" "${IREE_BINARY_DIR}/runtime/src")
  list(APPEND _COPTS "${_RULE_COPTS}")

  if (_RULE_ARCH STREQUAL "arm_32")
    # Silence "warning: unknown platform, assuming -mfloat-abi=soft"
    list(APPEND _COPTS "-mfloat-abi=soft")
  elseif(_RULE_ARCH STREQUAL "riscv_32")
    # On RISC-V, linking LLVM modules requires matching target-abi.
    # https://lists.llvm.org/pipermail/llvm-dev/2020-January/138450.html
    # The choice of ilp32d is simply what we have in existing riscv_32 tests.
    # Open question - how do we scale to supporting all RISC-V ABIs?
    list(APPEND _COPTS "-mabi=ilp32d")
  elseif(_RULE_ARCH STREQUAL "riscv_64")
    # Same comments as above riscv_32 case.
    list(APPEND _COPTS "-mabi=lp64d")
  endif()

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    set(_BITCODE_FILE "${_RULE_NAME}_${_SRC}.bc")
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
        "${_RULE_INTERNAL_HDRS}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  add_custom_command(
    OUTPUT
      ${_OUT}
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
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

function(iree_cuda_bitcode_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;CUDA_ARCH"
    "SRCS;COPTS"
    ${ARGN}
  )

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  set(_CUDA_ARCH "${_RULE_CUDA_ARCH}")

  set(_COPTS
    "-x" "cuda"

    # Target architecture.
    "--cuda-gpu-arch=${_CUDA_ARCH}"

    "--cuda-path=${CUDAToolkit_ROOT}"

    # Suppress warnings about missing path to cuda lib,
    # and benign warning about CUDA version.
    "-Wno-unknown-cuda-version"
    "-nocudalib"
    "--cuda-device-only"

    # https://github.com/llvm/llvm-project/issues/54609
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

    # Optimized and unstamped.
    "-O3"

    # Object file only in bitcode format:
    "-c"
    "-emit-llvm"
  )

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    set(_BITCODE_FILE "${_RULE_NAME}_${_SRC}.bc")
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
      ${_OUT}
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
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

# iree_amdgpu_bitcode_library()
#
# Builds an AMDGPU LLVM bitcode library from an input file via clang.
#
# Parameters:
# NAME: Name of the target.
# GPU_ARCH: Target AMDGPU architecture, e.g. gfx942.
# SRCS: Source files to pass to clang. Headers (*.h) are for dependency
#       tracking only. Current limitation: only one non-header source is
#       supported.
# COPTS: Additional flags to pass to clang.
# OUT: Output file name. Defaults to {source.c}.{gpu_arch}.bc.
#
function(iree_amdgpu_bitcode_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;GPU_ARCH"
    "SRCS;COPTS"
    ${ARGN}
  )

  set(_SRC "")
  foreach(_SRCS_ENTRY IN LISTS _RULE_SRCS)
    if(_SRCS_ENTRY MATCHES "\.h$")
      continue()
    endif()
    if (_SRC)
      message(SEND_ERROR "Currently limitation: only one non-header file allowed in SRCS.")
    endif()
    set(_SRC "${_SRCS_ENTRY}")
  endforeach()
  if(NOT _SRC)
    message(SEND_ERROR "Error: no non-header file found in SRCS=${_RULE_SRCS}.")
  endif()

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_SRC}.${_RULE_GPU_ARCH}.bc")
  endif()

  set(_COPTS
    # Language: C23
    "-std=c23"

    # Avoid dependencies.
    "-nogpulib"

    # Avoid ABI issues.
    "-fno-short-wchar"  # Shouldn't matter to us, but doesn't hurt.

    # Target architecture/machine.
    "-target"
    "amdgcn-amd-amdhsa"
    "-march=${_RULE_GPU_ARCH}"
    "-fgpu-rdc"  # NOTE: may not be required for all targets.

    # Optimized.
    "-O3"
    "-fno-ident"
    "-fvisibility=hidden"

    # Object file only in bitcode format.
    "-c"
    "-emit-llvm"
  )

  add_custom_command(
    OUTPUT
      "${_OUT}"
    COMMAND
      "${IREE_CLANG_BINARY}"
      ${_COPTS}
      "-I" "${IREE_SOURCE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC}"
      "-o" "${_OUT}"
    DEPENDS
      "${IREE_CLANG_BINARY}"
      "${_RULE_SRCS}"
    COMMENT
      "Compiling ${_SRC} to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

# iree_link_bitcode()
#
# Builds an LLVM bitcode library from an input file via clang
#
# Parameters:
# NAME: Name of target (see Note).
# SRCS: Source files to pass to clang.
# OUT: Output file name (defaults to NAME.bc).
function(iree_link_bitcode)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT"
    "SRCS"
    ${ARGN}
  )

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  set(_BITCODE_FILES "${_RULE_SRCS}")

  add_custom_command(
    OUTPUT
      ${_OUT}
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
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()
