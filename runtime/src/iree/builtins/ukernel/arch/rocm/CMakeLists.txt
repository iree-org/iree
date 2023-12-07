iree_add_all_subdirs()

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
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  set(_ROCM_ARCH "${_RULE_ROCM_ARCH}")

  set(_COPTS
    "-x" "hip"

    # Target architecture.
    "--offload-arch=${_ROCM_ARCH}"

    # Suppress warnings about missing path to rocm lib,
    # and benign warning about ROCM version.
    "-nogpulib"
    "--offload-device-only"

    # https://github.com/llvm/llvm-project/issues/54609
    "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

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
    message("BITCODE FILE IZ: ${_BITCODE_FILE}")
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
  message("OUT IZ ${_OUT}")

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  message("FOUND YOU: ${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

iree_rocm_bitcode_library(
  NAME
    rocm_argmax_ukernel
  ROCM_ARCH
    gfx940
  SRCS
    "argmax_ukernel.c"
)
