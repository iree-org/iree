# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_fetch_artifact()
#
# Download file from URL. NEVER Use this rule to download from untrusted
# sources, it doesn't unpack the file safely.
#
# Parameters:
# NAME: Name of target (see Note).
# SOURCE_URL: Source URL to donwload the file.
# OUTPUT: Path to the output file or directory to unpack.
# UNPACK: When added, it will try to unpack the archive if supported.
#
# Note:
# By default, it will create a target named ${_PACKAGE_NAME}_${_RULE_NAME}.
function(iree_fetch_artifact)
  cmake_parse_arguments(
    _RULE
    "UNPACK"
    "NAME;SOURCE_URL;OUTPUT"
    ""
    ${ARGN}
  )

  set(_ARGS "${IREE_ROOT_DIR}/build_tools/scripts/download_file.py")
  list(APPEND _ARGS "${_RULE_SOURCE_URL}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_RULE_OUTPUT}")

  if(_RULE_UNPACK)
    list(APPEND _ARGS "--unpack")
  endif()

  # TODO: CMake built-in file command can replace the python script. But python
  # script also provides streaming unpack (doesn't use double space when
  # unpacking). Need to evaluate if we want to replace.
  add_custom_command(
    OUTPUT "${_RULE_OUTPUT}"
    COMMAND
      "${Python3_EXECUTABLE}"
      ${_ARGS}
    DEPENDS
      "${IREE_ROOT_DIR}/build_tools/scripts/download_file.py"
    COMMENT "Downloading ${_RULE_SOURCE_URL}"
  )

  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS
      "${_RULE_OUTPUT}"
  )
endfunction()
