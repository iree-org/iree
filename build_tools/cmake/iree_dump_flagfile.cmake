# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_dump_flagfile()
#
# Generate the flagfile from a list of flags.
#
# Parameters:
#   TARGET_NAME: The target name to be created for this flagfile.
#   OUTPUT: The path to output the generated flagfile.
#   FLAGS: The flags to be outputed.
#   WORKING_DIRECTORY: The working directory.
#   COMMENT: The comment to be displayed when generating.
function(iree_dump_flagfile)
  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "TARGET_NAME;OUTPUT;WORKING_DIRECTORY;COMMENT"
    "FLAGS"
  )
  iree_validate_required_arguments(
    _RULE
    "TARGET_NAME;OUTPUT"
    "FLAGS"
  )

  # Generate the flagfile with python command. We can't use "file" because
  # it can't be part of a target's dependency and generated lazily. And
  # "cmake -E echo" doesn't work with newlines.
  add_custom_command(
    OUTPUT "${_RULE_OUTPUT}"
    COMMAND
      "${Python3_EXECUTABLE}" "${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
        --output "${_RULE_OUTPUT}"
        -- ${_RULE_FLAGS}
    DEPENDS
      "${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
    WORKING_DIRECTORY "${_RULE_WORKING_DIRECTORY}"
    COMMENT "${_RULE_COMMENT}"
  )
  add_custom_target("${_RULE_TARGET_NAME}"
    DEPENDS "${_RULE_OUTPUT}"
  )
endfunction()

