# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_import_tflite_model()
#
# Generates MLIR file from a TFLite model file. The generated target will be
# also added to the iree-benchmark-import-models.
#
# Parameters:
#   TARGET_NAME: The target name to be created for this module.
#   SOURCE: Source TF model direcotry
#   IMPORT_FLAGS: Flags to include in the import command.
#   OUTPUT_MLIR_FILE: The path to output the generated MLIR file.
function(iree_import_tflite_model)
  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "TARGET_NAME;SOURCE;OUTPUT_MLIR_FILE"
    "IMPORT_FLAGS"
  )
  iree_validate_required_arguments(
    _RULE
    "TARGET_NAME;SOURCE;OUTPUT_MLIR_FILE"
    ""
  )

  if(NOT IREE_IMPORT_TFLITE_PATH)
    message(SEND_ERROR "Benchmarks of ${_RULE_SOURCE} require"
                      " that iree-import-tflite be available "
                      " (either on PATH or via IREE_IMPORT_TFLITE_PATH). "
                      " Install from a release with "
                      " `python -m pip install iree-tools-tflite -f https://iree.dev/pip-release-links.html`")
  endif()

  if(NOT TARGET "${_RULE_TARGET_NAME}")
    cmake_path(GET _RULE_SOURCE FILENAME _MODEL_BASENAME)
    add_custom_command(
      OUTPUT "${_RULE_OUTPUT_MLIR_FILE}"
      COMMAND
        "${IREE_IMPORT_TFLITE_PATH}"
        "${_RULE_SOURCE}"
        "-o=${_RULE_OUTPUT_MLIR_FILE}"
        ${_RULE_IMPORT_FLAGS}
      DEPENDS
        "${_RULE_SOURCE}"
      COMMENT "Importing TFLite model ${_MODEL_BASENAME}"
      VERBATIM
    )
    add_custom_target("${_RULE_TARGET_NAME}"
      DEPENDS
        "${_RULE_OUTPUT_MLIR_FILE}"
      COMMENT
        "Importing ${_MODEL_BASENAME} into MLIR"
    )
  endif()
endfunction()

# iree_import_tf_model()
#
# Generates MLIR file from a TensorFlow SavedModel. The generated target will
# be also added to the iree-benchmark-import-models.
#
# Parameters:
#   TARGET_NAME: The target name to be created for this module.
#   SOURCE: Source TF model direcotry
#   IMPORT_FLAGS: Flags to include in the import command.
#   OUTPUT_MLIR_FILE: The path to output the generated MLIR file.
function(iree_import_tf_model)
  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "TARGET_NAME;SOURCE;OUTPUT_MLIR_FILE"
    "IMPORT_FLAGS"
  )
  iree_validate_required_arguments(
    _RULE
    "TARGET_NAME;SOURCE;OUTPUT_MLIR_FILE"
    ""
  )

  if(NOT IREE_IMPORT_TF_PATH)
    message(SEND_ERROR "Benchmarks of ${_RULE_SOURCE} require"
                      " that iree-import-tf be available "
                      " (either on PATH or via IREE_IMPORT_TF_PATH). "
                      " Install from a release with "
                      " `python -m pip install iree-tools-tf -f https://iree.dev/pip-release-links.html`")
  endif()

  if(NOT TARGET "${_RULE_TARGET_NAME}")
    cmake_path(GET _RULE_SOURCE FILENAME _MODEL_BASENAME)
    add_custom_command(
      OUTPUT "${_RULE_OUTPUT_MLIR_FILE}"
      COMMAND
        "${IREE_IMPORT_TF_PATH}"
        "${_RULE_SOURCE}"
        "-o=${_RULE_OUTPUT_MLIR_FILE}"
        ${_RULE_IMPORT_FLAGS}
      DEPENDS
        "${_RULE_SOURCE}"
      COMMENT "Importing TF model ${_MODEL_BASENAME}"
      VERBATIM
    )
    add_custom_target("${_RULE_TARGET_NAME}"
      DEPENDS
        "${_RULE_OUTPUT_MLIR_FILE}"
      COMMENT
        "Importing ${_MODEL_BASENAME} into MLIR"
    )
  endif()
endfunction()
