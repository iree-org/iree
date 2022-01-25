# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_benchmark_suite()
#
# Generates benchmark suites for MLIR input modules. The generated artifacts
# will be placed in the "<binary-root>/benchmark_suites/<category>" directory,
# where "<category>" is the name of the immediate directory containing the
# CMakeLists.txt. The generated artifacts are expected to be executed with
# `iree-benchmark-module`.
#
# Parameters:
#   MODULES: A list for model specification. Due to CMake's lack of data
#       structures, each module is represented as a list suitable to be parsed
#       by cmake_parse_arguments:
#       - NAME: The input module's name.
#       - TAGS: comma-separated tags for the input module.
#       - SOURCE: The input file for the input module. Supported formats are
#           MLIR files in the IREE input format (which should have a .mlir
#           extension) or TFLite flatbuffers (with a .tflite extension). In
#           addition to permitting a source file, this can be a URL ("http://"
#           or "https://") from which to download the file. This URL should
#           point to a file in one of the appropriate input formats, optionally
#           compressed in the gzip format, in which case it should have a
#           trailing ".gz" extension in addition to other extensions.
#       - ENTRY_FUNCTION: The entry function name for the input module.
#       - FUNCTION_INPUT: A list of comma-separated entry function inputs for
#           the input module.
#   BENCHMARK_MODES: A list strings, where ech one of them is a comma-
#       separated list of benchmark mode tags.
#   TARGET_BACKEND: The compiler target backend.
#   TARGET_ARCHITECTURE: The detailed target backend's architecture.
#   TRANSLATION_FLAGS: A list of command-line options and their values to
#       pass to the IREE translation tool for artifact generation.
#   DRIVER: The runtime driver.
#   RUNTIME_FLAGS: A list of command-line options and their values to pass
#       to the IREE runtime during benchmark exectuion.
#
# The above parameters largely fall into two categories: 1) for specifying
# the MLIR input module and its metadata, 2) for specifying the translation/
# runtime configuration.
#
# 1)
#
# The MODULES provide information about the input module and its metadata. For
# example, we can generate modules with idential names from different sources
# (TensorFlow, TFLite, PyTorch, etc.), and we can transform the same input
# module differently for benchmarking different aspects like fp32 vs fp16.
#
# 2)
#
# TARGET_BACKEND and TRANSLATION_FLAGS control how the input module will be
# converted into the final IREE deployable module format. DRIVER and
# RUNTIME_FLAGS specify how the module will be executed. BENCHMARK_MODES
# can be used to give descriptions of the translation/runtime configuration
# (e.g., full-inference vs. kernel-execution) and specify more contextual
# requirements (e.g., big-core vs. little-core).
#
function(iree_benchmark_suite)
  if(NOT IREE_BUILD_BENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "DRIVER;TARGET_BACKEND;TARGET_ARCHITECTURE"
    "BENCHMARK_MODES;BENCHMARK_TOOL;MODULES;TRANSLATION_FLAGS;RUNTIME_FLAGS"
  )

  iree_validate_required_arguments(
    _RULE
    "DRIVER;TARGET_BACKEND;TARGET_ARCHITECTURE"
    "BENCHMARK_MODES;BENCHMARK_TOOL;MODULES"
  )

  iree_package_name(PACKAGE_NAME)

  foreach(_MODULE IN LISTS _RULE_MODULES)
    cmake_parse_arguments(
      _MODULE
      ""
      "NAME;TAGS;SOURCE;ENTRY_FUNCTION;FUNCTION_INPUTS"
      ""
      ${_MODULE}
    )
    iree_validate_required_arguments(
      _MODULE
      "NAME;TAGS;SOURCE;ENTRY_FUNCTION;FUNCTION_INPUTS"
      ""
    )

    get_filename_component(_CATEGORY "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    set(_ROOT_ARTIFACTS_DIR "${IREE_BINARY_DIR}/benchmark_suites/${_CATEGORY}")
    set(_VMFB_ARTIFACTS_DIR "${_ROOT_ARTIFACTS_DIR}/vmfb")

    # The name of any custom target that drives creation of the final source
    # MLIR file. Depending on the format of the source, this will get updated.
    set(_MODULE_SOURCE_TARGET "")

    # If the source file is from the web, create a custom command to download
    # it and wrap that with a custom target so later we can use for dependency.
    if("${_MODULE_SOURCE}" MATCHES "^https?://")
      set(_SOURCE_URL "${_MODULE_SOURCE}")
      # Update the source file to the downloaded-to place.
      string(REPLACE "/" ";" _SOURCE_URL_SEGMENTS "${_SOURCE_URL}")
      list(POP_BACK _SOURCE_URL_SEGMENTS _LAST_URL_SEGMENT)
      set(_DOWNLOAD_TARGET "${PACKAGE_NAME}_iree-download-benchmark-source-${_LAST_URL_SEGMENT}")

      # Strip off gzip suffix if present (downloader unzips if necessary)
      string(REGEX REPLACE "\.gz$" "" _SOURCE_FILE_BASENAME "${_LAST_URL_SEGMENT}")
      set(_MODULE_SOURCE "${_ROOT_ARTIFACTS_DIR}/${_SOURCE_FILE_BASENAME}")
      if (NOT TARGET "${_DOWNLOAD_TARGET}")
        add_custom_command(
          OUTPUT "${_MODULE_SOURCE}"
          COMMAND
            "${Python3_EXECUTABLE}" "${IREE_ROOT_DIR}/scripts/download_file.py"
            "${_SOURCE_URL}" -o "${_MODULE_SOURCE}"
          DEPENDS
            "${IREE_ROOT_DIR}/scripts/download_file.py"
          COMMENT "Downloading ${_SOURCE_URL}"
        )
        add_custom_target("${_DOWNLOAD_TARGET}"
          DEPENDS "${_MODULE_SOURCE}"
        )
      endif()
      set(_MODULE_SOURCE_TARGET "${_DOWNLOAD_TARGET}")
    endif()

    # If the source is a TFLite file, import it.
    if("${_MODULE_SOURCE}" MATCHES "\.tflite$")
      if (NOT IREE_IMPORT_TFLITE_PATH)
        message(SEND_ERROR "Benchmarks of ${_MODULE_SOURCE} require"
                          " that iree-import-tflite be available "
                          " (either on PATH or via IREE_IMPORT_TFLITE_PATH)")
      endif()
      set(_TFLITE_FILE "${_MODULE_SOURCE}")
      set(_MODULE_SOURCE "${_TFLITE_FILE}.mlir")
      get_filename_component(_TFLITE_FILE_BASENAME "${_TFLITE_FILE}" NAME)
      set(_TFLITE_IMPORT_TARGET "${PACKAGE_NAME}_iree-import-tflite-${_TFLITE_FILE_BASENAME}")
      if (NOT TARGET "${_TFLITE_IMPORT_TARGET}")
        add_custom_command(
          OUTPUT "${_MODULE_SOURCE}"
          COMMAND
            "${IREE_IMPORT_TFLITE_PATH}"
            "${_TFLITE_FILE}"
            "-o=${_MODULE_SOURCE}"
          DEPENDS
            "${_TFLITE_FILE}"
          COMMENT "Importing TFLite file ${_TFLITE_FILE_BASENAME}"
        )
        add_custom_target("${_TFLITE_IMPORT_TARGET}"
          DEPENDS
            "${_MODULE_SOURCE}"
          COMMENT
            "Importing ${_TFLITE_FILE_BASENAME} into MLIR"
        )
      endif()
      set(_MODULE_SOURCE_TARGET "${_TFLITE_IMPORT_TARGET}")
    endif()

    # Next create the command and target for compiling the input module into
    # IREE deployable format for each benchmark mode.
    string(JOIN "-" _MODULE_DIR_NAME "${_MODULE_NAME}" "${_MODULE_TAGS}")
    foreach (_BENCHMARK_MODE IN LISTS _RULE_BENCHMARK_MODES)
      set(_BENCHMARK_DIR_NAME
          "iree-${_RULE_DRIVER}__${_RULE_TARGET_ARCHITECTURE}__${_BENCHMARK_MODE}")

      # A list of name segments for composing unique CMake target names.
      set(_COMMON_NAME_SEGMENTS "${_MODULE_NAME}")
      string(REPLACE "," "-" _TAGS "${_MODULE_TAGS}")
      string(REPLACE "," "-" _MODE "${_BENCHMARK_MODE}")
      list(APPEND _COMMON_NAME_SEGMENTS
            "${_TAGS}" "${_MODE}" "${_RULE_TARGET_BACKEND}"
            "${_RULE_TARGET_ARCHITECTURE}")

      # The full list of translation flags.
      set(_TRANSLATION_ARGS "--iree-mlir-to-vm-bytecode-module")
      list(APPEND _TRANSLATION_ARGS "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}")
      list(SORT _RULE_TRANSLATION_FLAGS)
      list(APPEND _TRANSLATION_ARGS ${_RULE_TRANSLATION_FLAGS})

      # Get a unique identifier for this IREE module file by hashing the command
      # line flags and input file. We will also use this for the CMake target.
      # Note that this is NOT A SECURE HASHING ALGORITHM. We just want
      # uniqueness and MD5 is fast. If that changes, switch to something much
      # better (like SHA256).
      string(MD5 _VMFB_HASH "${_TRANSLATION_ARGS};${_MODULE_SOURCE}")
      get_filename_component(_MODULE_SOURCE_BASENAME "${_MODULE_SOURCE}" NAME)
      set(_VMFB_FILE "${_VMFB_ARTIFACTS_DIR}/${_MODULE_SOURCE_BASENAME}-${_VMFB_HASH}.vmfb")

      # Register the target once and share across all benchmarks having the same
      # MLIR source and translation flags.
      set(
        _TRANSLATION_TARGET_NAME
        "${PACKAGE_NAME}_iree-generate-benchmark-artifact-${_MODULE_SOURCE_BASENAME}-${_VMFB_HASH}"
      )
      if(NOT TARGET "${_TRANSLATION_TARGET_NAME}")
        add_custom_command(
          OUTPUT "${_VMFB_FILE}"
          COMMAND
            "$<TARGET_FILE:iree::tools::iree-translate>"
              ${_TRANSLATION_ARGS}
              "--mlir-print-op-on-diagnostic=false"
              "${_MODULE_SOURCE}"
              -o "${_VMFB_FILE}"
          WORKING_DIRECTORY "${_VMFB_ARTIFACTS_DIR}"
          DEPENDS
            iree::tools::iree-translate
            "${_MODULE_SOURCE_TARGET}"
            COMMENT "Generating VMFB for ${_COMMON_NAME_SEGMENTS}"
        )

        add_custom_target("${_TRANSLATION_TARGET_NAME}"
          DEPENDS "${_VMFB_FILE}"
        )

        # Mark dependency so that we have one target to drive them all.
        add_dependencies(iree-benchmark-suites "${_TRANSLATION_TARGET_NAME}")
      endif(NOT TARGET "${_TRANSLATION_TARGET_NAME}")

      # Add a friendly target name to drive this benchmark and any others that
      # share the same easily-describable properties.
      set(_FRIENDLY_TARGET_NAME_LIST "iree-generate-benchmark-artifact")
      list(APPEND _FRIENDLY_TARGET_NAME_LIST ${_COMMON_NAME_SEGMENTS})
      list(JOIN _FRIENDLY_TARGET_NAME_LIST "__" _FRIENDLY_TARGET_NAME)

      if (NOT TARGET "${_FRIENDLY_TARGET_NAME}")
        add_custom_target("${_FRIENDLY_TARGET_NAME}")
      endif()
      add_dependencies("${_FRIENDLY_TARGET_NAME}" "${_TRANSLATION_TARGET_NAME}")

      set(_RUN_SPEC_DIR "${_ROOT_ARTIFACTS_DIR}/${_MODULE_DIR_NAME}/${_BENCHMARK_DIR_NAME}")

      # Create the command and target for the flagfile spec used to execute
      # the generated artifacts.
      set(_FLAG_FILE "${_RUN_SPEC_DIR}/flagfile")
      set(_ADDITIONAL_ARGS_CL "--additional_args=\"${_RULE_RUNTIME_FLAGS}\"")
      file(RELATIVE_PATH _MODULE_FILE_FLAG "${_RUN_SPEC_DIR}" "${_VMFB_FILE}")
      add_custom_command(
        OUTPUT "${_FLAG_FILE}"
        COMMAND
          "${Python3_EXECUTABLE}" "${IREE_ROOT_DIR}/scripts/generate_flagfile.py"
            --module_file="${_MODULE_FILE_FLAG}"
            --driver=${_RULE_DRIVER}
            --entry_function=${_MODULE_ENTRY_FUNCTION}
            --function_inputs=${_MODULE_FUNCTION_INPUTS}
            "${_ADDITIONAL_ARGS_CL}"
            -o "${_FLAG_FILE}"
        DEPENDS
          "${IREE_ROOT_DIR}/scripts/generate_flagfile.py"
        WORKING_DIRECTORY "${_RUN_SPEC_DIR}"
        COMMENT "Generating ${_FLAG_FILE}"
      )

      set(_FLAGFILE_GEN_TARGET_NAME_LIST "iree-generate-benchmark-flagfile")
      list(APPEND _FLAGFILE_GEN_TARGET_NAME_LIST ${_COMMON_NAME_SEGMENTS})
      list(JOIN _FLAGFILE_GEN_TARGET_NAME_LIST "__" _FLAGFILE_GEN_TARGET_NAME)
      set(_FLAGFILE_GEN_TARGET_NAME "${PACKAGE_NAME}_${_FLAGFILE_GEN_TARGET_NAME}")

      add_custom_target("${_FLAGFILE_GEN_TARGET_NAME}"
        DEPENDS "${_FLAG_FILE}"
      )

      # Create the command and target for the toolfile spec used to execute
      # the generated artifacts.
      set(_TOOL_FILE "${_RUN_SPEC_DIR}/tool")
      add_custom_command(
        OUTPUT "${_TOOL_FILE}"
        COMMAND ${CMAKE_COMMAND} -E echo ${_RULE_BENCHMARK_TOOL} > "${_TOOL_FILE}"
        WORKING_DIRECTORY "${_RUN_SPEC_DIR}"
        COMMENT "Generating ${_TOOL_FILE}"
      )

      set(_TOOLFILE_GEN_TARGET_NAME_LIST "iree-generate-benchmark-toolfile")
      list(APPEND _TOOLFILE_GEN_TARGET_NAME_LIST ${_COMMON_NAME_SEGMENTS})
      list(JOIN _TOOLFILE_GEN_TARGET_NAME_LIST "__" _TOOLFILE_GEN_TARGET_NAME)
      add_custom_target("${_TOOLFILE_GEN_TARGET_NAME}"
        DEPENDS "${_TOOL_FILE}"
      )

      # Mark dependency so that we have one target to drive them all.
      add_dependencies(iree-benchmark-suites
        "${_FLAGFILE_GEN_TARGET_NAME}"
        "${_TOOLFILE_GEN_TARGET_NAME}"
      )
    endforeach(_BENCHMARK_MODE IN LISTS _RULE_BENCHMARK_MODES)

  endforeach(_MODULE IN LISTS _RULE_MODULES)
endfunction(iree_benchmark_suite)
