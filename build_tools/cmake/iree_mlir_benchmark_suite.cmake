# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_check_lists_have_same_size()
#
# Note that the caller should pass in the list variables themselves to
# LIST1 and LIST2, not the list variables' values.
function(iree_check_lists_have_same_size LIST1 LIST2)
  list(LENGTH "${LIST1}" _LIST1_COUNT)
  list(LENGTH "${LIST2}" _LIST2_COUNT)
  if(NOT _LIST1_COUNT EQUAL _LIST2_COUNT)
    message(SEND_ERROR "${LIST1} count ${_LIST1_COUNT} does not "
                       "match ${LIST2} count ${_LIST2_COUNT}"
    )
  endif()
endfunction()

# iree_mlir_benchmark_suite()
#
# Generates benchmark suites for MLIR input modules. The generated artifacts
# will be executed with `iree-benchmark-module`.
#
# Parameters:
#   MODULE_NAMES: A list of input module names.
#   MODULE_TAGS: A list of tags for each input module.
#   MODULE_SOURCES: The initial generating source for each input module.
#   MLIR_SOURCES: The input file for each input module. It can be a file in
#       checked in the repository; it can also be a URL for downloading from.
#       the web. When it's a URL, the file should be a a direct .mlir file
#       or a tarball containing a .mlir file; for both cases, the .mlir file
#       should have a name matching the one in MODULE_NAMES.
#   ENTRY_FUNCTIONS: The entry function name for each input module.
#   FUNCTION_INPUTS: A list of entry function inputs for each input module.
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
# MODULE_NAMES, MODULE_TAGS, MODULE_SOURCES, MLIR_SOURCES, ENTRY_FUNCTIONS,
# and FUNCTION_INPUTS together provide good flexiblity for specifying the MLIR
# input module and its metadata. For example, we can generate modules with
# idential name from different sources (TensorFlow, TFLite, PyTorch, etc.),
# and we can transform the same input module differently for benchmarking
# different aspects like fp32 vs fp16.
#
# Note that the above parameters are all lists and they should have the name
# number of elements. This enables us to use the same CMake function call to
# generate benchmarks for many models and share the specification of
# translation/runtime configurations.
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
function(iree_mlir_benchmark_suite)
  if(NOT IREE_BUILD_BENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "DRIVER;TARGET_BACKEND;TARGET_ARCHITECTURE"
    "BENCHMARK_MODES;ENTRY_FUNCTIONS;FUNCTION_INPUTS;MLIR_SOURCES;MODULE_NAMES;MODULE_SOURCES;MODULE_TAGS;TRANSLATION_FLAGS;RUNTIME_FLAGS"
  )

  iree_check_lists_have_same_size(_RULE_MODULE_NAMES _RULE_MODULE_TAGS)
  iree_check_lists_have_same_size(_RULE_MODULE_NAMES _RULE_MODULE_SOURCES)
  iree_check_lists_have_same_size(_RULE_MODULE_NAMES _RULE_MLIR_SOURCES)
  iree_check_lists_have_same_size(_RULE_MODULE_NAMES _RULE_ENTRY_FUNCTIONS)
  iree_check_lists_have_same_size(_RULE_MODULE_NAMES _RULE_FUNCTION_INPUTS)

  # Loop over all modules and their sources to create targets.
  math(EXPR _MAX_INDEX "${_MODULE_NAMES_COUNT} - 1")
  foreach(_INDEX RANGE 0 "${_MAX_INDEX}")
    # Generate all benchmarks to the root build directory. This helps for
    # discovering them and execute them on devices.
    list(GET _RULE_MODULE_SOURCES ${_INDEX} _MODULE_SOURCE)
    set(_ROOT_ARTIFACTS_DIR "${IREE_BINARY_DIR}/benchmark_suites/${_MODULE_SOURCE}")

    list(GET _RULE_MODULE_NAMES ${_INDEX} _MODULE_NAME)
    list(GET _RULE_MODULE_TAGS ${_INDEX} _MODULE_TAGS)
    list(GET _RULE_MLIR_SOURCES ${_INDEX} _MLIR_SOURCE)
    list(GET _RULE_ENTRY_FUNCTIONS ${_INDEX} _ENTRY_FUNCTION)
    list(GET _RULE_FUNCTION_INPUTS ${_INDEX} _FUNCTION_INPUTS)

    # The source file used to generate benchmark artifacts.
    set(_SOURCE_FILE "${_MLIR_SOURCE}")
    # The CMake target's name if we need to download from the web.
    set(_DOWNLOAD_TARGET_NAME "")

    # If the source file is from the web, create a custom command to download it.
    # And wrap that with a custom target so later we can use for dependency.
    #
    # Note: We actually should not do this; instead, we should directly compile
    # from the initial source (i.e., TensorFlow Python models). But that is
    # tangled with the pending Python testing infrastructure revamp so we'd prefer
    # to not do that right now.
    if("${_MLIR_SOURCE}" MATCHES "^https?://")
      # Update the source file to the downloaded-to place.
      string(REPLACE "/" ";" _SOURCE_URL_SEGMENTS "${_MLIR_SOURCE}")
      # TODO: we can do `list(POP_BACK _SOURCE_URL_SEGMENTS _LAST_URL_SEGMENT)`
      # after migrating to CMake 3.15.
      list(LENGTH _SOURCE_URL_SEGMENTS _URL_SEGMENT_COUNT)
      math(EXPR _SEGMENT_LAST_INDEX "${_URL_SEGMENT_COUNT} - 1")
      list(GET _SOURCE_URL_SEGMENTS ${_SEGMENT_LAST_INDEX} _LAST_URL_SEGMENT)
      set(_DOWNLOAD_TARGET_NAME "iree-download-benchmark-source-${_LAST_URL_SEGMENT}")

      string(REPLACE "tar.gz" "mlir" _FILE_NAME "${_LAST_URL_SEGMENT}")
      set(_SOURCE_FILE "${_ROOT_ARTIFACTS_DIR}/${_MODULE_NAME}.mlir")

      if (NOT TARGET "${_DOWNLOAD_TARGET_NAME}")
        add_custom_command(
          OUTPUT "${_SOURCE_FILE}"
          COMMAND
            "${Python3_EXECUTABLE}" "${IREE_ROOT_DIR}/scripts/download_file.py"
            "${_MLIR_SOURCE}" -o "${_ROOT_ARTIFACTS_DIR}"
          DEPENDS
            "${IREE_ROOT_DIR}/scripts/download_file.py"
          COMMENT "Downloading ${_MLIR_SOURCE}"
        )
        add_custom_target("${_DOWNLOAD_TARGET_NAME}"
          DEPENDS "${_SOURCE_FILE}"
        )
      endif()
    endif()

    # Next create the command and target for compiling the input module into
    # IREE deployable format for each benchmark mode.
    string(JOIN "-" _MODULE_DIR_NAME "${_MODULE_NAME}" "${_MODULE_TAGS}")
    foreach (_BENCHMARK_MODE IN LISTS _RULE_BENCHMARK_MODES)
      set(_BENCHMARK_DIR_NAME
          "iree-${_RULE_DRIVER}__${_RULE_TARGET_ARCHITECTURE}__${_BENCHMARK_MODE}")
      set(_ARTIFACTS_DIR "${_ROOT_ARTIFACTS_DIR}/${_MODULE_DIR_NAME}/${_BENCHMARK_DIR_NAME}")

      set(_TRANSLATION_ARGS "--iree-mlir-to-vm-bytecode-module")
      list(APPEND _TRANSLATION_ARGS "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}")
      list(APPEND _TRANSLATION_ARGS ${_RULE_TRANSLATION_FLAGS})

      set(_VMFB_FILE "${_ARTIFACTS_DIR}/compiled.vmfb")
      add_custom_command(
        OUTPUT "${_VMFB_FILE}"
        COMMAND
          "$<TARGET_FILE:iree_tools_iree-translate>"
            ${_TRANSLATION_ARGS}
            "${_SOURCE_FILE}"
            -o "${_VMFB_FILE}"
        WORKING_DIRECTORY "${_ARTIFACTS_DIR}"
        DEPENDS
          iree_tools_iree-translate
          "${_DOWNLOAD_TARGET_NAME}"
        COMMENT "Generating ${_VMFB_FILE}"
      )

      set(_COMMON_NAME_SEGMENTS "${_MODULE_NAME}")
      string(REPLACE "," "-" _TAGS "${_MODULE_TAGS}")
      string(REPLACE "," "-" _MODE "${_BENCHMARK_MODE}")
      list(APPEND _COMMON_NAME_SEGMENTS
           "${_TAGS}" "${_MODE}" "${_RULE_TARGET_BACKEND}"
           "${_RULE_TARGET_ARCHITECTURE}")

      # Construct the benchmark artifact generation target name, which is the module
      # name, followed by benchmark mode, target backend, and configuration.
      set(_TRANSLATION_TARGET_NAME_LIST "iree-generate-benchmark-artifact")
      list(APPEND _TRANSLATION_TARGET_NAME_LIST ${_COMMON_NAME_SEGMENTS})
      list(JOIN _TRANSLATION_TARGET_NAME_LIST "__" _TRANSLATION_TARGET_NAME)

      add_custom_target("${_TRANSLATION_TARGET_NAME}"
        DEPENDS "${_VMFB_FILE}"
      )

      # Mark dependency so that we have one target to drive them all.
      add_dependencies(iree-benchmark-suites "${_TRANSLATION_TARGET_NAME}")

      # Finally create the command and target for the flagfile used to execute the
      # generated artifacts.
      set(_FLAG_FILE "${_ARTIFACTS_DIR}/flagfile")
      set(_ADDITIONAL_ARGS_CL "--additional_args=\"${_RULE_RUNTIME_FLAGS}\"")
      add_custom_command(
        OUTPUT "${_FLAG_FILE}"
        COMMAND
          "${Python3_EXECUTABLE}" "${IREE_ROOT_DIR}/scripts/generate_flagfile.py"
            --module_file=compiled.vmfb
            --driver=${_RULE_DRIVER}
            --entry_function=${_ENTRY_FUNCTION}
            --function_inputs=${_FUNCTION_INPUTS}
            "${_ADDITIONAL_ARGS_CL}"
            -o "${_FLAG_FILE}"
        DEPENDS
          "${IREE_ROOT_DIR}/scripts/generate_flagfile.py"
        WORKING_DIRECTORY "${_ARTIFACTS_DIR}"
        COMMENT "Generating ${_FLAG_FILE}"
      )

      set(_FLAGFILE_GEN_TARGET_NAME_LIST "iree-generate-benchmark-flagfile")
      list(APPEND _FLAGFILE_GEN_TARGET_NAME_LIST ${_COMMON_NAME_SEGMENTS})
      list(JOIN _FLAGFILE_GEN_TARGET_NAME_LIST "__" _FLAGFILE_GEN_TARGET_NAME)

      add_custom_target("${_FLAGFILE_GEN_TARGET_NAME}"
        DEPENDS "${_FLAG_FILE}"
      )

      # Mark dependency so that we have one target to drive them all.
      add_dependencies(iree-benchmark-suites "${_FLAGFILE_GEN_TARGET_NAME}")
    endforeach(_BENCHMARK_MODE IN LISTS _RULE_BENCHMARK_MODES)
  endforeach(_INDEX RANGE 0 "${_MAX_INDEX}")
endfunction()
