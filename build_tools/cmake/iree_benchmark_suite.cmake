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
                      " `python -m pip install iree-tools-tflite -f https://openxla.github.io/iree/pip-release-links.html`")
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
    )
    add_custom_target("${_RULE_TARGET_NAME}"
      DEPENDS
        "${_RULE_OUTPUT_MLIR_FILE}"
      COMMENT
        "Importing ${_MODEL_BASENAME} into MLIR"
    )
    add_dependencies(iree-benchmark-import-models "${_RULE_TARGET_NAME}")
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
                      " `python -m pip install iree-tools-tf -f https://openxla.github.io/iree/pip-release-links.html`")
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
    )
    add_custom_target("${_RULE_TARGET_NAME}"
      DEPENDS
        "${_RULE_OUTPUT_MLIR_FILE}"
      COMMENT
        "Importing ${_MODEL_BASENAME} into MLIR"
    )
    add_dependencies(iree-benchmark-import-models "${_RULE_TARGET_NAME}")
  endif()
endfunction()

# iree_benchmark_suite()
#
# Generates benchmark suites for MLIR input modules. The generated artifacts
# will be placed in the "<binary-root>/benchmark_suites/<category>" directory,
# where "<category>" is the name of the immediate directory containing the
# CMakeLists.txt. The generated artifacts are expected to be executed with
# `iree-benchmark-module`.
#
# Parameters:
#   GROUP_NAME: A group name this benchmark will join. Each group has its own
#       CMake's benchmark suite target: "iree-benchmark-suites-<GROUP_NAME>".
#   MODULES: A list for model specification. Due to CMake's lack of data
#       structures, each module is represented as a list suitable to be parsed
#       by cmake_parse_arguments:
#       - NAME: The input module's name.
#       - TAGS: comma-separated tags for the input module.
#       - SOURCE: The input file for the input module. Supported formats are
#           MLIR files in the IREE input format (which should have a .mlir
#           extension) or TFLite FlatBuffers (with a .tflite extension). In
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
#   COMPILATION_FLAGS: A list of command-line options and their values to
#       pass to the IREE compiler tool for artifact generation.
#   CONFIG: Benchmark runner configuration name.
#   DRIVER: The runtime driver.
#   RUNTIME_FLAGS: A list of command-line options and their values to pass
#       to the IREE runtime during benchmark exectuion.
#
# The above parameters largely fall into two categories: 1) for specifying
# the MLIR input module and its metadata, 2) for specifying the compilation/
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
# TARGET_BACKEND and COMPILATION_FLAGS control how the input module will be
# converted into the final IREE deployable module format. DRIVER and
# RUNTIME_FLAGS specify how the module will be executed. BENCHMARK_MODES
# can be used to give descriptions of the compilation/runtime configuration
# (e.g., full-inference vs. kernel-execution) and specify more contextual
# requirements (e.g., big-core vs. little-core).
#
function(iree_benchmark_suite)
  if(NOT IREE_BUILD_LEGACY_BENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "GROUP_NAME;CONFIG;DRIVER;TARGET_BACKEND;TARGET_ARCHITECTURE"
    "BENCHMARK_MODES;BENCHMARK_TOOL;MODULES;COMPILATION_FLAGS;RUNTIME_FLAGS"
  )

  iree_validate_required_arguments(
    _RULE
    "GROUP_NAME;CONFIG;DRIVER;TARGET_BACKEND;TARGET_ARCHITECTURE"
    "BENCHMARK_MODES;BENCHMARK_TOOL;MODULES"
  )

  # Try to check if the compiler supports the TARGET_BACKEND. If
  # IREE_HOST_BIN_DIR is set, we are using a compiler binary, in which
  # case we can't check its supported backends just by looking at this build
  # dir's cmake variables --- we would have to implement a configure-check
  # executing `iree-compile --iree-hal-list-target-backends`.
  if(NOT IREE_HOST_BIN_DIR)
    string(TOUPPER ${_RULE_TARGET_BACKEND} _UPPERCASE_TARGET_BACKEND)
    string(REPLACE "-" "_" _NORMALIZED_TARGET_BACKEND ${_UPPERCASE_TARGET_BACKEND})
    if(NOT IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
      return()
    endif()
  endif()

  iree_package_name(_PACKAGE_NAME)

  # Add the benchmark suite target.
  set(SUITE_SUB_TARGET "iree-benchmark-suites-${_RULE_GROUP_NAME}")
  if(NOT TARGET "${SUITE_SUB_TARGET}")
    add_custom_target("${SUITE_SUB_TARGET}")
  endif()

  foreach(_MODULE IN LISTS _RULE_MODULES)
    cmake_parse_arguments(
      _MODULE
      ""
      "NAME;TAGS;SOURCE;ENTRY_FUNCTION;FUNCTION_INPUTS"
      "IMPORT_FLAGS"
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
    file(MAKE_DIRECTORY ${_VMFB_ARTIFACTS_DIR})

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
      set(_DOWNLOAD_TARGET_NAME "iree-download-benchmark-source-${_LAST_URL_SEGMENT}")

      # Strip off gzip/tar suffix if present (downloader unpacks if necessary)
      string(REGEX REPLACE "(\.gz)|(\.tar\.gz)$" "" _SOURCE_FILE_BASENAME "${_LAST_URL_SEGMENT}")
      set(_MODULE_SOURCE "${_ROOT_ARTIFACTS_DIR}/${_SOURCE_FILE_BASENAME}")
      if(NOT TARGET "${_PACKAGE_NAME}_${_DOWNLOAD_TARGET_NAME}")
        iree_fetch_artifact(
          NAME
            "${_DOWNLOAD_TARGET_NAME}"
          SOURCE_URL
            "${_SOURCE_URL}"
          OUTPUT
            "${_MODULE_SOURCE}"
          UNPACK
        )
      endif()
      set(_MODULE_SOURCE_TARGET "${_PACKAGE_NAME}_${_DOWNLOAD_TARGET_NAME}")
    endif()

    # If the source is a TFLite file, import it.
    if("${_MODULE_SOURCE}" MATCHES "\.tflite$")
      cmake_path(GET _MODULE_SOURCE FILENAME _MODEL_BASENAME)
      set(_MODULE_SOURCE_TARGET "${_PACKAGE_NAME}_iree-import-tf-${_MODEL_BASENAME}")
      iree_import_tflite_model(
        TARGET_NAME "${_MODULE_SOURCE_TARGET}"
        SOURCE "${_MODULE_SOURCE}"
        IMPORT_FLAGS
          "--output-format=mlir-bytecode"
          ${_MODULE_IMPORT_FLAGS}
        OUTPUT_MLIR_FILE "${_MODULE_SOURCE}.mlir"
      )
      set(_MODULE_SOURCE "${_MODULE_SOURCE}.mlir")
    endif()

    # If the source is a TensorFlow SavedModel directory, import it.
    if("${_MODULE_SOURCE}" MATCHES "-tf-model$")
      cmake_path(GET _MODULE_SOURCE FILENAME _MODEL_BASENAME)
      set(_MODULE_SOURCE_TARGET "${_PACKAGE_NAME}_iree-import-tf-${_MODEL_BASENAME}")
      iree_import_tf_model(
        TARGET_NAME "${_MODULE_SOURCE_TARGET}"
        SOURCE "${_MODULE_SOURCE}"
        IMPORT_FLAGS
          "--output-format=mlir-bytecode"
          ${_MODULE_IMPORT_FLAGS}
        OUTPUT_MLIR_FILE "${_MODULE_SOURCE}.mlir"
      )
      set(_MODULE_SOURCE "${_MODULE_SOURCE}.mlir")
    endif()

    # Next create the command and target for compiling the input module into
    # IREE deployable format for each benchmark mode.
    string(JOIN "-" _MODULE_DIR_NAME "${_MODULE_NAME}" "${_MODULE_TAGS}")
    foreach(_BENCHMARK_MODE IN LISTS _RULE_BENCHMARK_MODES)
      set(_BENCHMARK_DIR_NAME
          "${_RULE_CONFIG}__${_RULE_TARGET_ARCHITECTURE}__${_BENCHMARK_MODE}")

      # A list of name segments for composing unique CMake target names.
      set(_COMMON_NAME_SEGMENTS "${_MODULE_NAME}")
      string(REPLACE "," "-" _TAGS "${_MODULE_TAGS}")
      string(REPLACE "," "-" _MODE "${_BENCHMARK_MODE}")
      list(APPEND _COMMON_NAME_SEGMENTS
            "${_TAGS}" "${_MODE}" "${_RULE_TARGET_BACKEND}"
            "${_RULE_TARGET_ARCHITECTURE}")

      # Add a friendly target name to drive this benchmark and any others that
      # share the same easily-describable properties.
      set(_FRIENDLY_TARGET_NAME_LIST "iree-generate-benchmark-artifact")
      list(APPEND _FRIENDLY_TARGET_NAME_LIST ${_COMMON_NAME_SEGMENTS})
      list(JOIN _FRIENDLY_TARGET_NAME_LIST "__" _FRIENDLY_TARGET_NAME)

      # The full list of compilation flags.
      set(_COMPILATION_ARGS "")
      list(APPEND _COMPILATION_ARGS "--mlir-print-op-on-diagnostic=false")
      list(APPEND _COMPILATION_ARGS "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}")
      list(SORT _RULE_COMPILATION_FLAGS)
      list(APPEND _COMPILATION_ARGS ${_RULE_COMPILATION_FLAGS})

      # Get a unique identifier for this IREE module file by hashing the command
      # line flags and input file. We will also use this for the CMake target.
      # Note that this is NOT A SECURE HASHING ALGORITHM. We just want
      # uniqueness and MD5 is fast. If that changes, switch to something much
      # better (like SHA256).
      string(MD5 _VMFB_HASH "${_COMPILATION_ARGS};${_MODULE_SOURCE}")
      get_filename_component(_MODULE_SOURCE_BASENAME "${_MODULE_SOURCE}" NAME)
      set(_MODULE_SOURCE_BASENAME_WITH_HASH "${_MODULE_SOURCE_BASENAME}-${_VMFB_HASH}")
      set(_VMFB_FILE "${_VMFB_ARTIFACTS_DIR}/${_MODULE_SOURCE_BASENAME_WITH_HASH}.vmfb")

      # Register the target once and share across all benchmarks having the same
      # MLIR source and compilation flags.
      set(_COMPILATION_NAME
        "iree-generate-benchmark-artifact-${_MODULE_SOURCE_BASENAME_WITH_HASH}"
      )
      set(_COMPILATION_TARGET_NAME "${_PACKAGE_NAME}_${_COMPILATION_NAME}")
      if(NOT TARGET "${_COMPILATION_TARGET_NAME}")
        iree_bytecode_module(
          NAME
            "${_COMPILATION_NAME}"
          MODULE_FILE_NAME
            "${_VMFB_FILE}"
          SRC
            "${_MODULE_SOURCE}"
          FLAGS
            ${_COMPILATION_ARGS}
          DEPENDS
            "${_MODULE_SOURCE_TARGET}"
          FRIENDLY_NAME
            "${_FRIENDLY_TARGET_NAME}"
        )

        # Mark dependency so that we have one target to drive them all.
        add_dependencies(iree-benchmark-suites "${_COMPILATION_TARGET_NAME}")
        add_dependencies("${SUITE_SUB_TARGET}" "${_COMPILATION_TARGET_NAME}")
      endif()

      set(_COMPILE_STATS_COMPILATION_NAME
        "${_COMPILATION_NAME}-compile-stats"
      )
      set(_COMPILE_STATS_COMPILATION_TARGET_NAME
        "${_PACKAGE_NAME}_${_COMPILE_STATS_COMPILATION_NAME}"
      )
      set(_COMPILE_STATS_VMFB_FILE
        "${_VMFB_ARTIFACTS_DIR}/${_MODULE_SOURCE_BASENAME_WITH_HASH}-compile-stats.vmfb"
      )
      if(IREE_ENABLE_LEGACY_COMPILATION_BENCHMARKS AND NOT TARGET "${_COMPILE_STATS_COMPILATION_TARGET_NAME}")
        iree_bytecode_module(
          NAME
            "${_COMPILE_STATS_COMPILATION_NAME}"
          MODULE_FILE_NAME
            "${_COMPILE_STATS_VMFB_FILE}"
          SRC
            "${_MODULE_SOURCE}"
          FLAGS
            # Enable zip polyglot to provide component sizes.
            "--iree-vm-emit-polyglot-zip=true"
            # Disable debug symbols to provide correct component sizes.
            "--iree-llvmcpu-debug-symbols=false"
            ${_COMPILATION_ARGS}
          DEPENDS
            "${_MODULE_SOURCE_TARGET}"
          FRIENDLY_NAME
            "${_FRIENDLY_TARGET_NAME}"
        )

        # Mark dependency so that we have one target to drive them all.
        add_dependencies(iree-benchmark-suites
          "${_COMPILE_STATS_COMPILATION_TARGET_NAME}"
        )
        add_dependencies("${SUITE_SUB_TARGET}"
          "${_COMPILE_STATS_COMPILATION_TARGET_NAME}"
        )
      endif()

      if(NOT TARGET "${_FRIENDLY_TARGET_NAME}")
        add_custom_target("${_FRIENDLY_TARGET_NAME}")
      endif()
      add_dependencies("${_FRIENDLY_TARGET_NAME}" "${_COMPILATION_TARGET_NAME}")
      if(IREE_ENABLE_LEGACY_COMPILATION_BENCHMARKS)
        add_dependencies("${_FRIENDLY_TARGET_NAME}"
          "${_COMPILE_STATS_COMPILATION_TARGET_NAME}")
      endif()

      set(_RUN_SPEC_DIR "${_ROOT_ARTIFACTS_DIR}/${_MODULE_DIR_NAME}/${_BENCHMARK_DIR_NAME}")
      list(JOIN _COMMON_NAME_SEGMENTS "__" _RUN_SPEC_TARGET_SUFFIX)

      # Create the command and target for the flagfile spec used to execute
      # the generated artifacts.
      set(_FLAG_FILE "${_RUN_SPEC_DIR}/flagfile")
      set(_ADDITIONAL_ARGS "${_RULE_RUNTIME_FLAGS}")
      list(APPEND _ADDITIONAL_ARGS "--device_allocator=caching")
      set(_ADDITIONAL_ARGS_CL "--additional_args=\"${_ADDITIONAL_ARGS}\"")
      file(RELATIVE_PATH _MODULE_FILE_FLAG "${_RUN_SPEC_DIR}" "${_VMFB_FILE}")
      add_custom_command(
        OUTPUT "${_FLAG_FILE}"
        COMMAND
          "${Python3_EXECUTABLE}" "${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
            --module="${_MODULE_FILE_FLAG}"
            --device=${_RULE_DRIVER}
            --function=${_MODULE_ENTRY_FUNCTION}
            --inputs=${_MODULE_FUNCTION_INPUTS}
            "${_ADDITIONAL_ARGS_CL}"
            -o "${_FLAG_FILE}"
        DEPENDS
          "${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
        WORKING_DIRECTORY "${_RUN_SPEC_DIR}"
        COMMENT "Generating ${_FLAG_FILE}"
      )

      set(_FLAGFILE_GEN_TARGET_NAME
        "${_PACKAGE_NAME}_iree-generate-benchmark-flagfile__${_RUN_SPEC_TARGET_SUFFIX}")
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

      set(_TOOLFILE_GEN_TARGET_NAME
        "${_PACKAGE_NAME}_iree-generate-benchmark-toolfile__${_RUN_SPEC_TARGET_SUFFIX}")
      add_custom_target("${_TOOLFILE_GEN_TARGET_NAME}"
        DEPENDS "${_TOOL_FILE}"
      )

      # Generate a flagfile containing command-line options used to compile the
      # generated artifacts.
      set(_COMPILATION_FLAGFILE "${_RUN_SPEC_DIR}/compilation_flagfile")
      # Generate the flagfile with python command. We can't use "file" because
      # it can't be part of a target's dependency and generated lazily. And
      # "cmake -E echo" doesn't work with newlines.
      add_custom_command(
        OUTPUT "${_COMPILATION_FLAGFILE}"
        COMMAND
          "${Python3_EXECUTABLE}" "${IREE_ROOT_DIR}/build_tools/scripts/generate_compilation_flagfile.py"
            --output "${_COMPILATION_FLAGFILE}"
            -- ${_COMPILATION_ARGS}
        WORKING_DIRECTORY "${_RUN_SPEC_DIR}"
        COMMENT "Generating ${_COMPILATION_FLAGFILE}"
      )

      set(_COMPILATION_FLAGFILE_GEN_TARGET_NAME
        "${_PACKAGE_NAME}_iree-generate-benchmark-compilation-flagfile__${_RUN_SPEC_TARGET_SUFFIX}")
      add_custom_target("${_COMPILATION_FLAGFILE_GEN_TARGET_NAME}"
        DEPENDS "${_COMPILATION_FLAGFILE}"
      )

      # Mark dependency so that we have one target to drive them all.
      add_dependencies(iree-benchmark-suites
        "${_COMPILATION_FLAGFILE_GEN_TARGET_NAME}"
        "${_FLAGFILE_GEN_TARGET_NAME}"
        "${_TOOLFILE_GEN_TARGET_NAME}"
      )
      add_dependencies("${SUITE_SUB_TARGET}"
        "${_COMPILATION_FLAGFILE_GEN_TARGET_NAME}"
        "${_FLAGFILE_GEN_TARGET_NAME}"
        "${_TOOLFILE_GEN_TARGET_NAME}"
      )
    endforeach(_BENCHMARK_MODE IN LISTS _RULE_BENCHMARK_MODES)

  endforeach(_MODULE IN LISTS _RULE_MODULES)
endfunction(iree_benchmark_suite)
