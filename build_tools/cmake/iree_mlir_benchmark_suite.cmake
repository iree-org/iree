# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# iree_mlir_benchmark_suite()
#
# Generates benchmark suites for MLIR input modules. The generated artifacts
# will be executed with `iree-benchmark-module`.
#
# Parameters:
#   MODULE_NAMES: A list of MLIR input module names. The list size should be
#       the same as MLIR_SOURCES.
#   MLIR_SOURCES: A list of MLIR input module sources. Each one can be a file
#       checked in the repo; it can also be a URL for downloading form the web.
#       When it's a URL, it can be a tarball which contains a .mlir file
#       with the name as specified in the MODULE_NAMES. The list size should
#       be the same as MODULE_NAMES.
#   BENCHMARK_MODE: The mode of this benchmark suite.
#   TARGET_BACKEND: The compiler target backend.
#   TARGET_ARCH: The detailed target backend's architecture.
#   TRANSLATION_FLAGS: A list of command-line options and their values to
#       pass to the IREE translation tool for artifact generation.
#   DRIVER: The IREE runtime driver.
#   RUNTIME_FLAGS: A list of command-line options and their values to pass
#       to the IREE runtime during benchmark exectuion.

# The full CMake target is a combination of the MODULE_NAME, BENCHMARK_MODE,
# TARGET_BACKEND, and TARGET_ARCH.
#
function(iree_mlir_benchmark_suite)
  if(NOT IREE_BUILD_BENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    ""
    "BENCHMARK_MODE;DRIVER;TARGET_BACKEND;TARGET_ARCH"
    "MLIR_SOURCES;MODULE_NAMES;TRANSLATION_FLAGS;RUNTIME_FLAGS"
  )

  list(LENGTH _RULE_MODULE_NAMES _MODULE_NAMES_COUNT)
  list(LENGTH _RULE_MLIR_SOURCES _MLIR_SOURCES_COUNT)

  if(NOT _MODULE_NAMES_COUNT EQUAL _MLIR_SOURCES_COUNT)
    message(
      SEND_ERROR
        "MODULE_NAMES count ${_MODULE_NAMES_COUNT} does not match MLIR_SOURCES"
        " count ${_MLIR_SOURCES_COUNT}"
    )
  endif()

  # Generate all benchmarks to the root build directory. This helps for
  # discovering them and execute them on devices.
  set(_ROOT_ARTIFACTS_DIR "${CMAKE_BINARY_DIR}/benchmark_suites/mlir_modules")

  # Loop over all modules and their sources to create targets.
  math(EXPR _MAX_INDEX "${_MODULE_NAMES_COUNT} - 1")
  foreach(_INDEX RANGE 0 "${_MAX_INDEX}")
    list(GET _RULE_MODULE_NAMES ${_INDEX} _MODULE_NAME)
    list(GET _RULE_MLIR_SOURCES ${_INDEX} _MLIR_SOURCE)

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
      list(POP_BACK _SOURCE_URL_SEGMENTS _LAST_URL_SEGMENT)
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
    # IREE deployable format.
    set(_BENCHMARK_DIR_NAME "iree-${_RULE_DRIVER}__${_RULE_TARGET_ARCH}__${_RULE_BENCHMARK_MODE}")
    set(_ARTIFACTS_DIR "${_ROOT_ARTIFACTS_DIR}/${_MODULE_NAME}/${_BENCHMARK_DIR_NAME}")

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
    list(APPEND _COMMON_NAME_SEGMENTS "${_RULE_BENCHMARK_MODE}")
    list(APPEND _COMMON_NAME_SEGMENTS "${_RULE_TARGET_BACKEND}")
    list(APPEND _COMMON_NAME_SEGMENTS "${_RULE_TARGET_ARCH}")

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
  endforeach()
endfunction()
