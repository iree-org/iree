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

# iree_python_benchmark_suite()
#
# Generates benchmark suites for Python ML models/layers.
#
# Mirrors the bzl rule of the same name.
#
# Parameters:
#   MODELS: A list of models to generate benchmark artifacts for.
#   MODEL_SCRIPT: The Python script used to generate benchmark artifacts.
#   MODEL_SCRIPT_ARGS: Command-line options and their values to the
#       model script.
#   NEED_ARG_FOR_MODEL: Whether an additional "--model=" command-line option
#       should be given to the model generation script.
#
function(iree_python_benchmark_suite)
  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    "NEED_ARG_FOR_MODEL"
    "CONFIGURATION_NAME;MODEL_SCRIPT;TARGET_BACKEND"
    "COMPILATION_FLAGS;MODELS;MODEL_SCRIPT_ARGS;RUNTIME_FLAGS"
  )

  foreach(_MODEL IN LISTS _RULE_MODELS)
    # Construct the benchmark generation target name, which is the model name
    # followed by target backend and configuration.
    set(_NAME_LIST "generate_benchmark_artifact")
    list(APPEND _NAME_LIST "${_MODEL}")
    list(APPEND _NAME_LIST "${_RULE_TARGET_BACKEND}")
    list(APPEND _NAME_LIST "${_RULE_CONFIGURATION_NAME}")
    list(JOIN _NAME_LIST "__" _NAME)

    set(_MODEL_CL "")
    if (_RULE_NEED_ARG_FOR_MODEL)
      set(_MODEL_CL "--model=${_MODEL}")
    endif()

    set(_RUNTIME_FLAGS_CL "")
    if (_RULE_RUNTIME_FLAGS)
      set(_RUNTIME_FLAGS_CL "--runtime_flags=\"${_RULE_RUNTIME_FLAGS}\"")
    endif()

    add_custom_target("${_NAME}"
      COMMAND
        "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}"
        "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_MODEL_SCRIPT}"
          ${_RULE_MODEL_SCRIPT_ARGS}
          "${_MODEL_CL}"
          "--target_backend=${_RULE_TARGET_BACKEND}"
          "--configuration_names=${_RULE_CONFIGURATION_NAME}"
          "--compilation_flags=\"${_RULE_COMPILATION_FLAGS}\""
          "${_RUNTIME_FLAGS_CL}"
      WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
      COMMENT
        "${_NAME}"
    )

    add_dependencies(iree-generate-benchmark-suites "${_NAME}")
  endforeach()

endfunction()

