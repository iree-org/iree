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

# iree_tf_benchmark_suite()
#
# Generates benchmark suites for TensorFlow Python models/layers.
#
# Note that this CMake function works closely with tf_benchmark_utils.py;
# it uses command-line options in that script.
#
# Parameters:
#   MODELS: A list of models to generate benchmark artifacts for.
#   MODEL_SCRIPT: The Python script used to generate benchmark artifacts.
#   MODEL_SCRIPT_ARGS: A list of command-line options and their values to the
#       model script.
#   NEED_ARG_FOR_MODEL: Whether an additional "--model=<model>" command-line
#       option should be given to the model generation script.
#   TARGET_BACKEND: The target backend to generate benchmark artifacts for.
#   TARGET_ARCH: The detailed target backend's architecture.
#   COMPILATION_FLAGS: A list of command-line options and their values to
#       pass to the compiler for artifact generation.
#   RUNTIME_FLAGS: A list of command-line options and their values to pass
#       to the runtime when the benchmark is invoked.
#   BENCHMARK_MODE: The mode of this benchmark suite. This does not need
#       to be a unique identifier for this suite, which will also consider
#       the model, target backend, and architecture. It is mainly used to
#       differentiate suites on top of that.
#
function(iree_tf_benchmark_suite)
  if(NOT IREE_BUILD_BENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(
    PARSE_ARGV 0
    _RULE
    "NEED_ARG_FOR_MODEL"
    "BENCHMARK_MODE;MODEL_SCRIPT;TARGET_BACKEND;TARGET_ARCH"
    "COMPILATION_FLAGS;MODELS;MODEL_SCRIPT_ARGS;RUNTIME_FLAGS"
  )

  # Generate all benchmarks to the root build directory. This helps for
  # discovering them and launch them on devices.
  set(_ARTIFACTS_DIR "${CMAKE_BINARY_DIR}/benchmark_suites/models")

  foreach(_MODEL IN LISTS _RULE_MODELS)
    # Construct the benchmark generation target name, which is the model name
    # followed by target backend and configuration.
    set(_NAME_LIST "generate_benchmark_artifact")
    list(APPEND _NAME_LIST "${_MODEL}")
    list(APPEND _NAME_LIST "${_RULE_BENCHMARK_MODE}")
    list(APPEND _NAME_LIST "${_RULE_TARGET_BACKEND}")
    list(APPEND _NAME_LIST "${_RULE_TARGET_ARCH}")
    list(JOIN _NAME_LIST "__" _NAME)

    # Add a command-line option to specify the model if needed.
    set(_MODEL_CL "")
    if(_RULE_NEED_ARG_FOR_MODEL)
      set(_MODEL_CL "--model=${_MODEL}")
    endif()

    # Add a command-line option to specify the runtime flags if specified.
    set(_RUNTIME_FLAGS_CL "")
    if(_RULE_RUNTIME_FLAGS)
      set(_RUNTIME_FLAGS_CL "--runtime_flags=\"${_RULE_RUNTIME_FLAGS}\"")
    endif()

    set(_COMBINED_CONFG_NAME "${_RULE_TARGET_ARCH}__${_RULE_BENCHMARK_MODE}")

    add_custom_target("${_NAME}"
      COMMAND
        # Set the PYTHONPATH environment variable so benchmark Python scripts
        # can properly import iree packages under the build directory.
        "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}"
        "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_MODEL_SCRIPT}"
          ${_RULE_MODEL_SCRIPT_ARGS}
          "${_MODEL_CL}"
          "--artifacts_dir=${_ARTIFACTS_DIR}"
          "--configuration_names=${_COMBINED_CONFG_NAME}"
          "--target_backend=${_RULE_TARGET_BACKEND}"
          "--compilation_flags=\"${_RULE_COMPILATION_FLAGS}\""
          "${_RUNTIME_FLAGS_CL}"
      DEPENDS
        bindings_python_iree_tools_core_core
        bindings_python_iree_compiler_compiler
        bindings_python_iree_runtime_runtime
        integrations_tensorflow_bindings_python_iree_tools_tf_tf
      WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
      COMMENT
        "${_NAME}"
    )

    # Mark dependency so that we have one target to drive them all.
    add_dependencies(iree-generate-benchmark-suites "${_NAME}")
  endforeach()

endfunction()
