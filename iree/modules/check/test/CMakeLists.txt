# Copyright 2020 Google LLC
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

iree_add_all_subdirs()

# file(GLOB _GLOB_X_MLIR CONFIGURE_DEPENDS *.mlir)
# iree_lit_test_suite(
#   NAME
#     lit
#   SRCS
#     "${_GLOB_X_MLIR}"
#   DATA
#     iree::modules::check::iree-check-module
#     iree::tools::IreeFileCheck
#     iree::tools::iree-translate
# )



### iree_bytecode_module
set(_FLAGS "-iree-mlir-to-vm-bytecode-module")
set(_TRANSLATE_TOOL "iree_tools_iree-translate")

# Resolve the executable binary path from the target name.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:${_TRANSLATE_TOOL}>)
SET(_GENERATED_MODULE_NAME "success_test_module.module")

set(_ARGS "${_FLAGS}")
list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/success.mlir")
list(APPEND _ARGS "-o")
list(APPEND _ARGS "${_GENERATED_MODULE_NAME}")


add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_GENERATED_MODULE_NAME}"
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
  DEPENDS ${_TRANSLATE_TOOL}
)

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${_GENERATED_MODULE_NAME} PROPERTIES GENERATED TRUE)

add_custom_target(
  "success_test_module" ALL
   DEPENDS
     "${CMAKE_CURRENT_BINARY_DIR}/${_GENERATED_MODULE_NAME}"
)
###

### iree_check_test
add_test(
  NAME
    success_test
  COMMAND
    "$<TARGET_FILE:iree_modules_check_iree-check-module>"
    "--input_file=${CMAKE_CURRENT_BINARY_DIR}/${_GENERATED_MODULE_NAME}"
    "--driver=vmla"
)

set_tests_properties(
    success_test
  PROPERTIES
    REQUIRED_FILES
      "${CMAKE_CURRENT_BINARY_DIR}/${_GENERATED_MODULE_NAME}"
    DEPENDS
      success_test_module
)
