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

# Additional libraries containing statically registered functions/flags, which
# should always be linked in to binaries.

# iree_glob_lit_tests()
#
# CMake function to imitate Bazel's iree_glob_lit_tests rule.
function(iree_glob_lit_tests)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  iree_package_name(_PACKAGE_NAME)
  file(GLOB_RECURSE _TEST_FILES *.mlir)
  set(_TOOL_DEPS iree_tool_iree-opt IreeFileCheck)

  foreach(_TEST_FILE ${_TEST_FILES})
    get_filename_component(_TEST_FILE_LOCATION ${_TEST_FILE} DIRECTORY)
    get_filename_component(_TEST_NAME ${_TEST_FILE} NAME_WE)
    set(_NAME "${_PACKAGE_NAME}_${_TEST_NAME}")

    add_test(NAME ${_NAME} COMMAND ${CMAKE_SOURCE_DIR}/iree/tools/run_lit.sh ${_TEST_FILE} ${CMAKE_SOURCE_DIR}/iree/tools/IreeFileCheck.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/iree/tools)
    set_tests_properties(${_NAME} PROPERTIES DEPENDS _TOOL_DEPS)
  endforeach()
endfunction()
