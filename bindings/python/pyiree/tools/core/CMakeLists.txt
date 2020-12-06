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

iree_py_library(
  NAME
    core
  SRCS
    "__init__.py"
)

# Adds a command to TARGET which symlinks a tool from elsewhere
# (FROM_TOOL_TARGET_NAME) to a local file name (TO_EXE_NAME).
function(_symlink_tool)
  cmake_parse_arguments(
    ARG
    ""
    "TARGET;FROM_TOOL_TARGET;TO_EXE_NAME"
    ""
    ${ARGN}
  )

  # Transform TARGET
  iree_package_ns(_PACKAGE_NS)
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${ARG_TARGET}")
  set(_FROM_TOOL_TARGET ${ARG_FROM_TOOL_TARGET})

  add_custom_command(
    TARGET "${_TARGET}"
    BYPRODUCTS
      "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:${_FROM_TOOL_TARGET}>"
        "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
  )
endfunction()

if(${IREE_BUILD_COMPILER})
  _symlink_tool(
    TARGET core
    FROM_TOOL_TARGET iree_tools_iree-translate
    TO_EXE_NAME iree-translate
  )
endif()
