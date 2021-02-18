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

add_subdirectory("compilation")
add_subdirectory("unittest")

iree_py_library(
  NAME
    testing
  SRCS
    "__init__.py"
    "lowerings.py"
    "test_runner.py"
    "utils.py"
)


function(iree_e2e_test src lowerings labels)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  string(REPLACE ".py" "" _name ${src})

  list(LENGTH lowerings lowering_count)
  list(LENGTH labels label_count)

  if(NOT lowering_count EQUAL lowering_count)
    message(SEND_ERROR
        "lowerings count ${lowering_count} does not match labels count ${label_count}")
  endif()

  math(EXPR _MAX_INDEX "${lowering_count} - 1")
  foreach(_index RANGE "${_MAX_INDEX}")
    list(GET lowerings ${_index} _lowering)
    list(GET labels ${_index} _label)

    set(_target_name "${_PACKAGE_PATH}/${_name}__${_lowering}")
    add_test(
      NAME
        ${_target_name}
      WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND
        "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
        "--lowering=${_lowering}"
    )

    set_property(TEST ${_target_name} PROPERTY LABELS "${_labels}")
    set_property(TEST ${_target_name} PROPERTY ENVIRONMENT
        "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python")
  endforeach()
endfunction()
