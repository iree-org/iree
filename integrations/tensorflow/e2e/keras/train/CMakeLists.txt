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

set(REFERENCE_BACKEND tf)

function(add_e2e_test filename target_backend labels)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  string(REPLACE ".py" "" _name ${filename})

  set(_name "${_PACKAGE_PATH}/${_name}__${target_backend}")
  add_test(
    NAME
      ${_name}
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND
      "${Python3_EXECUTABLE}" -B
      "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
      "--reference_backend=${REFERENCE_BACKEND}"
      "--target_backends=${target_backend}"
  )
  set_property(TEST ${_name} PROPERTY LABELS "${labels}")
  set_property(TEST ${_name} PROPERTY ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python")
endfunction()

add_e2e_test(regression_training_test.py tf "")
add_e2e_test(regression_training_test.py iree_vmla "")
add_e2e_test(regression_training_test.py iree_llvmaot "driver=dylib")
add_e2e_test(regression_training_test.py iree_vulkan "driver=vulkan")

add_e2e_test(classification_training_test.py tf "")
add_e2e_test(classification_training_test.py iree_vmla "")
add_e2e_test(classification_training_test.py iree_llvmaot "driver=dylib")
add_e2e_test(classification_training_test.py iree_vulkan "driver=vulkan")
