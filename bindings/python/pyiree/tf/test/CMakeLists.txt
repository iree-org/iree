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

# TODO(meadowlark): figure out how to put these in tf/testing/
set(ALL_TF_LOWERINGS
    "reference"
    "tflite"
    "vmla_via_mhlo"
    "llvmaot_via_mhlo"
    "vulkan_via_mhlo"
)
set(ALL_TF_LABELS
    "driver=tf"
    "driver=tflite"
    "driver=vmla"
    "driver=llvmaot"
    "driver=vulkan"
)

file(GLOB _all_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_test.py")
foreach(_filename ${_all_files})
  iree_e2e_test(${_filename} "${ALL_TF_LOWERINGS}" "${ALL_TF_LABELS}")
endforeach()
