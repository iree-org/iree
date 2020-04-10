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

file(GLOB _GLOB_X_MLIR CONFIGURE_DEPENDS *.mlir)
file(GLOB _GLOB_REVERSE_MLIR CONFIGURE_DEPENDS reverse.mlir)
list(REMOVE_ITEM _GLOB_X_MLIR ${_GLOB_REVERSE_MLIR})
iree_lit_test_suite(
  NAME
    lit
  SRCS
    "${_GLOB_X_MLIR}"
  DATA
    iree::tools::IreeFileCheck
    iree::tools::iree-run-mlir
)

iree_check_test_suite(
  NAME
    check
  SRCS
    "reverse.mlir"
  TARGET_BACKENDS
    "vmla"
    "vulkan-spirv"
  DRIVERS
    "vmla"
    "vulkan"
)
