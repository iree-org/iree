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

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "pw_add_multiwg.mlir"
    "reduce.mlir"
  DATA
    iree::tools::IreeFileCheck
    iree::tools::iree-run-mlir
  LABELS
    "hostonly"
)

file(GLOB _GLOB_X_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.mlir)
file(GLOB _GLOB_PW_ADD_MULTIWG_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS pw_add_multiwg.mlir)
list(REMOVE_ITEM _GLOB_X_MLIR ${_GLOB_PW_ADD_MULTIWG_MLIR})
file(GLOB _GLOB_REDUCE_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS reduce.mlir)
list(REMOVE_ITEM _GLOB_X_MLIR ${_GLOB_REDUCE_MLIR})
iree_check_single_backend_test_suite(
  NAME
    check_linalg-to-spirv_vulkan
  SRCS
    "${_GLOB_X_MLIR}"
  TARGET_BACKEND
    vulkan-spirv
  DRIVER
    vulkan
  COMPILER_FLAGS
    "-iree-use-linalg-to-spirv-path"
)
