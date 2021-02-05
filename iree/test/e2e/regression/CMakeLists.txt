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

file(GLOB _GLOB_X_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.mlir)
file(GLOB _GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS dynamic_linalg_matmul_on_tensors.mlir)
list(REMOVE_ITEM _GLOB_X_MLIR ${_GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_MLIR})
file(GLOB _GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_FUSE_0_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS dynamic_linalg_matmul_on_tensors_fuse_0.mlir)
list(REMOVE_ITEM _GLOB_X_MLIR ${_GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_FUSE_0_MLIR})
file(GLOB _GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_FUSE_1_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS dynamic_linalg_matmul_on_tensors_fuse_1.mlir)
list(REMOVE_ITEM _GLOB_X_MLIR ${_GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_FUSE_1_MLIR})
file(GLOB _GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_FUSE_2_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS dynamic_linalg_matmul_on_tensors_fuse_2.mlir)
list(REMOVE_ITEM _GLOB_X_MLIR ${_GLOB_DYNAMIC_LINALG_MATMUL_ON_TENSORS_FUSE_2_MLIR})
iree_lit_test_suite(
  NAME
    lit
  SRCS
    "${_GLOB_X_MLIR}"
  DATA
    iree::tools::IreeFileCheck
    iree::tools::iree-benchmark-module
    iree::tools::iree-run-mlir
    iree::tools::iree-translate
  LABELS
    "hostonly"
)
