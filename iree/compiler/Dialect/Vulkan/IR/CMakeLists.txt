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

file(GLOB _GLOB_X_TD LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.td)
iree_cc_library(
  NAME
    IR
  HDRS
    "VulkanAttributes.h"
    "VulkanAttributes.h.inc"
    "VulkanDialect.h"
    "VulkanEnums.h.inc"
    "VulkanTypes.h"
  SRCS
    "VulkanAttributes.cpp"
    "VulkanAttributes.cpp.inc"
    "VulkanDialect.cpp"
    "VulkanEnums.cpp.inc"
    "VulkanTypes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRSupport
    iree::compiler::Dialect::IREE::IR
  PUBLIC
)

iree_tablegen_library(
  NAME
    VulkanAttrsGen
  TD_FILE
    "VulkanAttributes.td"
  OUTS
    -gen-struct-attr-decls VulkanAttributes.h.inc
    -gen-struct-attr-defs VulkanAttributes.cpp.inc
)

iree_tablegen_library(
  NAME
    VulkanEnumsGen
  TD_FILE
    "VulkanBase.td"
  OUTS
    -gen-enum-decls VulkanEnums.h.inc
    -gen-enum-defs VulkanEnums.cpp.inc
)
