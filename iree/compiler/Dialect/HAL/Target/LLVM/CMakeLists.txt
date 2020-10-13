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

if(NOT ${IREE_TARGET_BACKEND_LLVM-IR})
  return()
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    LLVMBaseTarget
  HDRS
    "LLVMBaseTarget.h"
  SRCS
    "LLVMBaseTarget.cpp"
  DEPS
    ::LLVMIRPasses
    ::LLVMTargetOptions
    LLVMSupport
    MLIRAffine
    MLIRLLVMIR
    MLIRLinalg
    MLIRSCF
    MLIRVector
    iree::compiler::Conversion::CodegenUtils
    iree::compiler::Conversion::LinalgToLLVM
    iree::compiler::Dialect::HAL::Target
  PUBLIC
)

iree_cc_library(
  NAME
    LLVMIRPasses
  HDRS
    "LLVMIRPasses.h"
  SRCS
    "LLVMIRPasses.cpp"
  DEPS
    ::LLVMTargetOptions
    LLVMCore
    LLVMPasses
    LLVMSupport
    LLVMTarget
    MLIRSupport
  PUBLIC
)

iree_cc_library(
  NAME
    LLVMTargetOptions
  HDRS
    "LLVMTargetOptions.h"
  SRCS
    "LLVMTargetOptions.cpp"
  DEPS
    LLVMPasses
    LLVMSupport
    LLVMTarget
  PUBLIC
)
