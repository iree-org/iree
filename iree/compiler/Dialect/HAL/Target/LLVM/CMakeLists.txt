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

add_subdirectory(test)

iree_cc_library(
  NAME
    LLVM
  HDRS
    "LLVMTarget.h"
  SRCS
    "LLVMTarget.cpp"
  DEPS
    LLVMCore
    MLIRIR
    MLIRLinalgOps
    MLIRLinalgTransforms
    MLIRLoopToStandard
    MLIRStandardOps
    MLIRStandardToLLVM
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTransforms
    iree::compiler::Dialect::HAL::Target::ExecutableTarget
    iree::compiler::Translation::XLAToLinalg
    iree::compiler::Translation::XLAToLinalg::IREELinalgTensorToBuffer
    iree::schemas::llvmir_executable_def_cc_fbs
  ALWAYSLINK
  PUBLIC
)
