# Copyright 2019 Google LLC
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

iree_cc_library(
  NAME
    HALToVM
  HDRS
    "ConvertHALToVM.h"
  SRCS
    "ConvertAllocatorOps.cpp"
    "ConvertBufferOps.cpp"
    "ConvertBufferViewOps.cpp"
    "ConvertCommandBufferOps.cpp"
    "ConvertControlFlowOps.cpp"
    "ConvertDeviceOps.cpp"
    "ConvertExecutableOps.cpp"
    "ConvertExperimentalOps.cpp"
    "ConvertHALToVM.cpp"
    "ConvertSemaphoreOps.cpp"
    "ConvertVariableOps.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRStandardOps
    MLIRTransforms
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Utils
    iree::compiler::Dialect::HAL::hal_imports
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::VM::Conversion
    iree::compiler::Dialect::VM::Conversion::StandardToVM
    iree::compiler::Dialect::VM::IR
  PUBLIC
)
