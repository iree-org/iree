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

if(NOT ${IREE_TARGET_BACKEND_METAL-SPIRV})
  return()
endif()

iree_add_all_subdirs()

iree_cc_library(
  NAME
    MetalSPIRV
  HDRS
    "MetalSPIRVTarget.h"
  SRCS
    "MetalSPIRVTarget.cpp"
  DEPS
    ::SPIRVToMSL
    LLVMSupport
    MLIRIR
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRSupport
    flatbuffers
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::HAL::Target::SPIRVCommon
    iree::schemas::metal_executable_def_cc_fbs
  PUBLIC
)

iree_cc_library(
  NAME
    SPIRVToMSL
  HDRS
    "SPIRVToMSL.h"
  SRCS
    "SPIRVToMSL.cpp"
  DEPS
    LLVMSupport
    spirv-cross-msl
  INCLUDES
    ${PROJECT_SOURCE_DIR}/third_party
  PUBLIC
)
