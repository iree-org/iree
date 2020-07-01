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

iree_cc_library(
  NAME
    ModelBuilder
  HDRS
    "ModelBuilder.h"
  SRCS
    "ModelBuilder.cpp"
  DEPS
    MLIRAffineOps
    MLIREDSC
    MLIRGPU
    MLIRIR
    MLIRLinalgOps
    MLIRLinalgTransforms
    MLIRSCF
    MLIRSCFTransforms
    MLIRSPIRV
    MLIRStandardOps
    MLIRVector
  PUBLIC
)

iree_cc_library(
  NAME
    VulkanLaunchWrapper
  HDRS
    "VulkanWrapperPass.h"
  SRCS
    "VulkanWrapperPass.cpp"
  DEPS
    LLVMSupport
    MLIRStandardToLLVM
    MLIRIR
    MLIRLLVMIR
    MLIRPass
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRStandardOps
    MLIRSupport
  PUBLIC
)

iree_cc_library(
  NAME
    ModelRunner
  HDRS
    "MemRefUtils.h"
    "ModelRunner.h"
  SRCS
    "ModelRunner.cpp"
  DEPS
    ::ModelBuilder
    LLVMSupport
    MLIRExecutionEngine
    MLIRGPU
    MLIRGPUToSPIRVTransforms
    MLIRGPUToVulkanTransforms
    MLIRSPIRVTransforms
    MLIRIR
    MLIRLinalgToLLVM
    MLIRLinalgTransforms
    MLIRPass
    MLIRSPIRV
    MLIRStandardToLLVM
    MLIRStandardToSPIRVTransforms
    MLIRTargetLLVMIR
    MLIRTransformUtils
    MLIRVector
    MLIRVectorToLLVM
    MLIRVectorToSCF
  PUBLIC
)
