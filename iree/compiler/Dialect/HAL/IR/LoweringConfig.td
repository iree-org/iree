// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_DIALECT_HAL_IR_LOWERINGCONFIG
#define IREE_COMPILER_DIALECT_HAL_IR_LOWERINGCONFIG

// Putting this in HAL dialect for now.
include "iree/compiler/Dialect/HAL/IR/HALDialect.td"

// List of pre-existing pipelines for translating executables.
def CPU_Default
    : StrEnumAttrCase<"CPUDefault">;
def CPU_Vectorization
    : StrEnumAttrCase<"CPUVectorization">;
def CPU_TensorToVectors
    : StrEnumAttrCase<"CPUTensorToVectors">;
def LLVMGPU_SimpleDistribute
    : StrEnumAttrCase<"LLVMGPUDistribute">;
def LLVMGPU_Vectorize
    : StrEnumAttrCase<"LLVMGPUVectorize">;
def LLVMGPU_MatmulSimt
    : StrEnumAttrCase<"LLVMGPUMatmulSimt">;
def SPIRV_SimpleDistribute
    : StrEnumAttrCase<"SPIRVDistribute">;
def SPIRV_Vectorize
    : StrEnumAttrCase<"SPIRVVectorize">;
def SPIRV_DistributeToGlobalID
    : StrEnumAttrCase<"SPIRVDistributeToGlobalID">;
def None
    : StrEnumAttrCase<"None">;

// EnumAttrCase for all known lowerings for ops within dispatch region
// to scalar/native-vector code.
def DispatchLoweringPassPipelineEnum : StrEnumAttr<
    "DispatchLoweringPassPipeline",
    "identifier for pass pipeline use to lower dispatch region",
    [CPU_Default, CPU_TensorToVectors, CPU_Vectorization,
     LLVMGPU_SimpleDistribute, LLVMGPU_Vectorize, LLVMGPU_MatmulSimt,
     SPIRV_SimpleDistribute, SPIRV_Vectorize, SPIRV_DistributeToGlobalID,
     None]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def TileSizesListAttr :
    TypedArrayAttrBase<I64ArrayAttr,
                       "list of tile sizes for all levels"> { }

// Attribute that captures information needed for translating the executables.
def TranslationInfoAttr :
  StructAttr<"TranslationInfo", HAL_Dialect, [
    StructFieldAttr<"passPipeline", DispatchLoweringPassPipelineEnum>,
    StructFieldAttr<"workloadPerWorkgroup",
        DefaultValuedAttr<I64ArrayAttr, "{}">>,
  ]>;

// Attribute that carries information needed to perform
// tiling/vectorization, etc.
def HAL_LoweringConfigAttr :
  StructAttr<"LoweringConfig", HAL_Dialect, [
    StructFieldAttr<"tileSizes",
        DefaultValuedAttr<TileSizesListAttr, "{}">>,
    StructFieldAttr<"nativeVectorSize",
        DefaultValuedAttr<I64ArrayAttr, "{}">>,
    StructFieldAttr<"passPipeline",
        DefaultValuedAttr<
            DispatchLoweringPassPipelineEnum,
            "\"IREE::HAL::DispatchLoweringPassPipeline::None\"">>,
    StructFieldAttr<"workgroupSize",
        DefaultValuedAttr<I64ArrayAttr, "{}">>
  ]>;

#endif // IREE_COMPILER_DIALECT_HAL_IR_LOWERINGCONFIG
