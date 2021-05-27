// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_DIALECT_HAL_IR_LOWERINGCONFIG
#define IREE_COMPILER_DIALECT_HAL_IR_LOWERINGCONFIG

// Putting this in HAL dialect for now.
include "iree/compiler/Dialect/HAL/IR/HALDialect.td"

def CPU_Default
    : I32EnumAttrCase<"CPUDefault", 0>;
def CPU_Vectorization
    : I32EnumAttrCase<"CPUVectorization", 1>;

// EnumAttrCase for all known lowerings for ops within dispatch region
// to scalar/native-vector code.
def DispatchLoweringPassPipelineEnum : I32EnumAttr<
    "DispatchLoweringPassPipeline",
    "identifier for pass pipeline use to lower dispatch region",
    [CPU_Default, CPU_Vectorization]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";
}

def TileSizesListAttr :
    TypedArrayAttrBase<I64ArrayAttr,
                       "list of tile sizes for all levels"> { }

// Attribute that carries information needed to perform
// tiling/vectorization, etc.
def HAL_LoweringConfigAttr :
  StructAttr<"LoweringConfig", HAL_Dialect, [
    StructFieldAttr<"tileSizes",
        DefaultValuedAttr<TileSizesListAttr, "{}">>,
    StructFieldAttr<"nativeVectorSize",
        DefaultValuedAttr<I64ArrayAttr, "{}">>,
  ]>;

#endif // IREE_COMPILER_DIALECT_HAL_IR_LOWERINGCONFIG
