// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
