// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// GPU Workgroup Processor (WGP) Level Feature/Limit Enums
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Compute

// IEEE 754 double precision floating point format in computation
def IREEGPU_CFBW_64 : I32BitEnumAttrCaseBit<"FP64", 0, "fp64">;
// IEEE 754 single precision floating point format in computation
def IREEGPU_CFBW_32 : I32BitEnumAttrCaseBit<"FP32", 1, "fp32">;
// IEEE 754 half precision floating point format in computation
def IREEGPU_CFBW_16 : I32BitEnumAttrCaseBit<"FP16", 2, "fp16">;
// Signed/unsigned 64-bit integer format in computation
def IREEGPU_CIBW_64 : I32BitEnumAttrCaseBit<"Int64", 3, "int64">;
// Signed/unsigned 32-bit integer format in computation
def IREEGPU_CIBW_32 : I32BitEnumAttrCaseBit<"Int32", 4, "int32">;
// Signed/unsigned 16-bit integer format in computation
def IREEGPU_CIBW_16 : I32BitEnumAttrCaseBit<"Int16", 5, "int16">;
// Signed/unsigned 8-bit integer format in computation
def IREEGPU_CIBW_8  : I32BitEnumAttrCaseBit<"Int8",  6, "int8">;

def IREEGPU_ComputeBitwidths : I32BitEnumAttr<
  "ComputeBitwidths", "Supported bitwidths for compute",
  [IREEGPU_CFBW_64, IREEGPU_CFBW_32, IREEGPU_CFBW_16,
   IREEGPU_CIBW_64, IREEGPU_CIBW_32, IREEGPU_CIBW_16, IREEGPU_CIBW_8]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Storage

// Direct 64-bit value access from/to memory storage
def IREEGPU_SBW_64 : I32BitEnumAttrCaseBit<"B64", 0, "b64">;
// Direct 32-bit value access from/to memory storage
def IREEGPU_SBW_32 : I32BitEnumAttrCaseBit<"B32", 1, "b32">;
// Direct 16-bit value access from/to memory storage
def IREEGPU_SBW_16 : I32BitEnumAttrCaseBit<"B16", 2, "b16">;
// Direct 8-bit value access from/to memory storage
def IREEGPU_SBW_8  : I32BitEnumAttrCaseBit<"B8",  3, "b8">;

def IREEGPU_StorageBitwidths : I32BitEnumAttr<
  "StorageBitwidths", "Supported bitwidths for storage",
  [IREEGPU_SBW_64, IREEGPU_SBW_32, IREEGPU_SBW_16, IREEGPU_SBW_8]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Subgroup operations

def IREEGPU_SO_None       : I32BitEnumAttrCaseNone<"None", "none">;
// Subgroup shuffle index/xor operation
def IREEGPU_SO_Shuffle    : I32BitEnumAttrCaseBit<"Shuffle",    0, "shuffle">;
// Subgroup arithmetic add/mul/min/max/and/or/xor reduction operation
def IREEGPU_SO_Arithmetic : I32BitEnumAttrCaseBit<"Arithmetic", 1, "arithmetic">;

def IREEGPU_SubgroupOps : I32BitEnumAttr<
  "SubgroupOps", "Supported subgroup ops",
  [IREEGPU_SO_None, IREEGPU_SO_Shuffle, IREEGPU_SO_Arithmetic]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Dot product operations

def IREEGPU_DPO_None      : I32BitEnumAttrCaseNone<"None", "none">;
// Dot product 4xi8 -> i32 operation
def IREEGPU_DPO_4xI8ToI32 : I32BitEnumAttrCaseBit<"DP4xI8ToI32", 0, "dp4xi8toi32">;

def IREEGPU_DotProductOps : I32BitEnumAttr<
  "DotProductOps", "Supported dot product ops",
  [IREEGPU_DPO_None, IREEGPU_DPO_4xI8ToI32]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// MMA intrinsic

class IREEGPU_I32MmaEnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

// Format: <kind>_<input-type>_<M>x<N>x<K>_<output-type>
def MFMA_F32_16x16x4_F32 : I32EnumAttrCase<"MFMA_F32_16x16x4_F32", 0>;
def MFMA_F32_16x16x16_F16 : I32EnumAttrCase<"MFMA_F32_16x16x16_F16", 1>;
def MFMA_F32_32x32x8_F16  : I32EnumAttrCase<"MFMA_F32_32x32x8_F16", 2>;
def MFMA_F32_16x16x32_F8E4M3FNUZ  : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E4M3FNUZ", 3>;
def MFMA_I32_16x16x32_I8  : I32EnumAttrCase<"MFMA_I32_16x16x32_I8", 4>;
def MFMA_I32_32x32x16_I8  : I32EnumAttrCase<"MFMA_I32_32x32x16_I8", 5>;
// TODO: Create separate WMMA ops for AMD and NVIDIA GPUs
def WMMA_F32_16x16x16_F16 : I32EnumAttrCase<"WMMA_F32_16x16x16_F16", 6>;
def WMMA_F16_16x16x16_F16 : I32EnumAttrCase<"WMMA_F16_16x16x16_F16", 7>;

def IREEGPU_MMAIntrinsic : IREEGPU_I32MmaEnumAttr<"MMAIntrinsic",
    "Descriptor for different MMA intrinsics", [
      MFMA_F32_16x16x4_F32,
      MFMA_F32_16x16x16_F16,
      MFMA_F32_32x32x8_F16,
      MFMA_F32_16x16x32_F8E4M3FNUZ,
      MFMA_I32_16x16x32_I8,
      MFMA_I32_32x32x16_I8,
      WMMA_F32_16x16x16_F16,
      WMMA_F16_16x16x16_F16
    ]>;

def MMA_LHS : I32EnumAttrCase<"Lhs", 0>;
def MMA_RHS : I32EnumAttrCase<"Rhs", 1>;
def MMA_ACC : I32EnumAttrCase<"Acc", 2>;

def IREEGPU_MMAFragment : IREEGPU_I32MmaEnumAttr<"MMAFragment",
    "Descriptor for a particular fragment of an MMA operation", [
      MMA_LHS,
      MMA_RHS,
      MMA_ACC
    ]>;

//===----------------------------------------------------------------------===//
// Tiling levels

def Workgroup : I32EnumAttrCase<"Workgroup", 0>;
def Reduction : I32EnumAttrCase<"Reduction", 1>;
def PartialReduction : I32EnumAttrCase<"PartialReduction", 2>;
def Thread : I32EnumAttrCase<"Thread", 3>;
def Subgroup : I32EnumAttrCase<"Subgroup", 4>;
def Lane : I32EnumAttrCase<"Lane", 5>;

/// Enum descriptor for the set of tiling levels for GPU pass pipelines.
/// Note that `Thread` tiling is mutually exclusive with `Subgroup` and
/// `Lane` tiling, and `Lane` tiling is only legal if the same operation
/// is also tiled or fused to subgroups.
def IREEGPU_TilingLevel : IREEGPU_I32MmaEnumAttr<"TilingLevel",
    "Descriptor for tiling levels for GPU lowering configs", [
      Workgroup,
      Reduction,
      PartialReduction,
      Thread,
      Subgroup,
      Lane
    ]>;

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS
