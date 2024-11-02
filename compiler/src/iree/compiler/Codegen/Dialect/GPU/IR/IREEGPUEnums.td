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

// Format: <virtual><kind>_<output-type>_<M>x<N>x<K>_<input-type>
//
// "virtual": Prefixes intrinsic with "V" to represent Non native-MFMA
//             emulating a larger MMA with smaller ones. This is useful
//             to interleave reads in K-dim, S.T we can have wider reads
//             or align layouts between matmuls.
//
// Values: 0xABCD where:
// * A = vendor:
//   * 0 = AMD
//   * 1 = NVIDIA
// * B is architecture:
//   * For AMD:
//     * 0 = RDNA3
//     * 8 = CDNA2
//     * 9 = CDNA3
// * C is A/B data type:
//   * 0 = f32
//   * 1 = f16
//   * 2 = bf16
//   * 3 = f8e5m2 (and variants like fnuz).
//   * 4 = f8e4m3 (and variants like fnuz).
//   * 8 = i8
// * D enumerates intrinsics for the same data type.
//
// CDNA3 instrinsics
def MFMA_F32_16x16x4_F32 : I32EnumAttrCase<"MFMA_F32_16x16x4_F32", 0x0900>;
def MFMA_F32_16x16x16_F16 : I32EnumAttrCase<"MFMA_F32_16x16x16_F16", 0x0910>;
def MFMA_F32_32x32x8_F16  : I32EnumAttrCase<"MFMA_F32_32x32x8_F16", 0x0911>;
def VMFMA_F32_16x16x32_F16  : I32EnumAttrCase<"VMFMA_F32_16x16x32_F16", 0x0912>;
def VMFMA_F32_32x32x16_F16  : I32EnumAttrCase<"VMFMA_F32_32x32x16_F16", 0x0913>;
def MFMA_F32_16x16x16_BF16  : I32EnumAttrCase<"MFMA_F32_16x16x16_BF16", 0x0920>;
def MFMA_F32_32x32x8_BF16  : I32EnumAttrCase<"MFMA_F32_32x32x8_BF16", 0x0921>;
def MFMA_F32_16x16x32_F8E5M2FNUZ  : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E5M2FNUZ", 0x0930>;
def MFMA_F32_16x16x32_F8E4M3FNUZ  : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E4M3FNUZ", 0x0940>;
// V-Intrinsic below interleaves read from K-dim from one 8xF8 to two 4xF8.
// (Useful in F8 chained-MM to align B-layout of 2nd MM to C-layout of 1st MM)
def VMFMA_F32_16x16x32_F8E4M3FNUZ  : I32EnumAttrCase<"VMFMA_F32_16x16x32_F8E4M3FNUZ", 0x0941>;
def MFMA_I32_16x16x32_I8  : I32EnumAttrCase<"MFMA_I32_16x16x32_I8", 0x0980>;
def MFMA_I32_32x32x16_I8  : I32EnumAttrCase<"MFMA_I32_32x32x16_I8", 0x0981>;

// CDNA2 instrinsics
def MFMA_I32_16x16x16_I8  : I32EnumAttrCase<"MFMA_I32_16x16x16_I8", 0x0880>;
def MFMA_I32_32x32x8_I8  : I32EnumAttrCase<"MFMA_I32_32x32x8_I8", 0x0881>;

// TODO: Create separate WMMA ops for AMD and NVIDIA GPUs
def WMMA_F32_16x16x16_F16 : I32EnumAttrCase<"WMMA_F32_16x16x16_F16", 0x0010>;
def WMMA_F16_16x16x16_F16 : I32EnumAttrCase<"WMMA_F16_16x16x16_F16", 0x0011>;

// TODO: The actual I8 instruction allows specifying (mixed) signedness.
// This will need to become its own class of MMA attribute.
def WMMA_I32_16x16x16_I8 : I32EnumAttrCase<"WMMA_I32_16x16x16_I8", 0x0080>;

def IREEGPU_MMAIntrinsic : IREEGPU_I32MmaEnumAttr<"MMAIntrinsic",
    "Descriptor for different MMA intrinsics", [
      MFMA_F32_16x16x4_F32,
      MFMA_F32_16x16x16_F16,
      MFMA_F32_32x32x8_F16,
      VMFMA_F32_16x16x32_F16,
      VMFMA_F32_32x32x16_F16,
      MFMA_F32_16x16x16_BF16,
      MFMA_F32_32x32x8_BF16,
      MFMA_F32_16x16x32_F8E4M3FNUZ,
      MFMA_F32_16x16x32_F8E5M2FNUZ,
      VMFMA_F32_16x16x32_F8E4M3FNUZ,
      MFMA_I32_16x16x32_I8,
      MFMA_I32_32x32x16_I8,
      MFMA_I32_16x16x16_I8,
      MFMA_I32_32x32x8_I8,
      WMMA_F32_16x16x16_F16,
      WMMA_F16_16x16x16_F16,
      WMMA_I32_16x16x16_I8
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

def MMA_Workgroup : I32EnumAttrCase<"Workgroup", 0>;
def MMA_Subgroup : I32EnumAttrCase<"Subgroup", 1>;

def IREEGPU_MMAScope : IREEGPU_I32MmaEnumAttr<"MMAScope",
    "Descriptor for a particular scope of an MMA operation", [
      MMA_Workgroup,
      MMA_Subgroup
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

//===----------------------------------------------------------------------===//
// Pipeline options
//===----------------------------------------------------------------------===//

class IREEGPU_I32PipelineEnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

// ReorderWorkgroups EnumAttrCases.
def ReorderWorkgroupsNone : I32EnumAttrCase<"None", 0>;
def ReorderWorkgroupsSwizzle : I32EnumAttrCase<"Swizzle", 1>;
def ReorderWorkgroupsTranspose : I32EnumAttrCase<"Transpose", 2>;

// EnumAttr for workgroup reordering strategy enums.
def IREEGPU_ReorderWorkgroupsStrategy : IREEGPU_I32PipelineEnumAttr<"ReorderWorkgroupsStrategy",
    "Strategy for workgroup reordering", [
      ReorderWorkgroupsNone,
      ReorderWorkgroupsSwizzle,
      ReorderWorkgroupsTranspose
    ]> {
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS
