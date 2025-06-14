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

class IREEGPU_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let genSpecializedAttr = 0;
}

// These enum values are loosely named to match the target intrinsics that they
// lower to, with wiggle room. The general name format used here is, similar to
// typical target intrinsics:
//     <kind>_<elem-type-C>_<M>x<N>x<K>_<elem-type-A>[_<elem-type-B>]
//
// Deviations from target intrinsic naming include:
// * The same enum values are reused when the same intrinsic exists in different
//   architecture versions with slightly different target intrinsic name. For
//   instance, our MFMA_F32_16x16x4_F32 lowers to:
//   - v_mfma_f32_16x16x4f32 on CDNA1/CDNA2.
//   - v_mfma_f32_16x16x4_f32 on CDNA3.
// * Each enum value corresponds to one fully type-specialized intrinsic, in
//   contrast to target intrinsics which are sometimes type-generic. For
//   instance, on RDNA3, our WMMAR3_I32_16x16x16_I8 lowers to
//   v_wmma_i32_16x16x16_iu8 specialized for signed int8 operands.
// * Since RDNA3 and RDNA4 have mutually-incompatible WMMA instructions that
//   share a name, we use WMMAR3 and WMMAR4as their prefixes.
// * Each enum value uses IREE-style naming of element type. For example,
//   our MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ lowers to
//   v_mfma_f32_16x16x32_bf8_fp8.
//
// Values: 0xABCD where:
// * A = vendor:
//   * 1 = AMD
//   * 2 = NVIDIA
// * B = architecture. When an intrinsic exists in multiple architectures, this
//       should be the architecture it was introduced in, as long as it still
//       has the same semantics. If a new architecture breaks an existing
//       intrinsic's semantics, we can use that field for versioning.
//   * For AMD:
//     * 0 = CDNA1
//     * 1 = CDNA2
//     * 2 = CDNA3
//     * 3 = CDNA4
//     * 8 = RDNA3
//     * 9 = RDNA4
// * C = element type of A-matrix:
//   * 0 = 64-bit float (e.g. IEEE754 double precision)
//   * 1 = 32-bit float (e.g. IEEE754 single precision, and "xf32" fast variants)
//   * 2 = 16-bit float (incl. IREE754 half and bf16)
//   * 3 = 8-bit float (incl. f8E5M2, f8E4M3FN and "FNUZ" variants)
//   * C = 8-bit integer (any signedness)
// * D enumerates intrinsics that share the same 0xABC* bits.
//

// Introduced in CDNA1
def MFMA_F32_16x16x4_F32 : I32EnumAttrCase<"MFMA_F32_16x16x4_F32", 0x1010>;
def MFMA_F32_16x16x16_F16 : I32EnumAttrCase<"MFMA_F32_16x16x16_F16", 0x1020>;
def MFMA_F32_32x32x8_F16 : I32EnumAttrCase<"MFMA_F32_32x32x8_F16", 0x1021>;
def MFMA_I32_16x16x16_I8 : I32EnumAttrCase<"MFMA_I32_16x16x16_I8", 0x10C0>;
def MFMA_I32_32x32x8_I8 : I32EnumAttrCase<"MFMA_I32_32x32x8_I8", 0x10C1>;

// Introduced in CDNA2
def MFMA_F32_16x16x8_BF16 : I32EnumAttrCase<"MFMA_F32_16x16x8_BF16", 0x1120>;
def MFMA_F32_32x32x4_BF16 : I32EnumAttrCase<"MFMA_F32_32x32x4_BF16", 0x1121>;
def MFMA_F64_16x16x4_F64 : I32EnumAttrCase<"MFMA_F64_16x16x4_F64", 0x1100>;

// Introduced in CDNA3
def MFMA_F32_16x16x16_BF16 : I32EnumAttrCase<"MFMA_F32_16x16x16_BF16", 0x1220>;
def MFMA_F32_32x32x8_BF16 : I32EnumAttrCase<"MFMA_F32_32x32x8_BF16", 0x1221>;
def MFMA_F32_16x16x32_F8E5M2FNUZ : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E5M2FNUZ", 0x1230>;
def MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ", 0x1231>;
def MFMA_F32_16x16x32_F8E4M3FNUZ : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E4M3FNUZ", 0x1232>;
def MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ", 0x1233>;
def MFMA_F32_32x32x16_F8E5M2FNUZ : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E5M2FNUZ", 0x1234>;
def MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ", 0x1235>;
def MFMA_F32_32x32x16_F8E4M3FNUZ : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E4M3FNUZ", 0x1236>;
def MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ", 0x1237>;
def MFMA_I32_16x16x32_I8 : I32EnumAttrCase<"MFMA_I32_16x16x32_I8", 0x12C0>;
def MFMA_I32_32x32x16_I8 : I32EnumAttrCase<"MFMA_I32_32x32x16_I8", 0x12C1>;

// Introduced in CDNA4
def MFMA_F32_16x16x32_F16 : I32EnumAttrCase<"MFMA_F32_16x16x32_F16", 0x1320>;
def MFMA_F32_32x32x16_F16 : I32EnumAttrCase<"MFMA_F32_32x32x16_F16", 0x1321>;
def MFMA_F32_16x16x32_BF16 : I32EnumAttrCase<"MFMA_F32_16x16x32_BF16", 0x1322>;
def MFMA_F32_32x32x16_BF16 : I32EnumAttrCase<"MFMA_F32_32x32x16_BF16", 0x1323>;
def MFMA_F32_16x16x32_F8E5M2 : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E5M2", 0x1330>;
def MFMA_F32_16x16x32_F8E5M2_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E5M2_F8E4M3FN", 0x1331>;
def MFMA_F32_16x16x32_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E4M3FN", 0x1332>;
def MFMA_F32_16x16x32_F8E4M3FN_F8E5M2 : I32EnumAttrCase<"MFMA_F32_16x16x32_F8E4M3FN_F8E5M2", 0x1333>;
def MFMA_F32_32x32x16_F8E5M2 : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E5M2", 0x1334>;
def MFMA_F32_32x32x16_F8E5M2_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E5M2_F8E4M3FN", 0x1335>;
def MFMA_F32_32x32x16_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E4M3FN", 0x1336>;
def MFMA_F32_32x32x16_F8E4M3FN_F8E5M2 : I32EnumAttrCase<"MFMA_F32_32x32x16_F8E4M3FN_F8E5M2", 0x1337>;
def MFMA_F32_16x16x128_F8E5M2 : I32EnumAttrCase<"MFMA_F32_16x16x128_F8E5M2", 0x1338>;
def MFMA_F32_16x16x128_F8E5M2_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_16x16x128_F8E5M2_F8E4M3FN", 0x1339>;
def MFMA_F32_16x16x128_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_16x16x128_F8E4M3FN", 0x133A>;
def MFMA_F32_16x16x128_F8E4M3FN_F8E5M2 : I32EnumAttrCase<"MFMA_F32_16x16x128_F8E4M3FN_F8E5M2", 0x133B>;
def MFMA_F32_32x32x64_F8E5M2 : I32EnumAttrCase<"MFMA_F32_32x32x64_F8E5M2", 0x133C>;
def MFMA_F32_32x32x64_F8E5M2_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_32x32x64_F8E5M2_F8E4M3FN", 0x133D>;
def MFMA_F32_32x32x64_F8E4M3FN : I32EnumAttrCase<"MFMA_F32_32x32x64_F8E4M3FN", 0x133E>;
def MFMA_F32_32x32x64_F8E4M3FN_F8E5M2 : I32EnumAttrCase<"MFMA_F32_32x32x64_F8E4M3FN_F8E5M2", 0x133F>;
// TODO: add fp4, fp6, and their mixed-type variants
def MFMA_I32_16x16x64_I8 : I32EnumAttrCase<"MFMA_I32_16x16x64_I8", 0x13C0>;
def MFMA_I32_32x32x32_I8 : I32EnumAttrCase<"MFMA_I32_32x32x32_I8", 0x13C1>;

// Introduced in RDNA3
def WMMAR3_F32_16x16x16_F16 : I32EnumAttrCase<"WMMAR3_F32_16x16x16_F16", 0x1820>;
def WMMAR3_F16_16x16x16_F16 : I32EnumAttrCase<"WMMAR3_F16_16x16x16_F16", 0x1821>;
def WMMAR3_F32_16x16x16_BF16 : I32EnumAttrCase<"WMMAR3_F32_16x16x16_BF16", 0x1822>;
def WMMAR3_BF16_16x16x16_BF16 : I32EnumAttrCase<"WMMAR3_BF16_16x16x16_BF16", 0x1823>;
def WMMAR3_I32_16x16x16_I8 : I32EnumAttrCase<"WMMAR3_I32_16x16x16_I8", 0x18C0>;

// Introduced in RDNA4
def WMMAR4_F32_16x16x16_F16 : I32EnumAttrCase<"WMMAR4_F32_16x16x16_F16", 0x1920>;
def WMMAR4_F16_16x16x16_F16 : I32EnumAttrCase<"WMMAR4_F16_16x16x16_F16", 0x1921>;
def WMMAR4_F32_16x16x16_BF16 : I32EnumAttrCase<"WMMAR4_F32_16x16x16_BF16", 0x1922>;
def WMMAR4_BF16_16x16x16_BF16 : I32EnumAttrCase<"WMMAR4_BF16_16x16x16_BF16", 0x1923>;
def WMMAR4_F32_16x16x16_F8E5M2 : I32EnumAttrCase<"WMMAR4_F32_16x16x16_F8E5M2", 0x1930>;
def WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN : I32EnumAttrCase<"WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN", 0x1931>;
def WMMAR4_F32_16x16x16_F8E4M3FN : I32EnumAttrCase<"WMMAR4_F32_16x16x16_F8E4M3FN", 0x1932>;
def WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2 : I32EnumAttrCase<"WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2", 0x1933>;
def WMMAR4_I32_16x16x16_I8 : I32EnumAttrCase<"WMMAR4_I32_16x16x16_I8", 0x19C0>;

// NV intrinsics
def NV_WMMA_F32_16x16x16_F16 : I32EnumAttrCase<"NV_WMMA_F32_16x16x16_F16", 0x2020>;
def NV_WMMA_F16_16x16x16_F16 : I32EnumAttrCase<"NV_WMMA_F16_16x16x16_F16", 0x2021>;

def IREEGPU_MMAIntrinsic : IREEGPU_I32EnumAttr<"MMAIntrinsic",
    "Descriptor for different MMA intrinsics", [
      // Introduced in CDNA1
      MFMA_F32_16x16x4_F32,
      MFMA_F32_16x16x16_F16,
      MFMA_F32_32x32x8_F16,
      MFMA_I32_16x16x16_I8,
      MFMA_I32_32x32x8_I8,

      // Introduced in CDNA2
      MFMA_F32_16x16x8_BF16,
      MFMA_F32_32x32x4_BF16,
      MFMA_F64_16x16x4_F64,

      // Introduced in CDNA3
      MFMA_F32_16x16x16_BF16,
      MFMA_F32_32x32x8_BF16,
      MFMA_F32_16x16x32_F8E5M2FNUZ,
      MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ,
      MFMA_F32_16x16x32_F8E4M3FNUZ,
      MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ,
      MFMA_F32_32x32x16_F8E5M2FNUZ,
      MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ,
      MFMA_F32_32x32x16_F8E4M3FNUZ,
      MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ,
      MFMA_I32_16x16x32_I8,
      MFMA_I32_32x32x16_I8,

      // Introduced in CDNA4
      MFMA_F32_16x16x32_F16,
      MFMA_F32_32x32x16_F16,
      MFMA_F32_16x16x32_BF16,
      MFMA_F32_32x32x16_BF16,
      MFMA_F32_16x16x32_F8E5M2,
      MFMA_F32_16x16x32_F8E5M2_F8E4M3FN,
      MFMA_F32_16x16x32_F8E4M3FN,
      MFMA_F32_16x16x32_F8E4M3FN_F8E5M2,
      MFMA_F32_32x32x16_F8E5M2,
      MFMA_F32_32x32x16_F8E5M2_F8E4M3FN,
      MFMA_F32_32x32x16_F8E4M3FN,
      MFMA_F32_32x32x16_F8E4M3FN_F8E5M2,
      MFMA_F32_16x16x128_F8E5M2,
      MFMA_F32_16x16x128_F8E5M2_F8E4M3FN,
      MFMA_F32_16x16x128_F8E4M3FN,
      MFMA_F32_16x16x128_F8E4M3FN_F8E5M2,
      MFMA_F32_32x32x64_F8E5M2,
      MFMA_F32_32x32x64_F8E5M2_F8E4M3FN,
      MFMA_F32_32x32x64_F8E4M3FN,
      MFMA_F32_32x32x64_F8E4M3FN_F8E5M2,
      MFMA_I32_16x16x64_I8,
      MFMA_I32_32x32x32_I8,

      // RDNA3 intrinsics
      WMMAR3_F32_16x16x16_F16,
      WMMAR3_F16_16x16x16_F16,
      WMMAR3_F32_16x16x16_BF16,
      WMMAR3_BF16_16x16x16_BF16,
      WMMAR3_I32_16x16x16_I8,

      // RDNA4 intrinsics
      WMMAR4_F32_16x16x16_F16,
      WMMAR4_F16_16x16x16_F16,
      WMMAR4_F32_16x16x16_BF16,
      WMMAR4_BF16_16x16x16_BF16,
      WMMAR4_F32_16x16x16_F8E5M2,
      WMMAR4_F32_16x16x16_F8E5M2_F8E4M3FN,
      WMMAR4_F32_16x16x16_F8E4M3FN,
      WMMAR4_F32_16x16x16_F8E4M3FN_F8E5M2,
      WMMAR4_I32_16x16x16_I8,

      // NV intrinsics
      NV_WMMA_F32_16x16x16_F16,
      NV_WMMA_F16_16x16x16_F16,
    ]>;

def VMFMA_F32_16x16x32_F16  : I32EnumAttrCase<"VMFMA_F32_16x16x32_F16", 0>;
def VMFMA_F32_32x32x16_F16  : I32EnumAttrCase<"VMFMA_F32_32x32x16_F16", 1>;
def VMFMA_F32_16x16x32_F8E4M3FNUZ  : I32EnumAttrCase<"VMFMA_F32_16x16x32_F8E4M3FNUZ", 2>;
def VMFMA_F32_32x32x16_F8E4M3FNUZ  : I32EnumAttrCase<"VMFMA_F32_32x32x16_F8E4M3FNUZ", 3>;

def IREEGPU_VirtualMMAIntrinsic : IREEGPU_I32EnumAttr<"VirtualMMAIntrinsic",
    "Descriptor for different Virtual MMA intrinsics", [
      VMFMA_F32_16x16x32_F16,
      VMFMA_F32_32x32x16_F16,
      VMFMA_F32_16x16x32_F8E4M3FNUZ,
      VMFMA_F32_32x32x16_F8E4M3FNUZ,
    ]>;

def MMA_LHS : I32EnumAttrCase<"Lhs", 0>;
def MMA_RHS : I32EnumAttrCase<"Rhs", 1>;
def MMA_ACC : I32EnumAttrCase<"Acc", 2>;

def IREEGPU_MMAFragment : IREEGPU_I32EnumAttr<"MMAFragment",
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
def IREEGPU_TilingLevel : IREEGPU_I32EnumAttr<"TilingLevel",
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
def ReorderWorkgroupsTranspose : I32EnumAttrCase<"Transpose", 1>;

// EnumAttr for workgroup reordering strategy enums.
def IREEGPU_ReorderWorkgroupsStrategy : IREEGPU_I32PipelineEnumAttr<"ReorderWorkgroupsStrategy",
    "Strategy for workgroup reordering", [
      ReorderWorkgroupsNone,
      ReorderWorkgroupsTranspose
    ]> {
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUENUMS
