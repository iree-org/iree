// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUENUMS
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUENUMS

include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// MMA intrinsic
//===----------------------------------------------------------------------===//

class IREECPU_I32EnumAttr<string name, string summary,
                          list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::CPU";
  let genSpecializedAttr = 0;
}

// Enum values for CPU MMA (matrix-multiply-accumulate) intrinsics.
//
// The may be "virtual" intrinsics not directly corresponding to a LLVM
// intrinsic. Here are a few possible scenarios where that happens:
//   * The target may lack an intrinsic for some narrow LHS/RHS type like i8,
//     requiring upcasting i8 to i16 and using i16 intrinsics. In that case, the
//     enum here represents a i8 matmul but the lowering creates the cast and
//     i16 matmul combination.
//   * The target may have an intrinsic consuming only a segment of one of its
//     register operands. This is typical on Arm with
//     multiply-accumulate-by-element instructions. Here, we may choose between
//     two approaches:
//     - Either reflect the actual intrinsic 1:1, consuming a fractional
//     register.
//       The advantage is greater flexibility, and only needing one enum value.
//       The downside is reliance on compiler optimization to produce the
//       expected code in the typical case where several such intrinsics
//       collectively use the whole register.
//     - Or have enum values representing a group of such intrinsics,
//     collectively
//       using whole registers. The advantage is ensuring that the generated
//       code is what is expected. The downside is needing several enum values
//       with different tile sizes.
//
// Note that even a scalar multiply-accumulate is technically a matmul
// with MxNxK shape 1x1x1. Vector element-wise multiply-accumulate with vector
// size N is seen here as a matrix multiply-accumulate with MxNxK shape 1xNx1.
// It could be equivalently seen as shape Nx1x1, so we have to pick a convention
// here, breaking the M-N symmetry: we choose to spread the vector dimension
// along the N matrix dimension. The MMA attribute types that wrap this enum
// shall combine it with a boolean "transpose" parameter flipping that, which
// will typically be used for narrow matmuls with a small N-dimension size.
//
// Symbolic names are of the form
//
//    MMA_ARCH_MxNxK_ACC_LHS_RHS_FLAVOR
//
// Where:
//    * ARCH denotes the architecture and ISA extension, e.g. X86_AVX512BF16.
//    * MxNxK denotes the matmul shape performed by this intrinsic.
//    * ACC_LHS_RHS denote the element types. The _RHS is omitted when unneeded.
//    * FLAVOR distinguishes intrinsics otherwise agreeing, and/or hints at how
//      the intrinsic is meant to be lowered in case it's not just a direct
//      lowering to the target LLVM intrinsic. For example, for i8 matmul on
//      targets only providing instructions for i32 matmuls, or for f16 matmuls
//      on targets only providing instructions for f32 matmuls, we may need to
//      lower to a cast followed by a matmul on the wider type. This would be
//      indicated by a _CAST_I16 or _CAST_F32 suffix.
//
// Values are 0xABCD where:
// * A denotes the architecture:
//   - 0 is reserved for the None-value.
//   - 1 is x86.
//   - 2 is Arm.
//   - 3 is RISC-V.
// * B denotes the instruction family / ISA extension set.
//   - For x86:
//     - 1 is SSE.
//     - 2 is AVX/AVX2.
//     - 3 is AVX-512.
//   - For Arm:
//     - 1 is NEON.
//     - 2 is SVE/SVE2.
//     - 3 is SME/SME2.
// * C denotes the element type of the A-matrix (LHS):
//   * 0 = float64 (IEEE754 half precision).
//   * 1 = float32 (IEEE754 half precision).
//   * 2 = float16 (IEEE754 half precision).
//   * 3 = bfloat16 ("Brain" float format, i.e. high half of float32).
//   * 4 = 8-bit float (incl. f8E5M2, f8E4M3FN, etc).
//   * A = 16-bit integer (any signedness).
//   * C = 8-bit integer (any signedness).
// * D enumerates intrinsics that share the same 0xABC* bits.
//
def IREECPU_MMA_None : I32EnumAttrCase<"None", 0>;
def MMA_X86_AVX512_1x8x1_F64_F64
    : I32EnumAttrCase<"MMA_X86_AVX512_1x8x1_F64_F64", 0x1300>;
def MMA_X86_AVX512_1x16x1_F32_F32
    : I32EnumAttrCase<"MMA_X86_AVX512_1x16x1_F32_F32", 0x1310>;
def MMA_X86_AVX512_1x16x1_F32_F16_CASTF32
    : I32EnumAttrCase<"MMA_X86_AVX512_1x16x1_F32_F16_CASTF32", 0x1320>;
def MMA_X86_AVX512FP16_1x32x1_F16_F16
    : I32EnumAttrCase<"MMA_X86_AVX512FP16_1x32x1_F16_F16", 0x1321>;
def MMA_X86_AVX512BF16_1x16x2_F32_BF16
    : I32EnumAttrCase<"MMA_X86_AVX512BF16_1x16x2_F32_BF16", 0x1330>;
def MMA_X86_AVX512_1x16x2_I32_I16
    : I32EnumAttrCase<"MMA_X86_AVX512_1x16x2_I32_I16", 0x13A0>;
def MMA_X86_AVX512VNNI_1x16x2_I32_I16
    : I32EnumAttrCase<"MMA_X86_AVX512VNNI_1x16x2_I32_I16", 0x13A1>;
def MMA_X86_AVX512_1x16x2_I32_I8_CASTI16
    : I32EnumAttrCase<"MMA_X86_AVX512_1x16x2_I32_I8_CASTI16", 0x13C0>;
def MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16
    : I32EnumAttrCase<"MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16", 0x13C1>;

def IREECPU_MMAIntrinsic
    : IREECPU_I32EnumAttr<
          "MMAIntrinsic", "Descriptor for different MMA intrinsics",
          [IREECPU_MMA_None,

           // X86 AVX-512
           MMA_X86_AVX512_1x8x1_F64_F64, MMA_X86_AVX512_1x16x1_F32_F32,
           MMA_X86_AVX512_1x16x1_F32_F16_CASTF32,
           MMA_X86_AVX512FP16_1x32x1_F16_F16,
           MMA_X86_AVX512BF16_1x16x2_F32_BF16, MMA_X86_AVX512_1x16x2_I32_I16,
           MMA_X86_AVX512VNNI_1x16x2_I32_I16,
           MMA_X86_AVX512_1x16x2_I32_I8_CASTI16,
           MMA_X86_AVX512VNNI_1x16x2_I32_I8_CASTI16]>;

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUENUMS
