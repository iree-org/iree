// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// CPU features: IREE cpu_data bits and mapping to LLVM target attribute keys.
//===----------------------------------------------------------------------===//
//
// Refer to the file comment in cpu_data.h. Summary:
// - This is included in both compiler and runtime.
// - Unconditionally define CPU features for all target architectures, not just
//   host, because this is needed by the compiler when targeting non-host.
// - The bit values will soon be set in stone, because they will be encoded in
//   generated modules.
// - Try to pack related features in the same cpu_data field and in nearby bits
//   if possible, on a best-effort basis.

#ifndef IREE_CPU_FEATURE_BIT
#error Define IREE_CPU_FEATURE_BIT before including this file.
#endif

// Format:
//   IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, "llvm_name")
//
// Where:
//   - `arch` is the CPU architecture that this CPU feature applies to, in
//     IREE's uppercase convention (e.g. ARM_64, X86_64; see IREE_ARCH_*).
//   - `field_index` is the index into the array returned by `iree_cpu_data_fields()`.
//     Allowed values range from 0 to (IREE_CPU_DATA_FIELD_COUNT-1).
//   - `bit_pos` is the position of the feature bit within that cpu data field.
//     As these fields are uint64_t, the range of `bit_pos` is 0..63.
//   - `bit_name` is the suffix to use to form the IREE C identifier for this
//     feature's bit value.
//   - `llvm_name` is the string name of the corresponding LLVM target attribute
//     (without a leading +).
//
// The bit_name should consistently be just the llvm_name uppercase'd and with
// non-alphanumeric characters dropped. Each CPU feature may have different
// names in each compiler, in each OS, and in the ISA manual, so we can't be
// consistent with everything. This matters particularly because we sometimes
// have side by side a compile-time (#ifdef) and a runtime check. If names are
// not consistent, the #ifdef may be unwittingly testing for the wrong token.
//
//===----------------------------------------------------------------------===//
// IREE_ARCH_ARM_64 / aarch64
//===----------------------------------------------------------------------===//

// General features and high-level switches.
// FP_ARMV8: whether FP and NEON are enabled. Armv8 spec says they must agree.
IREE_CPU_FEATURE_BIT(ARM_64, 0, 0, FP_ARMV8, "fp-armv8")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 1, LSE, "lse")  // Armv8.1 atomics
IREE_CPU_FEATURE_BIT(ARM_64, 0, 2, LSE128, "lse128")  // Armv8.1 atomics

// SIMD features, not SVE-specific.
IREE_CPU_FEATURE_BIT(ARM_64, 0, 10, FP16, "fp16")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 11, FP16FML, "fp16fml")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 12, DOTPROD, "dotprod")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 13, I8MM, "i8mm")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 14, BF16, "bf16")

// SVE features.
IREE_CPU_FEATURE_BIT(ARM_64, 0, 20, SVE, "sve")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 21, SVE2, "sve2")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 22, SVE2P1, "sve2p1")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 23, SVE2_BITPERM, "sve2-bitperm")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 24, F32MM, "f32mm")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 25, F64MM, "f64mm")

// SME features.
IREE_CPU_FEATURE_BIT(ARM_64, 0, 30, SME, "sme")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 31, SME2, "sme2")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 32, SME2P1, "sme2p1")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 33, SME_F16F16, "sme-f16f16")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 34, SME_F64F64, "sme-f64f64")
IREE_CPU_FEATURE_BIT(ARM_64, 0, 35, SME_I16I64, "sme-i16i64")

//===----------------------------------------------------------------------===//
// IREE_ARCH_X86_64 / x86-64
//===----------------------------------------------------------------------===//

// SSE features. Note: SSE and SSE2 are mandatory parts of X86-64.
IREE_CPU_FEATURE_BIT(X86_64, 0, 0, SSE3, "sse3")
IREE_CPU_FEATURE_BIT(X86_64, 0, 1, SSSE3, "ssse3")
IREE_CPU_FEATURE_BIT(X86_64, 0, 2, SSE41, "sse4.1")
IREE_CPU_FEATURE_BIT(X86_64, 0, 3, SSE42, "sse4.2")
IREE_CPU_FEATURE_BIT(X86_64, 0, 4, SSE4A, "sse4a")

// AVX features.
IREE_CPU_FEATURE_BIT(X86_64, 0, 10, AVX, "avx")
IREE_CPU_FEATURE_BIT(X86_64, 0, 11, FMA, "fma")
IREE_CPU_FEATURE_BIT(X86_64, 0, 12, FMA4, "fma4")
IREE_CPU_FEATURE_BIT(X86_64, 0, 13, XOP, "xop")
IREE_CPU_FEATURE_BIT(X86_64, 0, 14, F16C, "f16c")
IREE_CPU_FEATURE_BIT(X86_64, 0, 15, AVX2, "avx2")

// AVX-512 features.
IREE_CPU_FEATURE_BIT(X86_64, 0, 20, AVX512F, "avx512f")
IREE_CPU_FEATURE_BIT(X86_64, 0, 21, AVX512CD, "avx512cd")
IREE_CPU_FEATURE_BIT(X86_64, 0, 22, AVX512VL, "avx512vl")
IREE_CPU_FEATURE_BIT(X86_64, 0, 23, AVX512DQ, "avx512dq")
IREE_CPU_FEATURE_BIT(X86_64, 0, 24, AVX512BW, "avx512bw")
IREE_CPU_FEATURE_BIT(X86_64, 0, 25, AVX512IFMA, "avx512ifma")
IREE_CPU_FEATURE_BIT(X86_64, 0, 26, AVX512VBMI, "avx512vbmi")
IREE_CPU_FEATURE_BIT(X86_64, 0, 27, AVX512VPOPCNTDQ, "avx512vpopcntdq")
IREE_CPU_FEATURE_BIT(X86_64, 0, 28, AVX512VNNI, "avx512vnni")
IREE_CPU_FEATURE_BIT(X86_64, 0, 29, AVX512VBMI2, "avx512vbmi2")
IREE_CPU_FEATURE_BIT(X86_64, 0, 30, AVX512BITALG, "avx512bitalg")
IREE_CPU_FEATURE_BIT(X86_64, 0, 31, AVX512BF16, "avx512bf16")
IREE_CPU_FEATURE_BIT(X86_64, 0, 32, AVX512FP16, "avx512fp16")

// AMX features.
IREE_CPU_FEATURE_BIT(X86_64, 0, 50, AMXTILE, "amx-tile")
IREE_CPU_FEATURE_BIT(X86_64, 0, 51, AMXINT8, "amx-int8")
IREE_CPU_FEATURE_BIT(X86_64, 0, 52, AMXBF16, "amx-bf16")

//===----------------------------------------------------------------------===//
// IREE_ARCH_RISCV_64 / riscv64
//===----------------------------------------------------------------------===//

// General features and high-level switches.
// RISCV vector extension. bit: 'V' - 'A' = 21
IREE_CPU_FEATURE_BIT(RISCV_64, 0, 21, RVV, "rvv")
