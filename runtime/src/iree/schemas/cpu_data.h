// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SCHEMAS_CPU_DATA_H_
#define IREE_SCHEMAS_CPU_DATA_H_

//===----------------------------------------------------------------------===//
// CPU processor data field values
//===----------------------------------------------------------------------===//
// IREE treats architecture-specific processor information as an opaque set of
// 64-bit values, each of which encodes fields of interest to the generated
// code from the compiler and the runtime. This file is included from both the
// compiler, runtime, and compiler-generated binaries and must have zero
// includes and unconditionally define all values to enable the compiler to
// target non-host architectures.
//
// The format of the data is architecture-specific as by construction no value
// will ever be used in a compiled binary from another architecture. This
// allows us to simplify this interface as we can't for example load the same
// executable library for both aarch64 on riscv32 and don't need to normalize
// any of the fields across them both.
//
// As the values of the fields are encoded in generated binaries their meaning
// here cannot change once deployed: only new meaning for unused bits can be
// specified, and zeros must always be assumed to be unknown/undefined/false.
// Since IREE executables can run on many architectures and operating systems we
// also cannot directly expose the architecture-specific registers as not all
// environments and access-levels can query them.
//
// On a best-effort basis, we try to pack the most commonly used values in data
// field 0, and we try to have some consistency in the bit allocation: ideally,
// ISA extensions that are either closely related or from the same era should
// occupy bits close to each other, if only so that the bit values enumedated
// below from lowest to highest bits are easier to read (e.g. look up at a
// glance which AVX512 features we already have bits for). Inevitably, the
// aforementioned requirement that bits are set in stone, will force us away
// from that at times. To strike a decent compromise, we typically try to
// reserve some range of bits for families or eras of ISA extensions, but don't
// overthink it.
//
// This is similar in functionality to getauxval(AT_HWCAP*) in linux but
// platform-independent and with additional fields and values that may not yet
// be available in cpufeature.h. AT_HWCAP stores only bit flags but we can
// store any bit-packed scalar value as well such as cache sizes.
//
// Wherever possible compile-time checks should be used instead to allow for
// smaller and more optimizable code. For example if compiling for armv8.6+ the
// I8MM feature will always be available and does not need to be tested.
//
// NOTE: when adding values with more than 1 bit define both a shift amount
// and mask used for selecting the bits from the field. An example of where this
// could be used is a field that contains one bit per physical processor
// indicating whether it is big/LITTLE that can be indexed by the processor ID.

// Number of 64-bit data values captured.
// The current value gives us hundreds of bits to work with and can be extended
// in the future.
#define IREE_CPU_DATA_FIELD_COUNT 8

// Bitmasks and values for processor data field 0.
enum iree_cpu_data_field_0_e {

  //===--------------------------------------------------------------------===//
  // IREE_ARCH_ARM_64 / aarch64
  //===--------------------------------------------------------------------===//

  // TODO: add several common ARM ISA extensions and allocate some ranges of
  // bits for some families/eras. If we just start out with bits 0 and 1
  // allocated for dotprod and i8mm, we are quickly going to have a hard-to-read
  // enumeration here.
  IREE_CPU_DATA0_ARM_64_DOTPROD = 1ull << 0,
  IREE_CPU_DATA0_ARM_64_I8MM = 1ull << 1,

  //===--------------------------------------------------------------------===//
  // IREE_ARCH_X86_64 / x86-64
  //===--------------------------------------------------------------------===//

  // SSE features. Note: SSE and SSE2 are mandatory parts of X86-64.
  IREE_CPU_DATA0_X86_64_SSE3 = 1ull << 0,
  IREE_CPU_DATA0_X86_64_SSSE3 = 1ull << 1,
  IREE_CPU_DATA0_X86_64_SSE41 = 1ull << 2,
  IREE_CPU_DATA0_X86_64_SSE42 = 1ull << 3,
  IREE_CPU_DATA0_X86_64_SSE4A = 1ull << 4,

  // AVX features.
  IREE_CPU_DATA0_X86_64_AVX = 1ull << 10,
  IREE_CPU_DATA0_X86_64_FMA3 = 1ull << 11,
  IREE_CPU_DATA0_X86_64_FMA4 = 1ull << 12,
  IREE_CPU_DATA0_X86_64_XOP = 1ull << 13,
  IREE_CPU_DATA0_X86_64_F16C = 1ull << 14,
  IREE_CPU_DATA0_X86_64_AVX2 = 1ull << 15,

  // AVX-512 features.
  IREE_CPU_DATA0_X86_64_AVX512F = 1ull << 20,
  IREE_CPU_DATA0_X86_64_AVX512CD = 1ull << 21,
  IREE_CPU_DATA0_X86_64_AVX512VL = 1ull << 22,
  IREE_CPU_DATA0_X86_64_AVX512DQ = 1ull << 23,
  IREE_CPU_DATA0_X86_64_AVX512BW = 1ull << 24,
  IREE_CPU_DATA0_X86_64_AVX512IFMA = 1ull << 25,
  IREE_CPU_DATA0_X86_64_AVX512VBMI = 1ull << 26,
  IREE_CPU_DATA0_X86_64_AVX512VPOPCNTDQ = 1ull << 27,
  IREE_CPU_DATA0_X86_64_AVX512VNNI = 1ull << 28,
  IREE_CPU_DATA0_X86_64_AVX512VBMI2 = 1ull << 29,
  IREE_CPU_DATA0_X86_64_AVX512BITALG = 1ull << 30,
  IREE_CPU_DATA0_X86_64_AVX512BF16 = 1ull << 31,
  IREE_CPU_DATA0_X86_64_AVX512FP16 = 1ull << 32,

  // AMX features.
  IREE_CPU_DATA0_X86_64_AMXTILE = 1ull << 50,
  IREE_CPU_DATA0_X86_64_AMXINT8 = 1ull << 51,
  IREE_CPU_DATA0_X86_64_AMXBF16 = 1ull << 52,

};

#endif  // IREE_SCHEMAS_CPU_DATA_H_
