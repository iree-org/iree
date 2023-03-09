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

#define IREE_CPU_FEATURE_BIT_NAME(arch, field_index, bit_name) \
  IREE_CPU_DATA##field_index##_##arch##_##bit_name

// Bitmasks and values for processor data field 0.
enum iree_cpu_data_field_0_e {

#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  IREE_CPU_FEATURE_BIT_NAME(arch, field_index, bit_name) = 1ull << bit_pos,
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT

};

#endif  // IREE_SCHEMAS_CPU_DATA_H_
