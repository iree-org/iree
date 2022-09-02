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

  // Indicates support for Dot Product instructions.
  //
  // UDOT and SDOT instructions implemented.
  //
  // Source: ID_AA64ISAR0_EL1.DP [47:44] == 0b0001 / HWCAP_ASIMDDP
  // Canonical key: "dotprod"
  IREE_CPU_DATA_FIELD_0_AARCH64_HAVE_DOTPROD = 1ull << 0,

  // Indicates support for Advanced SIMD and Floating-point Int8 matrix
  // multiplication instructions.
  //
  // SMMLA, SUDOT, UMMLA, USMMLA, and USDOT instructions are implemented.
  //
  // Source: ID_AA64ISAR1_EL1.I8MM [55:52] == 0b0001 / HWCAP2_I8MM
  // Canonical key: "i8mm"
  IREE_CPU_DATA_FIELD_0_AARCH64_HAVE_I8MM = 1ull << 1,

};

#endif  // IREE_SCHEMAS_CPU_DATA_H_
