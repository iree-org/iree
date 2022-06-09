// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/utils/cpu_features.h"

#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_LINUX) && defined(IREE_ARCH_ARM_64)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(IREE_ARCH_ARM_64)
typedef enum {
  iree_cpu_features_aarch64_register_ID_AA64ISAR0_EL1,
  iree_cpu_features_aarch64_register_ID_AA64ISAR1_EL1,
  iree_cpu_features_aarch64_register_count
} iree_cpu_features_aarch64_register_t;

struct iree_cpu_features_t {
  uint64_t registers[iree_cpu_features_aarch64_register_count];
  bool registers_initialized[iree_cpu_features_aarch64_register_count];
};

// Returns true if aarch64 cpu feature registers are accessible.
static bool iree_devices_features_aarch64_register_access() {
#if defined(IREE_PLATFORM_LINUX) && defined(HWCAP_CPUID)
  // We are going to use access some special registers not accessible through
  // userspace. That is only going to work thanks to
  //   https://www.kernel.org/doc/html/latest/arm64/cpu-feature-registers.html
  // so we must first check if this feature is supported.
  //
  // Why not use this getauxval method to directly query the CPU features
  // that we care about, e.g. HWCAP2_I8MM? Because for each new CPU feature
  // that comes out, Linux needs to be updated to expose it in this way, and
  // sometimes we dont't want to wait for a device to receive an updated
  // Linux kernel before we can use the new feature. By only relying
  // on HWCAP_CPUID, we freeze once and for all our Linux kernel version
  // requirement to one that's already widespread.
  //
  // For the same reason, we're content to use the system-headers-provided
  // HWCAP_CPUID and #if-out this block if it's not defined, because if it
  // were not defined, at this point, that would mean we're compiling against
  // aging Linux headers so maybe we're not interested about recent CPU
  // features. But feel free to change that if you need to.
  //
  // Why query this everytime and not try to cache this in `features`?
  // Because getauxval is really cheap. It's just a memory read (auxv), no IO.
  // By contrast, what's expensive is the `mrs` instruction reading the actual
  // registers below, because that triggers a (would-be SIGILL) exception that
  // the kernel has to trap.
  return getauxval(AT_HWCAP) & HWCAP_CPUID;
#endif
  return false;
}

// Defines a local helper function returning the value of an aarch64 cpu feature
// register.
// clang-format off
#define IREE_CPU_FEATURES_AARCH64_READ_REGISTER(REG)               \
  static uint64_t iree_cpu_features_aarch64_read_##REG(void) {     \
    uint64_t retval = 0;                                           \
    asm("mrs %[dst], " #REG                                        \
      : /* outputs */                                              \
      [dst] "=r"(retval)                                           \
      : /* inputs */                                               \
      : /* clobbers */);                                           \
    return retval;                                                 \
  }
// clang-format on

// Define the access functions for the cpu feature registers that we need.
IREE_CPU_FEATURES_AARCH64_READ_REGISTER(ID_AA64ISAR0_EL1)
IREE_CPU_FEATURES_AARCH64_READ_REGISTER(ID_AA64ISAR1_EL1)

// Helper for iree_cpu_features_aarch64_get_register:
// Immediately reads the value of a cpu feature register, by enum index.
// Expensive!
static uint64_t iree_cpu_features_aarch64_read_register(
    iree_cpu_features_aarch64_register_t idx) {
  switch (idx) {
    case iree_cpu_features_aarch64_register_ID_AA64ISAR0_EL1:
      return iree_cpu_features_aarch64_read_ID_AA64ISAR0_EL1();
    case iree_cpu_features_aarch64_register_ID_AA64ISAR1_EL1:
      return iree_cpu_features_aarch64_read_ID_AA64ISAR1_EL1();
    case iree_cpu_features_aarch64_register_count:  // case, not default, so
      break;  // the compiler can warn if we forget an enum value.
  }
  assert(false && "bad iree_cpu_features_aarch64_register_t value");
  return 0;
}

// Returns the value of a cpu feature register. Caches the value in the
// iree_cpu_features_t struct.
static uint64_t iree_cpu_features_aarch64_get_register(
    iree_cpu_features_t* features, iree_cpu_features_aarch64_register_t idx) {
  if (!features->registers_initialized[idx]) {
    if (iree_devices_features_aarch64_register_access()) {
      features->registers[idx] = iree_cpu_features_aarch64_read_register(idx);
    } /* else: leave registers[idx] with its initial 0 bits */
    features->registers_initialized[idx] = true;
  }
  return features->registers[idx];
}

// Returns true if +dotprod is supported.
static bool iree_cpu_features_aarch64_dotprod(iree_cpu_features_t* features) {
  uint64_t register_value = iree_cpu_features_aarch64_get_register(
      features, iree_cpu_features_aarch64_register_ID_AA64ISAR0_EL1);
  return (register_value >> 44) & 1;
}

// Returns true if +i8mm is supported.
static bool iree_cpu_features_aarch64_i8mm(iree_cpu_features_t* features) {
  uint64_t register_value = iree_cpu_features_aarch64_get_register(
      features, iree_cpu_features_aarch64_register_ID_AA64ISAR1_EL1);
  return (register_value >> 52) & 1;
}

typedef iree_cpu_features_t iree_cpu_features_t;

#else  // not defined(IREE_ARCH_ARM_64)

// Not-implemented case. Decided in PR #8316 to make it non-empty just to avoid
// edge cases with empty structs.
struct iree_cpu_features_t {
  int unused;
};

#endif  // defined(IREE_ARCH_ARM_64)

iree_status_t iree_cpu_features_allocate(iree_allocator_t allocator,
                                         iree_cpu_features_t** cpu_features) {
  return iree_allocator_malloc(allocator, sizeof(iree_cpu_features_t),
                               (void**)cpu_features);
}

void iree_cpu_features_free(iree_allocator_t allocator,
                            iree_cpu_features_t* cpu_features) {
  iree_allocator_free(allocator, cpu_features);
}

iree_status_t iree_cpu_features_query(iree_cpu_features_t* cpu_features,
                                      iree_string_view_t feature,
                                      bool* out_result) {
  *out_result = false;

#ifdef IREE_ARCH_ARM_64
  if (iree_string_view_equal(feature, iree_make_cstring_view("+dotprod"))) {
    *out_result = iree_cpu_features_aarch64_dotprod(cpu_features);
    return iree_ok_status();
  }
  if (iree_string_view_equal(feature, iree_make_cstring_view("+i8mm"))) {
    *out_result = iree_cpu_features_aarch64_i8mm(cpu_features);
    return iree_ok_status();
  }
#endif  // IREE_ARCH_ARM_64

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unhandled CPU feature: '%.*s'", (int)feature.size,
                          feature.data);
}
