// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/tools/util.h"

#include "iree/base/internal/cpu.h"
#include "iree/builtins/ukernel/tools/test.h"
#include "iree/schemas/cpu_data.h"

static void iree_uk_test_make_cpu_data_for_features_case(
    iree_uk_test_t* test, const char* cpu_features,
    const iree_uk_uint64_t* expected) {
  iree_uk_uint64_t actual[IREE_CPU_DATA_FIELD_COUNT] = {0};
  iree_uk_make_cpu_data_for_features(cpu_features, actual);
  for (int i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    if (actual[i] != expected[i]) {
      IREE_UK_TEST_FAIL(test);
    }
  }
}

static void iree_uk_test_make_cpu_data_for_features(iree_uk_test_t* test,
                                                    const void* params) {
  (void)params;
  // Special CPU feature strings understood across architectures.
  iree_uk_uint64_t expected[IREE_CPU_DATA_FIELD_COUNT] = {0};
  iree_uk_test_make_cpu_data_for_features_case(test, "", expected);
  iree_uk_test_make_cpu_data_for_features_case(
      test, "host", (const iree_uk_uint64_t*)iree_cpu_data_fields());

#if defined(IREE_UK_ARCH_X86_64)
  // Individual x86-64 features.
  expected[0] = IREE_CPU_DATA0_X86_64_AVX;
  iree_uk_test_make_cpu_data_for_features_case(test, "avx", expected);
  // Comma-separated lists of x86-64 features.
  expected[0] = IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_AVX2 |
                IREE_CPU_DATA0_X86_64_FMA;
  iree_uk_test_make_cpu_data_for_features_case(test, "avx,avx2,fma", expected);
  // Named x86-64 feature sets.
  iree_uk_uint64_t avx2_fma =
      IREE_CPU_DATA0_X86_64_AVX2 | IREE_CPU_DATA0_X86_64_FMA;
  iree_uk_uint64_t avx512_base =
      avx2_fma | IREE_CPU_DATA0_X86_64_AVX512F |
      IREE_CPU_DATA0_X86_64_AVX512BW | IREE_CPU_DATA0_X86_64_AVX512DQ |
      IREE_CPU_DATA0_X86_64_AVX512VL | IREE_CPU_DATA0_X86_64_AVX512CD;
  iree_uk_uint64_t avx512_vnni = avx512_base | IREE_CPU_DATA0_X86_64_AVX512VNNI;
  expected[0] = avx2_fma;
  iree_uk_test_make_cpu_data_for_features_case(test, "avx2_fma", expected);
  expected[0] = avx512_base;
  iree_uk_test_make_cpu_data_for_features_case(test, "avx512_base", expected);
  expected[0] = avx512_vnni;
  iree_uk_test_make_cpu_data_for_features_case(test, "avx512_vnni", expected);

#elif defined(IREE_UK_ARCH_ARM_64)
  // Individual arm64 features.
  expected[0] = IREE_CPU_DATA0_ARM_64_DOTPROD;
  iree_uk_test_make_cpu_data_for_features_case(test, "dotprod", expected);
  // Comma-separated lists of arm features.
  expected[0] = IREE_CPU_DATA0_ARM_64_DOTPROD | IREE_CPU_DATA0_ARM_64_I8MM;
  iree_uk_test_make_cpu_data_for_features_case(test, "dotprod,i8mm", expected);
  // Named arm64 feature sets: none at the moment.

#endif  // defined(IREE_UK_ARCH_X86_64)
}

int main(int argc, char** argv) {
  iree_uk_test("make_cpu_data_for_features",
               iree_uk_test_make_cpu_data_for_features, NULL, "");
  return EXIT_SUCCESS;  // failures are fatal
}
