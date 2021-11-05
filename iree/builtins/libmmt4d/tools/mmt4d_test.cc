// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/libmmt4d/mmt4d.h"

#include <cstring>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

template <int M0, int K0, int N0, typename lhs_t, typename rhs_t,
          typename dst_t, int K_SIZE = 4 * K0>
struct TestState {
  size_t k_size = K_SIZE;
  lhs_t lhs[K_SIZE * M0] = {0};
  rhs_t rhs[K_SIZE * N0] = {0};
  dst_t ref[M0 * N0] = {0};
  dst_t dst[M0 * N0] = {0};

  TestState() {
    for (int i = 0; i < k_size * M0; i++) {
      lhs[i] = DummyRandom() % 5;
    }
    for (int i = 0; i < k_size * N0; i++) {
      rhs[i] = DummyRandom() % 6;
    }
    ReferenceMMT4D(k_size, lhs, rhs, ref);
  }

  static uint32_t DummyRandom() {
    static uint32_t state = 0;
    state = (state * 123 + 456) % 321;
    return state;
  }

  static void ReferenceMMT4D(int k_size, const lhs_t* lhs, const rhs_t* rhs,
                             dst_t* dst) {
    dst_t acc[M0 * N0] = {0};
    for (int k = 0; k < k_size; k += K0) {
      for (int m0 = 0; m0 < M0; m0++) {
        for (int n0 = 0; n0 < N0; n0++) {
          dst_t a = 0;
          for (int k0 = 0; k0 < K0; k0++) {
            a += lhs[m0 * K0 + k0] * rhs[n0 * K0 + k0];
          }
          acc[m0 * N0 + n0] += a;
        }
      }
      lhs += M0 * K0;
      rhs += N0 * K0;
    }
    std::memcpy(dst, acc, M0 * N0 * sizeof(*dst));
  }

  template <typename element_t>
  static void DumpMatrix(int r, int c, const element_t* data) {
    for (int m = 0; m < r; m++) {
      for (int n = 0; n < c; n++) {
        std::cerr << (int)data[m * c + n] << ' ';
      }
      std::cerr << std::endl;
    }
  }

  bool AssertEqual() {
    if (std::memcmp(ref, dst, sizeof(dst)) == 0) return true;
    std::cerr << "LHS:" << std::endl;
    DumpMatrix(K_SIZE, M0, lhs);
    std::cerr << "RHS:" << std::endl;
    DumpMatrix(K_SIZE, N0, lhs);
    std::cerr << "Expected:" << std::endl;
    DumpMatrix(M0, N0, ref);
    std::cerr << "Actual:" << std::endl;
    DumpMatrix(M0, N0, dst);
    return false;
  }
};

TEST(MMT4DTest, mmt4d_8x4x8_i8i8i32) {
  TestState<8, 4, 8, int8_t, int8_t, int32_t> test_state;
  mmt4d_8x4x8_i8i8i32(test_state.k_size, test_state.lhs, test_state.rhs,
                      test_state.dst);
  ASSERT_TRUE(test_state.AssertEqual());
}
