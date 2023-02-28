// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Design rationale and code creep warning!
//
// Summary:
//
//   The goal of this test is to provide 100% coverage across all
//   internal kernel variants, which is not convenient to do in e2e tests.
//   Resist the temptation to reimplement here all the niceties of the e2e test.
//   Stick to guaranteeing that if the test succeeds, then the mmt4d builtin,
//   with all its asm code path variants, is correct. In case of failure, the
//   user is expected to be happy to jump into a debugger.
//
// Longer story:
//
// It is said by an ancient prophecy that all matrix multiplication tests grow
// to be thousands of lines of code.
//
// In fact, we already have one, it's the end-to-end matmul test under
// iree/tests/e2e/matmul. That one is needed anyway, and needs to be large
// anyway, being end-to-end and applying to all target backends, including those
// where device!=host. And so it makes sense for that one to have extra bells
// and whistles such as fuzzy comparisons, pretty-printing of numerical errors
// to aid debugging, and yet more special logic to make numerical errors easier
// to debug.
//
// Let's not duplicate all that here! Note also that, tempting as it would
// be to borrow the matrix-pretty-printing stuff from e2e/matmul, that applies
// to plain row-major 2D matrices, while here we are dealing with 4D arrays /
// tiled-layout matrices. Trying to bridge over that difference would bring yet
// more complexity.
//
// Instead, let us keep a sharp focus on why we need this separate micro test.
// The motivation is not the usual "because micro tests are easier to debug than
// e2e" but rather because it would be difficult to have 100% code coverage in
// e2e. There are many variants of mmt4d builtin ukernels for various CPU
// features and tuned for various CPU models. We have to iterate over all these
// variants. Trying to do so in e2e tests would require exposing knobs for
// things that we would otherwise prefer to keep internal in the mmt4d builtin
// implementation, and would make e2e/matmul tests even more expensive.

#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/tools/ukernel_test_utils.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

template <typename lhs_t, typename rhs_t, typename out_t>
static void iree_mmt4d_reference(const iree_uk_mmt4d_params_t& params) {
  bool accumulate = params.flags & IREE_UK_FLAG_ACCUMULATE;
  iree_uk_ssize_t lhs_tile_size = params.M0 * params.K0;
  iree_uk_ssize_t rhs_tile_size = params.N0 * params.K0;
  iree_uk_ssize_t out_tile_size = params.M0 * params.N0;
  for (iree_uk_ssize_t i = 0; i < params.M; ++i) {
    for (iree_uk_ssize_t j = 0; j < params.N; ++j) {
      out_t* out_tile_ptr = ((out_t*)params.out_buffer) +
                            i * params.out_stride + j * out_tile_size;
      const lhs_t* lhs_panel_ptr =
          ((const lhs_t*)params.lhs_buffer) + i * params.lhs_stride;
      const rhs_t* rhs_panel_ptr =
          ((const rhs_t*)params.rhs_buffer) + j * params.rhs_stride;
      for (iree_uk_ssize_t i0 = 0; i0 < params.M0; ++i0) {
        for (iree_uk_ssize_t j0 = 0; j0 < params.N0; ++j0) {
          const lhs_t* lhs_tile_ptr = lhs_panel_ptr;
          const rhs_t* rhs_tile_ptr = rhs_panel_ptr;
          out_t* out_ptr = out_tile_ptr + i0 * params.N0 + j0;
          out_t acc = accumulate ? *out_ptr : 0.f;
          for (iree_uk_ssize_t k = 0; k < params.K; ++k) {
            for (iree_uk_ssize_t k0 = 0; k0 < params.K0; ++k0) {
              out_t lhs_val = lhs_tile_ptr[i0 * params.K0 + k0];
              out_t rhs_val = rhs_tile_ptr[j0 * params.K0 + k0];
              acc += lhs_val * rhs_val;
            }
            lhs_tile_ptr += lhs_tile_size;
            rhs_tile_ptr += rhs_tile_size;
          }
          *out_ptr = acc;
        }
      }
    }
  }
}

static void iree_mmt4d_reference(const iree_uk_mmt4d_params_t& params) {
  switch (params.type) {
    case iree_uk_mmt4d_type_f32f32f32:
      iree_mmt4d_reference<float, float, float>(params);
      break;
    case iree_uk_mmt4d_type_i8i8i32:
      iree_mmt4d_reference<iree_uk_int8_t, iree_uk_int8_t, iree_uk_int32_t>(
          params);
      break;
    default:
      assert(false && "unknown type");
  }
}

static void test_one_matmul_using_given_lhs_rhs(
    const iree_uk_mmt4d_params_t& shared_params,
    iree_uk_test_random_engine_t* engine) {
  assert(!shared_params.out_buffer);

  iree_uk_mmt4d_params_t reference_params;
  memcpy(&reference_params, &shared_params, sizeof shared_params);
  iree_uk_type_t out_type = iree_uk_mmt4d_out_type(shared_params.type);
  iree_uk_ssize_t out_buffer_size = iree_uk_test_2d_buffer_length(
      out_type, shared_params.M, shared_params.out_stride);
  reference_params.out_buffer = malloc(out_buffer_size);
  iree_uk_test_write_random_buffer(reference_params.out_buffer, out_buffer_size,
                                   out_type, engine);

  iree_uk_mmt4d_params_t actual_params;
  memcpy(&actual_params, &shared_params, sizeof shared_params);
  actual_params.out_buffer = malloc(out_buffer_size);
  memcpy(actual_params.out_buffer, reference_params.out_buffer,
         out_buffer_size);

  iree_mmt4d_reference(reference_params);
  iree_uk_mmt4d(&actual_params);

  // For now we use exact comparisons, even for float, even though the reference
  // code accumulates in a different order compared to the actual code. This
  // relies on picking input test matrix elements so that all intermediate
  // values are exactly representable - i.e. small integer numerators. This
  // become problematic when we do float16. See the comment at the top of this
  // file explaining how we refrain from letting this grow into a 1000-line-long
  // fully-featured test.
  if (memcmp(actual_params.out_buffer, reference_params.out_buffer,
             out_buffer_size)) {
    const auto& p = actual_params;
    fprintf(stderr, "mmt4d test failure with the following params:\n");
    char types_str[32];
    iree_uk_test_type_triple_str(types_str, sizeof types_str, p.type);
    fprintf(stderr, "  types: %s\n", types_str);
    fprintf(stderr, "  flags: accumulate=%d\n",
            (bool)(p.flags & IREE_UK_FLAG_ACCUMULATE));
    fprintf(stderr, "  M=%d, N=%d, K=%d\n", (int)p.M, (int)p.N, (int)p.K);
    fprintf(stderr, "  M0=%d, N0=%d, K0=%d\n", (int)p.M0, (int)p.N0, (int)p.K0);
    fprintf(stderr, "  lhs_stride=%zu, rhs_stride=%zu, out_stride=%zu\n",
            (size_t)p.lhs_stride, (size_t)p.rhs_stride, (size_t)p.out_stride);
    char cpu_feat_str[32];
    iree_uk_test_cpu_features_str(cpu_feat_str, sizeof cpu_feat_str, p.cpu_data,
                                  1);
    fprintf(stderr, "  cpu features: %s\n", cpu_feat_str);
    // Don't even try to pretty-print matrices. See the comment at the top of
    // this file. Don't try to use GTest primitives to show expected vs actual
    // since that would require dispatching to type-specific code paths.
    // Also, at this point it's easy for the user to rerun this test
    // in a debugger and manually inspect values.
    //
    // We want fatal here - that is what the user running this in a debugger
    // wants us to do, so they can inspect values while they exist in memory.
    // What's the GTest-sanctioned fatal error? GTEST_FAIL() has a comment that
    // says that it's fatal, but that's a lie at least here on Android.
    iree_abort();
  }

  free(reference_params.out_buffer);
  free(actual_params.out_buffer);
}

static void test_one_matmul_creating_lhs_rhs_for_given_shape(
    const iree_uk_mmt4d_params_t& shared_params,
    iree_uk_test_random_engine_t* engine) {
  iree_uk_mmt4d_params_t params;
  memcpy(&params, &shared_params, sizeof params);
  assert(!params.lhs_buffer);
  assert(!params.rhs_buffer);
  assert(!params.out_buffer);
  assert(!params.lhs_stride);
  assert(!params.rhs_stride);
  assert(!params.out_stride);
  // Populate strides first - we need them below to compute buffer lengths.
  // Randomly make strides either tight or not to exercise all cases.
  params.lhs_stride = params.K * params.M0 * params.K0 +
                      iree_uk_test_random_engine_get_0_1(engine);
  params.rhs_stride = params.K * params.N0 * params.K0 +
                      iree_uk_test_random_engine_get_0_1(engine);
  params.out_stride = params.N * params.M0 * params.N0 +
                      iree_uk_test_random_engine_get_0_1(engine);
  iree_uk_type_t lhs_type = iree_uk_mmt4d_lhs_type(params.type);
  iree_uk_type_t rhs_type = iree_uk_mmt4d_rhs_type(params.type);
  iree_uk_ssize_t lhs_buffer_size =
      iree_uk_test_2d_buffer_length(lhs_type, params.M, params.lhs_stride);
  iree_uk_ssize_t rhs_buffer_size =
      iree_uk_test_2d_buffer_length(rhs_type, params.N, params.rhs_stride);
  void* lhs_buffer = malloc(lhs_buffer_size);
  void* rhs_buffer = malloc(rhs_buffer_size);
  iree_uk_test_write_random_buffer(lhs_buffer, lhs_buffer_size, lhs_type,
                                   engine);
  iree_uk_test_write_random_buffer(rhs_buffer, rhs_buffer_size, rhs_type,
                                   engine);
  params.lhs_buffer = lhs_buffer;
  params.rhs_buffer = rhs_buffer;
  test_one_matmul_using_given_lhs_rhs(params, engine);
  free(lhs_buffer);
  free(rhs_buffer);
}

static void test_matmuls_for_various_MNK_shapes_and_flags(
    const iree_uk_mmt4d_params_t& shared_params,
    iree_uk_test_random_engine_t* engine) {
  iree_uk_mmt4d_params_t params;
  memcpy(&params, &shared_params, sizeof params);
  assert(params.M == 0);
  assert(params.N == 0);
  assert(params.K == 0);
  assert(params.flags == 0);
  struct shape_mnk_t {
    int m, n, k;
  };
  std::vector<shape_mnk_t> shapes{
      // Degenerate case M==0. Vacuous.
      {0, 1, 1},
      {0, 5, 7},
      // Degenerate case N==0. Vacuous.
      {1, 0, 1},
      {5, 0, 7},
      // Degenerate case K==0. Vacuous if flags have ACCUMULATE. Zeroing the
      // output buffer otherwise.
      {1, 1, 0},
      {5, 7, 0},
      // Non-degenerate cases.
      {1, 1, 1},
      {1, 1, 2},
      {1, 1, 10},
      {1, 1, 1000},
      {2, 1, 1},
      {1, 2, 1},
      {2, 2, 2},
      {5, 7, 13},
  };
  for (shape_mnk_t shape : shapes) {
    params.M = shape.m;
    params.N = shape.n;
    params.K = shape.k;
    for (bool accumulate : {false, true}) {
      params.flags = accumulate ? IREE_UK_FLAG_ACCUMULATE : 0;
      test_one_matmul_creating_lhs_rhs_for_given_shape(params, engine);
    }
  }
}

// Tests mmt4d with the specific data type and specific M0xN0xK0 tile format.
// If cpu_data_field_0_bit is nonzero, it must then be a single bit (power of 2)
// and if the CPU supports the corresponding feature, the mmt4d tests are run a
// second time with that CPU feature enabled.
static void mmt4d_test(iree_uk_mmt4d_type_t type, int M0, int N0, int K0,
                       iree_uk_uint64_t cpu_data_field_0_bit) {
  // Letting each test create its own engine makes them independent: a testcase
  // succeeds or fails the same way if we isolate it or reorder it. The
  // potential downside of repeating the same pseudorandom sequence is OK
  // because any pseudorandom sequence should be equally good at coverage, and
  // different testcases tend to use different tile shapes anyway.
  iree_uk_test_random_engine_t* engine = iree_uk_test_random_engine_create();
  iree_uk_mmt4d_params_t params;
  memset(&params, 0, sizeof params);
  params.type = type;
  params.M0 = M0;
  params.N0 = N0;
  params.K0 = K0;
  const iree_uk_uint64_t local_cpu_data_default[IREE_CPU_DATA_FIELD_COUNT] = {
      0};
  params.cpu_data = local_cpu_data_default;
  // First try without any optional CPU feature. This matters even when the
  // feature is supported by the CPU because we want to test the fallback to
  // architecture-default or generic code.
  test_matmuls_for_various_MNK_shapes_and_flags(params, engine);
  // If this is nonzero, we are asked to test again with this CPU feature.
  if (cpu_data_field_0_bit) {
    const iree_uk_uint64_t local_cpu_data_with_bit[IREE_CPU_DATA_FIELD_COUNT] =
        {cpu_data_field_0_bit};
    params.cpu_data = local_cpu_data_with_bit;
    // Check if the CPU supports the feature (otherwise, we crash).
    bool supported = iree_cpu_data_field(0) & params.cpu_data[0];
    char cpu_feat_str[32];
    iree_uk_test_cpu_features_str(cpu_feat_str, sizeof cpu_feat_str,
                                  params.cpu_data, 1);
    if (supported) {
      // Run with the optional CPU feature.
      printf("Device supports CPU feature: %s\n", cpu_feat_str);
      test_matmuls_for_various_MNK_shapes_and_flags(params, engine);
    } else {
      printf("Skipped: device does not support CPU feature: %s\n",
             cpu_feat_str);
    }
  }
  iree_uk_test_random_engine_destroy(engine);
}

#define MMT4D_TEST(type, M0, N0, K0, test_suffix, feature_bit)      \
  TEST(Mmt4dTest, type##_tile_##M0##x##N0##x##K0##_##test_suffix) { \
    mmt4d_test(iree_uk_mmt4d_type_##type, M0, N0, K0, feature_bit); \
  }

// Generic tests, not matching any particular CPU feature. This is the place to
// test weird M0, N0, K0 to ensure e.g. that we haven't unwittingly baked in a
// power-of-two assumption
MMT4D_TEST(f32f32f32, 3, 5, 7, generic, 0)
MMT4D_TEST(i8i8i32, 9, 6, 3, generic, 0)

// ARM_64 tests.
#if defined(IREE_UK_ARCH_ARM_64)

#define MMT4D_ARM_64_TEST(type, M0, N0, K0) \
  MMT4D_TEST(type, M0, N0, K0, arm_64, 0)

#define MMT4D_ARM_64_TEST_WITH_CPU_FEATURE(type, M0, N0, K0, FEATURE) \
  MMT4D_TEST(type, M0, N0, K0, arm_64_##FEATURE,                      \
             IREE_CPU_DATA0_ARM_64_##FEATURE)

MMT4D_ARM_64_TEST(f32f32f32, 8, 8, 1)
MMT4D_ARM_64_TEST(i8i8i32, 8, 8, 1)
MMT4D_ARM_64_TEST_WITH_CPU_FEATURE(i8i8i32, 8, 8, 4, DOTPROD)
MMT4D_ARM_64_TEST_WITH_CPU_FEATURE(i8i8i32, 8, 8, 8, I8MM)
#endif  // defined(IREE_UK_ARCH_ARM_64)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iree_cpu_initialize(iree_allocator_system());
  return RUN_ALL_TESTS();
}
