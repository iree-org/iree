// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/pack.h"

#include <cstring>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/builtins/ukernel/tools/ukernel_test_utils.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

static void iree_pack_reference(const iree_uk_pack_params_t& params) {
  // For now, the input and output element types are always the same.
  iree_uk_type_t elem_type = iree_uk_pack_in_type(params.type);
  iree_uk_ssize_t elem_size = iree_uk_type_size(elem_type);
  iree_uk_ssize_t lsize0 = params.out_size0;
  iree_uk_ssize_t lsize1 = params.out_size1;
  iree_uk_ssize_t lsize2 = params.out_size2;
  iree_uk_ssize_t lsize3 = params.out_size3;
  iree_uk_ssize_t out_stride_l0 = params.out_stride0;
  iree_uk_ssize_t out_stride_l1 = params.out_size3 * params.out_size2;
  iree_uk_ssize_t out_stride_l2 = params.out_size3;
  iree_uk_ssize_t out_stride_l3 = 1;
  if (params.flags & IREE_UK_FLAG_PACK_TRANSPOSE_OUTER) {
    std::swap(lsize0, lsize1);
    std::swap(out_stride_l0, out_stride_l1);
  }
  if (params.flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER) {
    std::swap(lsize2, lsize3);
    std::swap(out_stride_l2, out_stride_l3);
  }
  assert(lsize0 * lsize2 == params.in_size0);
  assert(lsize1 * lsize3 == params.in_size1);
  for (iree_uk_ssize_t l0 = 0; l0 < lsize0; ++l0) {
    for (iree_uk_ssize_t l2 = 0; l2 < lsize2; ++l2) {
      for (iree_uk_ssize_t l1 = 0; l1 < lsize1; ++l1) {
        for (iree_uk_ssize_t l3 = 0; l3 < lsize3; ++l3) {
          iree_uk_ssize_t out_offset = l0 * out_stride_l0 + l2 * out_stride_l2 +
                                       l1 * out_stride_l1 + l3 * out_stride_l3;
          iree_uk_ssize_t i0 = l0 * lsize2 + l2;
          iree_uk_ssize_t i1 = l1 * lsize3 + l3;
          char* out_ptr = ((char*)params.out_buffer) + out_offset * elem_size;
          if (i0 >= params.in_size0 || i1 >= params.in_size1) {
            memcpy(out_ptr, params.padding_value, elem_size);
          } else {
            iree_uk_ssize_t in_offset = i1 + i0 * params.in_stride0;
            const char* in_ptr =
                ((char*)params.in_buffer) + in_offset * elem_size;
            memcpy(out_ptr, in_ptr, elem_size);
          }
        }
      }
    }
  }
}

static void test_one_pack_using_given_input(
    const iree_uk_pack_params_t& shared_params,
    iree_uk_test_random_engine_t* engine) {
  assert(!shared_params.out_buffer);

  iree_uk_pack_params_t reference_params;
  memcpy(&reference_params, &shared_params, sizeof shared_params);
  iree_uk_type_t out_type = iree_uk_pack_out_type(shared_params.type);
  iree_uk_ssize_t out_buffer_size = iree_uk_test_2d_buffer_length(
      out_type, shared_params.out_size0, shared_params.out_stride0);
  reference_params.out_buffer = malloc(out_buffer_size);
  iree_uk_test_write_random_buffer(reference_params.out_buffer, out_buffer_size,
                                   out_type, engine);

  iree_uk_pack_params_t actual_params;
  memcpy(&actual_params, &shared_params, sizeof shared_params);
  actual_params.out_buffer = malloc(out_buffer_size);
  iree_uk_test_write_random_buffer(actual_params.out_buffer, out_buffer_size,
                                   out_type, engine);

  iree_pack_reference(reference_params);
  iree_uk_status_t status = iree_uk_pack(&actual_params);
  if (status != iree_uk_status_ok) {
    fprintf(stderr, "FATAL: iree_uk_pack failed: %s\n",
            iree_uk_status_message(status));
    iree_abort();
  }

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
    fprintf(stderr, "pack test failure with the following params:\n");
    char types_str[32];
    iree_uk_test_type_pair_str(types_str, sizeof types_str, p.type);
    fprintf(stderr, "  types: %s\n", types_str);
    fprintf(stderr, "  flags: transpose_inner=%d, transpose_outer=%d\n",
            (bool)(p.flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER),
            (bool)(p.flags & IREE_UK_FLAG_PACK_TRANSPOSE_OUTER));
    fprintf(stderr, "  input shape: %dx%d\n", (int)p.in_size0, (int)p.in_size1);
    fprintf(stderr, "  output shape: %dx%dx%dx%d\n", (int)p.out_size0,
            (int)p.out_size1, (int)p.out_size2, (int)p.out_size3);
    fprintf(stderr, "  input stride: %d\n", (int)p.in_stride0);
    fprintf(stderr, "  output stride: %d\n", (int)p.out_stride0);
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

static void test_one_pack_creating_input_for_given_shape(
    const iree_uk_pack_params_t& shared_params,
    iree_uk_test_random_engine_t* engine) {
  iree_uk_pack_params_t params;
  memcpy(&params, &shared_params, sizeof params);
  assert(!params.in_buffer);
  assert(!params.out_buffer);
  assert(!params.in_stride0);
  assert(!params.out_stride0);
  // Populate strides first - we need them below to compute buffer lengths.
  // Randomly make strides either tight or not to exercise all cases.
  params.in_stride0 =
      params.in_size1 + iree_uk_test_random_engine_get_0_or_1(engine);
  params.out_stride0 = params.out_size1 * params.out_size2 * params.out_size3;
  iree_uk_test_random_engine_get_0_or_1(engine);
  iree_uk_type_t in_type = iree_uk_pack_in_type(params.type);
  iree_uk_ssize_t in_buffer_size = iree_uk_test_2d_buffer_length(
      in_type, params.in_size0, params.in_stride0);
  void* in_buffer = malloc(in_buffer_size);
  iree_uk_test_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  params.in_buffer = in_buffer;
  test_one_pack_using_given_input(params, engine);
  free(in_buffer);
}

static void pack_test(const iree_uk_pack_type_t& type) {
  iree_uk_test_random_engine_t* engine = iree_uk_test_random_engine_create();
  struct untransposed_out_shape_t {
    int size0, size1, size2, size3;
  };
  std::vector<untransposed_out_shape_t> untransposed_out_shapes{
      // Degenerate cases. Vacuous.
      {0, 1, 1, 1},
      {1, 0, 1, 1},
      // Non-degenerate cases.
      {1, 1, 1, 1},
      {2, 2, 2, 2},
      {3, 3, 2, 2},
      {2, 2, 3, 3},
      {2, 3, 2, 3},
      {11, 13, 7, 5},
      {4, 8, 16, 32},
  };
  for (const auto& shape : untransposed_out_shapes) {
    for (bool transpose_inner : {false, true}) {
      for (bool transpose_outer : {false, true}) {
        iree_uk_pack_params_t params = {};
        params.type = type;
        params.in_size0 = shape.size0 * shape.size2;
        params.in_size1 = shape.size1 * shape.size3;
        params.out_size0 = shape.size0;
        params.out_size1 = shape.size1;
        params.out_size2 = shape.size2;
        params.out_size3 = shape.size3;
        params.flags = 0;
        if (transpose_outer) {
          params.flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
          std::swap(params.out_size0, params.out_size1);
        }
        if (transpose_inner) {
          params.flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
          std::swap(params.out_size2, params.out_size3);
        }
        test_one_pack_creating_input_for_given_shape(params, engine);
      }
    }
  }
  iree_uk_test_random_engine_destroy(engine);
}

TEST(PackTest, f32f32) { pack_test(iree_uk_pack_type_f32f32); }

TEST(PackTest, i8i8) { pack_test(iree_uk_pack_type_i8i8); }

TEST(PackTest, i32i32) { pack_test(iree_uk_pack_type_i32i32); }
