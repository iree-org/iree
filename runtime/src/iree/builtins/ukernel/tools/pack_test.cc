// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/tools/ukernel_test_utils.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

static void iree_pack_reference(const iree_uk_pack_params_t& params) {
  // For now, the input and output element types are always the same.
  iree_uk_type_t elem_type = iree_uk_pack_in_type(params.type);
  iree_uk_ssize_t elem_size = iree_uk_type_size(elem_type);
  iree_uk_ssize_t outer_size0 = params.out_size0;
  iree_uk_ssize_t outer_size1 = params.out_size1;
  iree_uk_ssize_t tile_size0 = params.out_size2;
  iree_uk_ssize_t tile_size1 = params.out_size3;
  iree_uk_ssize_t out_stride_l0 = params.out_stride0;
  iree_uk_ssize_t out_stride_l1 = params.out_size3 * params.out_size2;
  iree_uk_ssize_t out_stride_l2 = params.out_size3;
  iree_uk_ssize_t out_stride_l3 = 1;
  if (params.flags & IREE_UK_FLAG_PACK_TRANSPOSE_OUTER) {
    std::swap(outer_size0, outer_size1);
    std::swap(out_stride_l0, out_stride_l1);
  }
  if (params.flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER) {
    std::swap(tile_size0, tile_size1);
    std::swap(out_stride_l2, out_stride_l3);
  }
  assert(outer_size0 * tile_size0 >= params.in_size0);
  assert(outer_size1 * tile_size1 >= params.in_size1);
  for (iree_uk_ssize_t outer_i0 = 0; outer_i0 < outer_size0; ++outer_i0) {
    for (iree_uk_ssize_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
      for (iree_uk_ssize_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
        for (iree_uk_ssize_t tile_i1 = 0; tile_i1 < tile_size1; ++tile_i1) {
          iree_uk_ssize_t out_offset =
              outer_i0 * out_stride_l0 + tile_i0 * out_stride_l2 +
              outer_i1 * out_stride_l1 + tile_i1 * out_stride_l3;
          iree_uk_ssize_t i0 = outer_i0 * tile_size0 + tile_i0;
          iree_uk_ssize_t i1 = outer_i1 * tile_size1 + tile_i1;
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
  iree_uk_pack(&actual_params);

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
      params.in_size1 + iree_uk_test_random_engine_get_0_1(engine);
  params.out_stride0 = params.out_size1 * params.out_size2 * params.out_size3;
  iree_uk_type_t in_type = iree_uk_pack_in_type(params.type);
  iree_uk_ssize_t in_buffer_size = iree_uk_test_2d_buffer_length(
      in_type, params.in_size0, params.in_stride0);
  void* in_buffer = malloc(in_buffer_size);
  iree_uk_test_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  params.in_buffer = in_buffer;
  test_one_pack_using_given_input(params, engine);
  free(in_buffer);
}

static void pack_test_for_various_tile_shapes_and_flags(
    iree_uk_pack_type_t type, int tile_size0, int tile_size1,
    const iree_uk_uint64_t* cpu_data, iree_uk_test_random_engine_t* engine) {
  struct outer_shape_t {
    int size0, size1;
  };
  std::vector<outer_shape_t> outer_shapes{
      // Degenerate cases. Vacuous.
      {0, 1},
      {1, 0},
      // Non-degenerate cases.
      {1, 1},
      {2, 3},
      {3, 4},
      {4, 5},
      {5, 6},
      {6, 7},
      {7, 8},
      {11, 13},
      {13, 11},
      {31, 33},
      {33, 31},
      {123, 89},
  };
  enum class Pad { None, OneIncompleteTile, TonsOfPaddingTiles };
  for (const auto& outer_shape : outer_shapes) {
    for (bool transpose_inner : {false, true}) {
      for (bool transpose_outer : {false, true}) {
        for (Pad pad :
             {Pad::None, Pad::OneIncompleteTile, Pad::TonsOfPaddingTiles}) {
          iree_uk_pack_params_t params = {};
          params.type = type;
          params.cpu_data = cpu_data;
          iree_uk_ssize_t out_size0 = outer_shape.size0;
          iree_uk_ssize_t out_size1 = outer_shape.size1;
          iree_uk_ssize_t out_size2 = tile_size0;
          iree_uk_ssize_t out_size3 = tile_size1;
          params.out_size0 = out_size0;
          params.out_size1 = out_size1;
          if (pad == Pad::TonsOfPaddingTiles) {
            params.out_size0 += 64;
            params.out_size1 += 64;
          }
          params.out_size2 = out_size2;
          params.out_size3 = out_size3;
          params.flags = 0;
          if (transpose_outer) {
            params.flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
            std::swap(out_size0, out_size1);
          }
          if (transpose_inner) {
            params.flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
            std::swap(out_size2, out_size3);
          }
          params.in_size0 = out_size0 * out_size2;
          params.in_size1 = out_size1 * out_size3;
          if (pad == Pad::OneIncompleteTile) {
            iree_uk_ssize_t pad_size0 =
                iree_uk_test_random_engine_get_0_65535(engine) % out_size2;
            iree_uk_ssize_t pad_size1 =
                iree_uk_test_random_engine_get_0_65535(engine) % out_size3;
            params.in_size0 =
                std::max<iree_uk_ssize_t>(0, params.in_size0 - pad_size0);
            params.in_size0 =
                std::max<iree_uk_ssize_t>(0, params.in_size0 - pad_size1);
          }
          iree_uk_type_t out_type = iree_uk_pack_out_type(type);
          int out_elem_size = iree_uk_type_size(out_type);
          void* padding_value_buffer = malloc(out_elem_size);
          iree_uk_test_write_random_buffer(padding_value_buffer, out_elem_size,
                                           out_type, engine);
          params.padding_value = padding_value_buffer;
          test_one_pack_creating_input_for_given_shape(params, engine);
          free(padding_value_buffer);
        }
      }
    }
  }
}

static void pack_test(iree_uk_pack_type_t type, int tile_size0, int tile_size1,
                      iree_uk_uint64_t cpu_data_field_0_bit) {
  const iree_uk_uint64_t local_cpu_data_default[IREE_CPU_DATA_FIELD_COUNT] = {
      0};
  iree_uk_test_random_engine_t* engine = iree_uk_test_random_engine_create();
  // First try without any optional CPU feature. This matters even when the
  // feature is supported by the CPU because we want to test the fallback to
  // architecture-default or generic code.
  pack_test_for_various_tile_shapes_and_flags(type, tile_size0, tile_size1,
                                              local_cpu_data_default, engine);
  // If this is nonzero, we are asked to test again with this CPU feature.
  if (cpu_data_field_0_bit) {
    const iree_uk_uint64_t local_cpu_data_with_bit[IREE_CPU_DATA_FIELD_COUNT] =
        {cpu_data_field_0_bit};
    // Check if the CPU supports the feature (otherwise, we crash).
    bool supported = iree_cpu_data_field(0) & cpu_data_field_0_bit;
    char cpu_feat_str[32];
    iree_uk_test_cpu_features_str(cpu_feat_str, sizeof cpu_feat_str,
                                  local_cpu_data_with_bit, 1);
    if (supported) {
      // Run with the optional CPU feature.
      printf("Device supports CPU feature: %s\n", cpu_feat_str);
      pack_test_for_various_tile_shapes_and_flags(
          type, tile_size0, tile_size1, local_cpu_data_with_bit, engine);
    } else {
      printf("Skipped: device does not support CPU feature: %s\n",
             cpu_feat_str);
    }
  }

  iree_uk_test_random_engine_destroy(engine);
}

#define PACK_TEST(type, tile_size0, tile_size1, test_suffix, feature_bit)     \
  TEST(PackTest, type##_tile_##tile_size0##x##tile_size1##_##test_suffix) {   \
    pack_test(iree_uk_pack_type_##type, tile_size0, tile_size1, feature_bit); \
  }

// Generic tests, not matching any particular CPU feature. This is the place to
// test weird tile shapes to ensure e.g. that we haven't unwittingly baked in a
// power-of-two assumption
PACK_TEST(f32f32, 3, 5, generic, 0)
PACK_TEST(i8i8, 4, 2, generic, 0)
PACK_TEST(i32i32, 3, 4, generic, 0)
PACK_TEST(i8i8, 8, 8, generic, 0)

// ARM_64 tests.
#if defined(IREE_UK_ARCH_ARM_64)

#define PACK_ARM_64_TEST(type, tile_size0, tile_size1) \
  PACK_TEST(type, tile_size0, tile_size1, arm_64, 0)

#define PACK_ARM_64_TEST_WITH_CPU_FEATURE(type, tile_size0, tile_size1, \
                                          FEATURE)                      \
  PACK_TEST(type, tile_size0, tile_size1, arm_64_##FEATURE,             \
            IREE_CPU_DATA0_ARM_64_##FEATURE)

PACK_ARM_64_TEST(f32f32, 8, 1)
PACK_ARM_64_TEST(f32f32, 8, 8)
PACK_ARM_64_TEST(i8i8, 8, 1)
PACK_ARM_64_TEST(i32i32, 8, 8)
PACK_ARM_64_TEST_WITH_CPU_FEATURE(i8i8, 8, 4, DOTPROD)
PACK_ARM_64_TEST_WITH_CPU_FEATURE(i8i8, 8, 8, I8MM)

#endif  // defined(IREE_UK_ARCH_ARM_64)

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iree_cpu_initialize(iree_allocator_system());
  return RUN_ALL_TESTS();
}
