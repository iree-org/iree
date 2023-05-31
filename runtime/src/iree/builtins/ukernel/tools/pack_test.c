// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/pack_internal.h"
#include "iree/builtins/ukernel/tools/test.h"
#include "iree/builtins/ukernel/tools/util.h"

static void iree_pack_reference(const iree_uk_pack_params_t* params) {
  // For now, the input and output element types are always the same.
  iree_uk_pack_type_t pack_type = iree_uk_pack_type(params->flags);
  iree_uk_type_t elem_type = iree_uk_pack_in_type(pack_type);
  iree_uk_index_t elem_size = iree_uk_type_size(elem_type);
  iree_uk_index_t outer_size0 = params->out_size0;
  iree_uk_index_t outer_size1 = params->out_size1;
  iree_uk_index_t tile_size0 = params->out_size2;
  iree_uk_index_t tile_size1 = params->out_size3;
  iree_uk_index_t out_stride_l0 = params->out_stride0;
  iree_uk_index_t out_stride_l1 = params->out_size3 * params->out_size2;
  iree_uk_index_t out_stride_l2 = params->out_size3;
  iree_uk_index_t out_stride_l3 = 1;
  if (params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_OUTER) {
    iree_uk_index_swap(&outer_size0, &outer_size1);
    iree_uk_index_swap(&out_stride_l0, &out_stride_l1);
  }
  if (params->flags & IREE_UK_FLAG_PACK_TRANSPOSE_INNER) {
    iree_uk_index_swap(&tile_size0, &tile_size1);
    iree_uk_index_swap(&out_stride_l2, &out_stride_l3);
  }
  IREE_UK_ASSERT(outer_size0 * tile_size0 >= params->in_size0);
  IREE_UK_ASSERT(outer_size1 * tile_size1 >= params->in_size1);
  for (iree_uk_index_t outer_i0 = 0; outer_i0 < outer_size0; ++outer_i0) {
    for (iree_uk_index_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
      for (iree_uk_index_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
        for (iree_uk_index_t tile_i1 = 0; tile_i1 < tile_size1; ++tile_i1) {
          iree_uk_index_t out_offset =
              params->out_offset + outer_i0 * out_stride_l0 +
              tile_i0 * out_stride_l2 + outer_i1 * out_stride_l1 +
              tile_i1 * out_stride_l3;
          iree_uk_index_t i0 = outer_i0 * tile_size0 + tile_i0;
          iree_uk_index_t i1 = outer_i1 * tile_size1 + tile_i1;
          char* out_ptr = ((char*)params->out_buffer) + out_offset * elem_size;
          if (i0 >= params->in_size0 || i1 >= params->in_size1) {
            if (elem_size == 1) {
              *(iree_uk_uint8_t*)out_ptr = params->padding_value;
            } else if (elem_size == 2) {
              *(iree_uk_uint16_t*)out_ptr = params->padding_value;
            } else if (elem_size == 4) {
              *(iree_uk_uint32_t*)out_ptr = params->padding_value;
            } else {
              for (iree_uk_index_t k = 0; k < elem_size; k += 8) {
                *(iree_uk_uint64_t*)(out_ptr + k) = params->padding_value;
              }
            }
          } else {
            iree_uk_index_t in_offset =
                params->in_offset + i1 + i0 * params->in_stride0;
            const char* in_ptr =
                ((char*)params->in_buffer) + in_offset * elem_size;
            memcpy(out_ptr, in_ptr, elem_size);
          }
        }
      }
    }
  }
}

static void iree_uk_test_pack_for_shape_params(
    iree_uk_test_t* test, const iree_uk_pack_params_t* src_params) {
  iree_uk_pack_params_t params;
  memcpy(&params, src_params, sizeof params);
  // Populate strides first - we need them below to compute buffer lengths.
  // Randomly make strides either tight or not to exercise all cases.
  iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);
  params.in_stride0 = params.in_size1 + iree_uk_random_engine_get_0_1(engine);
  params.out_stride0 = params.out_size1 * params.out_size2 * params.out_size3;
  iree_uk_pack_type_t pack_type = iree_uk_pack_type(params.flags);
  iree_uk_type_t in_type = iree_uk_pack_in_type(pack_type);
  iree_uk_index_t in_buffer_size =
      iree_uk_2d_buffer_length(in_type, params.in_size0, params.in_stride0);
  void* in_buffer = malloc(in_buffer_size);
  iree_uk_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  params.in_offset = iree_uk_random_engine_get_0_65535(engine);
  params.out_offset = iree_uk_random_engine_get_0_65535(engine);
  params.in_buffer =
      (const char*)in_buffer - (params.in_offset * iree_uk_type_size(in_type));

  iree_uk_pack_params_t reference_params;
  memcpy(&reference_params, &params, sizeof reference_params);
  iree_uk_type_t out_type = iree_uk_pack_out_type(pack_type);
  iree_uk_index_t out_buffer_size =
      iree_uk_2d_buffer_length(out_type, params.out_size0, params.out_stride0);
  void* reference_out_buffer = malloc(out_buffer_size);
  iree_uk_write_random_buffer(reference_out_buffer, out_buffer_size, out_type,
                              engine);
  reference_params.out_buffer =
      (char*)reference_out_buffer -
      (params.out_offset * iree_uk_type_size(out_type));

  iree_uk_pack_params_t actual_params;
  memcpy(&actual_params, &params, sizeof actual_params);
  void* actual_out_buffer = malloc(out_buffer_size);
  iree_uk_write_random_buffer(actual_out_buffer, out_buffer_size, out_type,
                              engine);
  actual_params.out_buffer = (char*)actual_out_buffer -
                             (params.out_offset * iree_uk_type_size(out_type));

  iree_pack_reference(&reference_params);
  iree_uk_pack(&actual_params);

  if (memcmp(actual_out_buffer, reference_out_buffer, out_buffer_size)) {
    IREE_UK_TEST_FAIL(test);
  }

  free(reference_out_buffer);
  free(actual_out_buffer);
  free(in_buffer);
}

static void iree_uk_test_pack_for_tile_params(iree_uk_test_t* test,
                                              const void* src_params) {
  typedef struct outer_shape_t {
    int size0, size1;
  } outer_shape_t;
  const outer_shape_t outer_shapes[] = {
      // Degenerate cases. Vacuous.
      {0, 1},
      {1, 0},
      // Non-degenerate cases.
      {1, 1},
      {3, 2},
      {9, 33},
  };
  typedef enum {
    pad_none,
    pad_one_incomplete_tile,
    pad_a_lot,
    pad_enum_end
  } pad_t;
  for (int i = 0; i < IREE_ARRAYSIZE(outer_shapes); ++i) {
    for (int transpose_inner = 0; transpose_inner <= 1; ++transpose_inner) {
      for (int transpose_outer = 0; transpose_outer <= 1; ++transpose_outer) {
        for (pad_t pad = 0; pad < pad_enum_end; ++pad) {
          iree_uk_pack_params_t params;
          memcpy(&params, src_params, sizeof params);
          params.cpu_data = iree_uk_test_cpu_data(test);
          outer_shape_t outer_shape = outer_shapes[i];
          if (pad == pad_a_lot) {
            outer_shape.size0 += 16;
            outer_shape.size1 += 16;
          }
          params.out_size0 = outer_shape.size0;
          params.out_size1 = outer_shape.size1;
          if (transpose_outer) {
            params.flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
            iree_uk_index_swap(&params.out_size0, &params.out_size1);
          }
          iree_uk_index_t tile_size0 = params.out_size2;
          iree_uk_index_t tile_size1 = params.out_size3;
          if (transpose_inner) {
            params.flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
            iree_uk_index_swap(&tile_size0, &tile_size1);
          }
          params.in_size0 = outer_shape.size0 * tile_size0;
          params.in_size1 = outer_shape.size1 * tile_size1;
          iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);
          if (pad == pad_one_incomplete_tile) {
            iree_uk_index_t pad_size0 =
                iree_uk_random_engine_get_0_65535(engine) % tile_size0;
            iree_uk_index_t pad_size1 =
                iree_uk_random_engine_get_0_65535(engine) % tile_size1;
            params.in_size0 = params.in_size0 - pad_size0;
            if (params.in_size0 < 0) params.in_size0 = 0;
            params.in_size1 = params.in_size1 - pad_size1;
            if (params.in_size1 < 0) params.in_size1 = 0;
          }
          params.padding_value = iree_uk_random_engine_get_uint64(engine);
          iree_uk_test_pack_for_shape_params(test, &params);
        }
      }
    }
  }
}

static void iree_uk_test_pack(iree_uk_uint32_t flags, int tile_size0,
                              int tile_size1, const char* cpu_features) {
  iree_uk_pack_params_t params = {
      .flags = flags, .out_size2 = tile_size0, .out_size3 = tile_size1};
  char types_str[32];
  iree_uk_pack_type_t type = iree_uk_pack_type(flags);
  iree_uk_type_pair_str(types_str, sizeof types_str, type);
  char test_label_str[256];
  snprintf(test_label_str, sizeof test_label_str, "types:%s tile:%dx%d",
           types_str, tile_size0, tile_size1);
  iree_uk_test(test_label_str, iree_uk_test_pack_for_tile_params, &params,
               cpu_features);
}

int main(int argc, char** argv) {
  // Generic tests, not matching any particular CPU feature. This is the place
  // to test weird tile shapes to ensure e.g. that we haven't unwittingly baked
  // in a power-of-two assumption
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_F32F32, 3, 5, "");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I8I8, 4, 2, "");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I32I32, 3, 4, "");

#if defined(IREE_ARCH_ARM_64)
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_F32F32, 8, 1, "");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_F32F32, 8, 8, "");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I8I8, 8, 1, "");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I32I32, 8, 8, "");
  // Tile size selected with CPU feature dotprod.
  // Not passing a cpu_features_list because the packing code itself
  // does not depend on any features.
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I8I8, 8, 4, "");
  // Tile size selected for CPU feature i8mm. Same comment as for dotprod.
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I8I8, 8, 8, "");
#elif defined(IREE_ARCH_X86_64)
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_F32F32, 8, 1, "avx2_fma");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I8I8, 8, 2, "avx2_fma");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_F32F32, 8, 8, "avx2_fma");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I32I32, 8, 8, "avx2_fma");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_F32F32, 16, 1, "avx512_base");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I8I8, 16, 2, "avx512_base");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_F32F32, 16, 16, "avx512_base");
  iree_uk_test_pack(IREE_UK_FLAG_PACK_TYPE_I32I32, 16, 16, "avx512_base");
  // avx512_vnni uses the same tile size and same pack code as avx512_base.
#endif  // defined(IREE_ARCH_ARM_64)

  return iree_uk_test_exit_status();
}
