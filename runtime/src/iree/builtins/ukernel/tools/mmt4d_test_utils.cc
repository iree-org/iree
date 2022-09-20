// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/tools/mmt4d_test_utils.h"

#include <cassert>
#include <random>

#include "iree/schemas/cpu_data.h"

iree_mmt4d_scalar_type_t iree_ukernel_mmt4d_lhs_type(
    const iree_ukernel_mmt4d_params_t* params) {
  switch (params->type) {
    case iree_ukernel_mmt4d_type_f32f32f32:
      return iree_mmt4d_scalar_type_f32;
    case iree_ukernel_mmt4d_type_i8i8i32:
      return iree_mmt4d_scalar_type_i8;
    default:
      assert(false && "unknown type");
      return iree_mmt4d_scalar_type_unknown;
  }
}

iree_mmt4d_scalar_type_t iree_ukernel_mmt4d_rhs_type(
    const iree_ukernel_mmt4d_params_t* params) {
  // same for now
  return iree_ukernel_mmt4d_lhs_type(params);
}

iree_mmt4d_scalar_type_t iree_ukernel_mmt4d_out_type(
    const iree_ukernel_mmt4d_params_t* params) {
  switch (params->type) {
    case iree_ukernel_mmt4d_type_f32f32f32:
      return iree_mmt4d_scalar_type_f32;
    case iree_ukernel_mmt4d_type_i8i8i32:
      return iree_mmt4d_scalar_type_i32;
    default:
      assert(false && "unknown type");
      return iree_mmt4d_scalar_type_unknown;
  }
}

iree_ukernel_ssize_t iree_ukernel_mmt4d_lhs_buffer_size(
    const iree_ukernel_mmt4d_params_t* params) {
  return params->M * params->lhs_stride *
         iree_ukernel_mmt4d_lhs_elem_size(params->type);
}

iree_ukernel_ssize_t iree_ukernel_mmt4d_rhs_buffer_size(
    const iree_ukernel_mmt4d_params_t* params) {
  return params->N * params->rhs_stride *
         iree_ukernel_mmt4d_rhs_elem_size(params->type);
}

iree_ukernel_ssize_t iree_ukernel_mmt4d_out_buffer_size(
    const iree_ukernel_mmt4d_params_t* params) {
  return params->M * params->out_stride *
         iree_ukernel_mmt4d_out_elem_size(params->type);
}

struct iree_mmt4d_test_random_engine_t {
  std::minstd_rand cpp_random_engine;
};

iree_mmt4d_test_random_engine_t* iree_mmt4d_test_random_engine_create() {
  return new iree_mmt4d_test_random_engine_t;
}

void iree_mmt4d_test_random_engine_destroy(iree_mmt4d_test_random_engine_t* e) {
  delete e;
}

static int iree_mmt4d_test_random_engine_get_in_uint16_range(
    iree_mmt4d_test_random_engine_t* e) {
  uint32_t v = e->cpp_random_engine();
  // return the second-least-signicant out of the 4 bytes of state. It avoids
  // some mild issues with the least-significant and most-significant bytes.
  return (v >> 8) & 0xffff;
}

int iree_mmt4d_test_random_engine_get_0_or_1(
    iree_mmt4d_test_random_engine_t* e) {
  int v = iree_mmt4d_test_random_engine_get_in_uint16_range(e);
  return v & 1;
}

int iree_mmt4d_test_random_engine_get_between_minus16_and_plus15(
    iree_mmt4d_test_random_engine_t* e) {
  int v = iree_mmt4d_test_random_engine_get_in_uint16_range(e);
  return (v % 32) - 16;
}

template <typename T>
static void write_random_buffer(T* buffer, iree_ukernel_ssize_t size_in_bytes,
                                iree_mmt4d_test_random_engine_t* engine) {
  iree_ukernel_ssize_t size_in_elems = size_in_bytes / sizeof(T);
  assert(size_in_elems * sizeof(T) == size_in_bytes && "bad size");
  for (iree_ukernel_ssize_t i = 0; i < size_in_elems; ++i) {
    // Small integers, should work for now for all the types we currently have
    // and enable exact float arithmetic, allowing to keep tests simpler for
    // now. Watch out for when we'll do float16!
    T random_val =
        iree_mmt4d_test_random_engine_get_between_minus16_and_plus15(engine);
    buffer[i] = random_val;
  }
}

void write_random_buffer(void* buffer, iree_ukernel_ssize_t size_in_bytes,
                         iree_mmt4d_scalar_type_t type,
                         iree_mmt4d_test_random_engine_t* engine) {
  switch (type) {
    case iree_mmt4d_scalar_type_f32:
      write_random_buffer(static_cast<float*>(buffer), size_in_bytes, engine);
      return;
    case iree_mmt4d_scalar_type_i32:
      write_random_buffer(static_cast<int32_t*>(buffer), size_in_bytes, engine);
      return;
    case iree_mmt4d_scalar_type_i8:
      write_random_buffer(static_cast<int8_t*>(buffer), size_in_bytes, engine);
      return;
    default:
      assert(false && "unknown type");
  }
}

const char* get_mmt4d_type_str(const iree_ukernel_mmt4d_params_t* params) {
  switch (params->type) {
#define GET_MMT4D_TYPE_STR_CASE(x) \
  case x:                          \
    return #x;
    GET_MMT4D_TYPE_STR_CASE(iree_ukernel_mmt4d_type_f32f32f32);
    GET_MMT4D_TYPE_STR_CASE(iree_ukernel_mmt4d_type_i8i8i32);
    default:
      assert(false && "unknown type");
      return "unknown type";
  }
}

const char* get_cpu_features_str(const iree_ukernel_mmt4d_params_t* params) {
  // We set only one feature bit at a time in this test --- not an actual
  // detected cpu data field. This might have to change in the future if some
  // code path relies on the combination of two features.
  // For now, asserting only one bit set, and taking advantage of that to work
  // with plain string literals.
  assert(0 == (params->cpu_data_field_0 & (params->cpu_data_field_0 - 1)));
  if (params->cpu_data_field_0 == 0) {
    return "(none)";
  }
#if defined(IREE_UKERNEL_ARCH_ARM_64)
  if (params->cpu_data_field_0 & IREE_CPU_DATA_FIELD_0_AARCH64_HAVE_I8MM) {
    return "i8mm";
  }
  if (params->cpu_data_field_0 & IREE_CPU_DATA_FIELD_0_AARCH64_HAVE_DOTPROD) {
    return "dotprod";
  }
#endif  // defined(IREE_UKERNEL_ARCH_ARM_64)
  assert(false && "unknown CPU feature");
  return "unknown CPU feature";
}
