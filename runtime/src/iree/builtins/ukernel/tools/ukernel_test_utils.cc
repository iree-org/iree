// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/tools/ukernel_test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "iree/schemas/cpu_data.h"

// Implementation of iree_uk_assert_fail failure is deferred to users code, i.e.
// to us here, as core ukernel/ code can't use the standard library.
extern "C" void iree_uk_assert_fail(const char* file, int line,
                                    const char* function,
                                    const char* condition) {
  fflush(stdout);
  // Must be a single fprintf call (which must make a single write) - typically
  // called from multiple worker threads concurrently.
  fprintf(stderr, "%s:%d: %s: assertion failed: %s\n", file, line, function,
          condition);
  fflush(stderr);
  abort();
}

iree_uk_ssize_t iree_uk_test_2d_buffer_length(iree_uk_type_t type,
                                              iree_uk_ssize_t size0,
                                              iree_uk_ssize_t stride0) {
  // Just for testing purposes, so it's OK to overestimate size.
  return size0 * stride0 << iree_uk_type_size_log2(type);
}

bool iree_uk_test_2d_buffers_equal(const void* buf1, const void* buf2,
                                   iree_uk_type_t type, iree_uk_ssize_t size0,
                                   iree_uk_ssize_t size1,
                                   iree_uk_ssize_t stride0) {
  iree_uk_ssize_t elem_size = iree_uk_type_size(type);
  const char* buf1_ptr = static_cast<const char*>(buf1);
  const char* buf2_ptr = static_cast<const char*>(buf2);
  for (iree_uk_ssize_t i0 = 0; i0 < size0; ++i0) {
    if (memcmp(buf1_ptr, buf2_ptr, elem_size * size1)) return false;
    buf1_ptr += elem_size * stride0;
    buf2_ptr += elem_size * stride0;
  }
  return true;
}

struct iree_uk_test_random_engine_t {
  std::minstd_rand cpp_random_engine;
};

iree_uk_test_random_engine_t* iree_uk_test_random_engine_create() {
  return new iree_uk_test_random_engine_t;
}

void iree_uk_test_random_engine_destroy(iree_uk_test_random_engine_t* e) {
  delete e;
}

int iree_uk_test_random_engine_get_0_65535(iree_uk_test_random_engine_t* e) {
  iree_uk_uint32_t v = e->cpp_random_engine();
  // Return the middle two out of the 4 bytes of state. It avoids
  // some mild issues with the least-significant and most-significant bytes.
  return (v >> 8) & 0xffff;
}

int iree_uk_test_random_engine_get_0_1(iree_uk_test_random_engine_t* e) {
  int v = iree_uk_test_random_engine_get_0_65535(e);
  return v & 1;
}

int iree_uk_test_random_engine_get_minus16_plus15(
    iree_uk_test_random_engine_t* e) {
  int v = iree_uk_test_random_engine_get_0_65535(e);
  return (v % 32) - 16;
}

template <typename T>
static void iree_uk_test_write_random_buffer(
    T* buffer, iree_uk_ssize_t size_in_bytes,
    iree_uk_test_random_engine_t* engine) {
  iree_uk_ssize_t size_in_elems = size_in_bytes / sizeof(T);
  IREE_UK_ASSERT(size_in_elems * sizeof(T) == size_in_bytes && "bad size");
  for (iree_uk_ssize_t i = 0; i < size_in_elems; ++i) {
    // Small integers, should work for now for all the types we currently have
    // and enable exact float arithmetic, allowing to keep tests simpler for
    // now. Watch out for when we'll do float16!
    T random_val = iree_uk_test_random_engine_get_minus16_plus15(engine);
    buffer[i] = random_val;
  }
}

void iree_uk_test_write_random_buffer(void* buffer,
                                      iree_uk_ssize_t size_in_bytes,
                                      iree_uk_type_t type,
                                      iree_uk_test_random_engine_t* engine) {
  switch (type) {
    case IREE_UK_TYPE_FLOAT_32:
      iree_uk_test_write_random_buffer(static_cast<float*>(buffer),
                                       size_in_bytes, engine);
      return;
    case IREE_UK_TYPE_INT_32:
      iree_uk_test_write_random_buffer(static_cast<iree_uk_int32_t*>(buffer),
                                       size_in_bytes, engine);
      return;
    case IREE_UK_TYPE_INT_8:
      iree_uk_test_write_random_buffer(static_cast<iree_uk_int8_t*>(buffer),
                                       size_in_bytes, engine);
      return;
    default:
      IREE_UK_ASSERT(false && "unknown type");
  }
}

static const char* iree_uk_test_type_category_str(const iree_uk_type_t type) {
  switch (type & IREE_UK_TYPE_CATEGORY_MASK) {
    case IREE_UK_TYPE_CATEGORY_OPAQUE:
      return "x";
    case IREE_UK_TYPE_CATEGORY_INTEGER:
      return "i";
    case IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED:
      return "si";
    case IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED:
      return "ui";
    case IREE_UK_TYPE_CATEGORY_FLOAT_IEEE:
      return "f";
    case IREE_UK_TYPE_CATEGORY_FLOAT_BRAIN:
      return "bf";
    default:
      IREE_UK_ASSERT(false && "unknown type category");
      return "(?)";
  }
}

int iree_uk_test_type_str(char* buf, int buf_length,
                          const iree_uk_type_t type) {
  return snprintf(buf, buf_length, "%s%d", iree_uk_test_type_category_str(type),
                  iree_uk_type_bit_count(type));
}

int iree_uk_test_type_pair_str(char* buf, int buf_length,
                               const iree_uk_type_pair_t pair) {
  char type0_buf[8];
  char type1_buf[8];
  iree_uk_test_type_str(type0_buf, sizeof type0_buf,
                        iree_uk_untie_type(0, pair));
  iree_uk_test_type_str(type1_buf, sizeof type1_buf,
                        iree_uk_untie_type(1, pair));
  return snprintf(buf, buf_length, "(%s,%s)", type0_buf, type1_buf);
}

int iree_uk_test_type_triple_str(char* buf, int buf_length,
                                 const iree_uk_type_triple_t triple) {
  char type0_buf[8];
  char type1_buf[8];
  char type2_buf[8];
  iree_uk_test_type_str(type0_buf, sizeof type0_buf,
                        iree_uk_untie_type(0, triple));
  iree_uk_test_type_str(type1_buf, sizeof type1_buf,
                        iree_uk_untie_type(1, triple));
  iree_uk_test_type_str(type2_buf, sizeof type2_buf,
                        iree_uk_untie_type(2, triple));
  return snprintf(buf, buf_length, "(%s,%s,%s)", type0_buf, type1_buf,
                  type2_buf);
}

int iree_uk_test_cpu_features_str(char* buf, int buf_length,
                                  const iree_uk_uint64_t* cpu_data,
                                  int cpu_data_length) {
  // In the future there will be multiple cpu data words. For now there's only
  // one. We assert that and take advantage to simplify the code.
  IREE_UK_ASSERT(cpu_data_length == 1);
  // We set only one feature bit at a time for now in this test. Not an actual
  // detected cpu data field. This might have to change in the future if some
  // code path relies on the combination of two features.
  // For now, asserting only one bit set, and taking advantage of that to work
  // with plain string literals.
  IREE_UK_ASSERT(0 == (cpu_data[0] & (cpu_data[0] - 1)));
  if (cpu_data[0] == 0) {
    return snprintf(buf, buf_length, "(none)");
  }
#if defined(IREE_UK_ARCH_ARM_64)
  if (cpu_data[0] & IREE_CPU_DATA0_ARM_64_I8MM) {
    return snprintf(buf, buf_length, "i8mm");
  }
  if (cpu_data[0] & IREE_CPU_DATA0_ARM_64_DOTPROD) {
    return snprintf(buf, buf_length, "dotprod");
  }
#endif  // defined(IREE_UK_ARCH_ARM_64)
  IREE_UK_ASSERT(false && "unknown CPU feature");
  return snprintf(buf, buf_length, "(unknown CPU feature)");
}
