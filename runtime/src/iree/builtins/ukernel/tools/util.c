// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/tools/util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/call_once.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/math.h"
#include "iree/schemas/cpu_data.h"

// Implementation of iree_uk_assert_fail failure is deferred to users code, i.e.
// to us here, as core ukernel/ code can't use the standard library.
void iree_uk_assert_fail(const char* file, int line, const char* function,
                         const char* condition) {
  fflush(stdout);
  // Must be a single fprintf call (which must make a single write) - typically
  // called from multiple worker threads concurrently.
  fprintf(stderr, "%s:%d: %s: assertion failed: %s\n", file, line, function,
          condition);
  fflush(stderr);
  abort();
}

iree_uk_index_t iree_uk_2d_buffer_length(iree_uk_type_t type,
                                         iree_uk_index_t size0,
                                         iree_uk_index_t stride0) {
  // As we require strides to be multiples of 8 bits, the stride value in bytes
  // is exact.
  return size0 * iree_uk_bits_to_bytes_exact(
                     stride0 << iree_uk_type_bit_count_log2(type));
}

bool iree_uk_2d_buffers_equal(const void* buf1, const void* buf2,
                              iree_uk_type_t type, iree_uk_index_t size0,
                              iree_uk_index_t size1, iree_uk_index_t stride0,
                              iree_uk_index_t stride1) {
  // Strides are required to be multiples of 8 bits.
  iree_uk_index_t stride0_bytes =
      iree_uk_bits_to_bytes_exact(stride0 << iree_uk_type_bit_count_log2(type));
  const char* buf1_ptr = buf1;
  const char* buf2_ptr = buf2;
  // Compare individual elements, but rounded up to whole enclosing bytes
  // in case of sub-byte-size elements. The assumption here is that
  // sub-byte-types aren't used with inner strides. Guard that assumption with
  // this assertion:
  IREE_UK_ASSERT(stride1 == 1 || iree_uk_type_bit_count(type) >= 8);
  const iree_uk_index_t elem_bytes_rounded_up =
      iree_uk_index_max(1, iree_uk_type_bit_count(type) / 8);
  for (iree_uk_index_t i0 = 0; i0 < size0; ++i0) {
    for (iree_uk_index_t i1 = 0; i1 < size1; ++i1) {
      iree_uk_index_t byte_offset = iree_uk_bits_to_bytes_exact(
          (i1 * stride1) << iree_uk_type_bit_count_log2(type));
      if (memcmp(buf1_ptr + byte_offset, buf2_ptr + byte_offset,
                 elem_bytes_rounded_up)) {
        return false;
      }
    }
    buf1_ptr += stride0_bytes;
    buf2_ptr += stride0_bytes;
  }
  return true;
}

// Parameter for locally defined lcg similar to std::minstd_rand.
#define IREE_PRNG_MULTIPLIER 48271
#define IREE_PRNG_MODULUS 2147483647

iree_uk_uint32_t iree_uk_random_engine_get_uint32(iree_uk_random_engine_t* e) {
  e->state = (e->state * IREE_PRNG_MULTIPLIER) % IREE_PRNG_MODULUS;
  return e->state;
}

iree_uk_uint64_t iree_uk_random_engine_get_uint64(iree_uk_random_engine_t* e) {
  iree_uk_uint64_t result = iree_uk_random_engine_get_uint32(e);
  result = (result << 32) + iree_uk_random_engine_get_uint32(e);
  return result;
}

int iree_uk_random_engine_get_0_65535(iree_uk_random_engine_t* e) {
  iree_uk_uint32_t v = iree_uk_random_engine_get_uint32(e);
  // Return the middle two out of the 4 bytes of state. It avoids
  // some mild issues with the least-significant and most-significant bytes.
  return (v >> 8) & 0xffff;
}

int iree_uk_random_engine_get_0_255(iree_uk_random_engine_t* e) {
  int v = iree_uk_random_engine_get_0_65535(e);
  return v & 0xff;
}

int iree_uk_random_engine_get_0_1(iree_uk_random_engine_t* e) {
  int v = iree_uk_random_engine_get_0_65535(e);
  return v & 1;
}

void iree_uk_write_random_buffer(void* buffer, iree_uk_index_t size_in_bytes,
                                 iree_uk_type_t type,
                                 iree_uk_random_engine_t* engine) {
  if (iree_uk_type_category(type) == IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS) {
    // Signless integers mean that the operation that will consume this buffer
    // should not care if the data is signed or unsigned integers, so let's
    // randomly exercise both and recurse so that the rest of this function
    // doesn't have to deal with signless again.
    iree_uk_type_t resolved_type = iree_uk_random_engine_get_0_1(engine)
                                       ? iree_uk_integer_type_as_signed(type)
                                       : iree_uk_integer_type_as_unsigned(type);
    iree_uk_write_random_buffer(buffer, size_in_bytes, resolved_type, engine);
    return;
  }
  // Special-case sub-byte-size integer types. Due to their narrow range, we
  // want to generate values over their entire range, and then it's down to
  // just generating random bytes.
  if (iree_uk_type_is_integer(type) && iree_uk_type_bit_count(type) < 8) {
    for (iree_uk_index_t i = 0; i < size_in_bytes; ++i) {
      ((uint8_t*)buffer)[i] = iree_uk_random_engine_get_0_255(engine);
    }
    return;
  }
  // All other element types.
  iree_uk_index_t elem_size = iree_uk_type_size(type);
  iree_uk_index_t size_in_elems = size_in_bytes / elem_size;
  for (iree_uk_index_t i = 0; i < size_in_elems; ++i) {
    // Small integers, should work for now for all the types we currently have
    // and enable exact float arithmetic, allowing to keep tests simpler for
    // now. Watch out for when we'll do float16!
    int random_val = iree_uk_random_engine_get_0_65535(engine);
    switch (type) {
      case IREE_UK_TYPE_FLOAT_32:
        ((float*)buffer)[i] = (random_val % 4) - 2;
        break;
      case IREE_UK_TYPE_FLOAT_16:
        ((uint16_t*)buffer)[i] =
            iree_math_f32_to_f16((float)((random_val % 16) - 8));
        break;
      case IREE_UK_TYPE_BFLOAT_16:
        ((uint16_t*)buffer)[i] =
            iree_math_f32_to_bf16((float)((random_val % 4) - 2));
        break;
      case IREE_UK_TYPE_SINT_32:
        ((int32_t*)buffer)[i] = (random_val % 2048) - 512;
        break;
      case IREE_UK_TYPE_UINT_32:
        ((uint32_t*)buffer)[i] = random_val % 2048;
        break;
      case IREE_UK_TYPE_SINT_16:
        ((int16_t*)buffer)[i] = (random_val % 2048) - 512;
        break;
      case IREE_UK_TYPE_UINT_16:
        ((uint16_t*)buffer)[i] = random_val % 2048;
        break;
      case IREE_UK_TYPE_SINT_8:
        ((int8_t*)buffer)[i] = (random_val % 256) - 128;
        break;
      case IREE_UK_TYPE_UINT_8:
        ((uint8_t*)buffer)[i] = random_val % 256;
        break;
      default:
        IREE_UK_ASSERT(false && "unknown type");
    }
  }
}

static const char* iree_uk_type_category_str(const iree_uk_type_t type) {
  switch (type & IREE_UK_TYPE_CATEGORY_MASK) {
    case IREE_UK_TYPE_CATEGORY_OPAQUE:
      return "x";
    case IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS:
      return "i";
    case IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED:
      return "s";
    case IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED:
      return "u";
    case IREE_UK_TYPE_CATEGORY_FLOAT_IEEE:
      return "f";
    case IREE_UK_TYPE_CATEGORY_FLOAT_BRAIN:
      return "bf";
    default:
      IREE_UK_ASSERT(false && "unknown type category");
      return "(?)";
  }
}

int iree_uk_type_str(char* buf, int buf_length, const iree_uk_type_t type) {
  return snprintf(buf, buf_length, "%s%d", iree_uk_type_category_str(type),
                  iree_uk_type_bit_count(type));
}

int iree_uk_type_pair_str(char* buf, int buf_length,
                          const iree_uk_type_pair_t pair) {
  char type0_buf[8];
  char type1_buf[8];
  iree_uk_type_str(type0_buf, sizeof type0_buf, iree_uk_untie_type(0, pair));
  iree_uk_type_str(type1_buf, sizeof type1_buf, iree_uk_untie_type(1, pair));
  return snprintf(buf, buf_length, "%s%s", type0_buf, type1_buf);
}

int iree_uk_type_triple_str(char* buf, int buf_length,
                            const iree_uk_type_triple_t triple) {
  char type0_buf[8];
  char type1_buf[8];
  char type2_buf[8];
  iree_uk_type_str(type0_buf, sizeof type0_buf, iree_uk_untie_type(0, triple));
  iree_uk_type_str(type1_buf, sizeof type1_buf, iree_uk_untie_type(1, triple));
  iree_uk_type_str(type2_buf, sizeof type2_buf, iree_uk_untie_type(2, triple));
  return snprintf(buf, buf_length, "%s%s%s", type0_buf, type1_buf, type2_buf);
}

static bool iree_uk_map_cpu_feature_name_to_bit(const char* cpu_feature_ptr,
                                                int cpu_feature_length,
                                                int* out_field_index,
                                                int* out_bit_pos) {
#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  if (IREE_ARCH_ENUM == IREE_ARCH_ENUM_##arch) {                              \
    if (!strncmp(cpu_feature_ptr, llvm_name, cpu_feature_length)) {           \
      *out_field_index = field_index;                                         \
      *out_bit_pos = bit_pos;                                                 \
      return true;                                                            \
    }                                                                         \
  }
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT
  return false;
}

void iree_uk_make_cpu_data_for_features(const char* cpu_features,
                                        iree_uk_uint64_t* out_cpu_data_fields) {
  const size_t data_fields_byte_size =
      IREE_CPU_DATA_FIELD_COUNT * sizeof(out_cpu_data_fields[0]);
  memset(out_cpu_data_fields, 0, data_fields_byte_size);
  // Empty string means architecture baseline. No bits set.
  if (!strcmp(cpu_features, "")) return;
  // Special case: when the name is "host", the list is required to be empty and
  // we detect capabilities of the host CPU.
  if (!strcmp(cpu_features, "host")) {
    memcpy(out_cpu_data_fields, iree_cpu_data_fields(), data_fields_byte_size);
    return;
  }

  // Named feature sets.
#if defined(IREE_ARCH_X86_64)
  iree_uk_uint64_t avx2_fma =
      IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_AVX2 |
      IREE_CPU_DATA0_X86_64_FMA | IREE_CPU_DATA0_X86_64_F16C;
  iree_uk_uint64_t avx512_base =
      avx2_fma | IREE_CPU_DATA0_X86_64_AVX512F |
      IREE_CPU_DATA0_X86_64_AVX512BW | IREE_CPU_DATA0_X86_64_AVX512DQ |
      IREE_CPU_DATA0_X86_64_AVX512VL | IREE_CPU_DATA0_X86_64_AVX512CD;
  if (!strcmp(cpu_features, "avx2_fma")) {
    out_cpu_data_fields[0] = avx2_fma;
    return;
  }
  if (!strcmp(cpu_features, "avx512_base")) {
    out_cpu_data_fields[0] = avx512_base;
    return;
  }
  if (!strcmp(cpu_features, "avx512_vnni")) {
    out_cpu_data_fields[0] = avx512_base | IREE_CPU_DATA0_X86_64_AVX512VNNI;
    return;
  }
  if (!strcmp(cpu_features, "avx512_bf16")) {
    out_cpu_data_fields[0] = avx512_base | IREE_CPU_DATA0_X86_64_AVX512BF16;
    return;
  }
#elif defined(IREE_ARCH_RISCV_64)
  iree_uk_uint64_t v = IREE_CPU_DATA0_RISCV_64_V;
  if (!strcmp(cpu_features, "v")) {
    out_cpu_data_fields[0] = v;
    return;
  }
#endif  // defined(IREE_ARCH_X86_64)

  // Fall back to interpreting cpu_features as a comma-separated list of LLVM
  // feature names.
  const char* cpu_features_end = cpu_features + strlen(cpu_features);
  while (true) {
    const char* first_comma = strchr(cpu_features, ',');
    const char* this_cpu_feature_end =
        first_comma ? first_comma : cpu_features_end;
    int this_cpu_feature_length = this_cpu_feature_end - cpu_features;
    int field_index;
    int bit_pos;
    if (!iree_uk_map_cpu_feature_name_to_bit(
            cpu_features, this_cpu_feature_length, &field_index, &bit_pos)) {
      fprintf(stderr, "CPU feature \"%s\" unknown on %s\n", cpu_features,
              IREE_ARCH);
      iree_abort();
    }
    out_cpu_data_fields[field_index] |= (1ull << bit_pos);
    if (this_cpu_feature_end == cpu_features_end) {
      break;
    }
    cpu_features = this_cpu_feature_end + 1;
  }
}

static void iree_uk_initialize_cpu_expensive(void) {
  iree_cpu_initialize(iree_allocator_system());
}

void iree_uk_initialize_cpu_once(void) {
  static iree_once_flag once = IREE_ONCE_FLAG_INIT;
  iree_call_once(&once, iree_uk_initialize_cpu_expensive);
}

bool iree_uk_cpu_supports(const iree_uk_uint64_t* cpu_data_fields) {
  for (int i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    if (cpu_data_fields[i] & ~iree_cpu_data_field(i)) return false;
  }
  return true;
}

static const char* iree_uk_cpu_feature_name(int feature_field_index,
                                            int feature_bit_pos) {
  IREE_UK_ASSERT(feature_field_index >= 0 &&
                 feature_field_index < IREE_CPU_DATA_FIELD_COUNT);
  IREE_UK_ASSERT(feature_bit_pos >= 0 && feature_bit_pos < 64);
#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  if (IREE_ARCH_ENUM == IREE_ARCH_ENUM_##arch) {                              \
    if (field_index == feature_field_index && bit_pos == feature_bit_pos) {   \
      return llvm_name;                                                       \
    }                                                                         \
  }
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT
  IREE_UK_ASSERT(false && "Unknown CPU feature bit");
  return NULL;
}

const char* iree_uk_cpu_first_unsupported_feature(
    const iree_uk_uint64_t* cpu_data_fields) {
  for (int i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    iree_uk_uint64_t unsupported_features_in_field =
        cpu_data_fields[i] & ~iree_cpu_data_field(i);
    for (int bit_pos = 0; bit_pos < 64; ++bit_pos) {
      iree_uk_uint64_t bit = 1ull << bit_pos;
      if (unsupported_features_in_field & bit) {
        return iree_uk_cpu_feature_name(i, bit_pos);
      }
    }
  }
  IREE_UK_ASSERT(false &&
                 "This function should only be called if there is an "
                 "unsupported CPU feature");
  return NULL;
}
