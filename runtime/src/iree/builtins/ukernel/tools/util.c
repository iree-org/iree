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

iree_uk_ssize_t iree_uk_2d_buffer_length(iree_uk_type_t type,
                                         iree_uk_ssize_t size0,
                                         iree_uk_ssize_t stride0) {
  // Just for testing purposes, so it's OK to overestimate size.
  return size0 * stride0 << iree_uk_type_size_log2(type);
}

bool iree_uk_2d_buffers_equal(const void* buf1, const void* buf2,
                              iree_uk_type_t type, iree_uk_ssize_t size0,
                              iree_uk_ssize_t size1, iree_uk_ssize_t stride0) {
  iree_uk_ssize_t elem_size = iree_uk_type_size(type);
  const char* buf1_ptr = buf1;
  const char* buf2_ptr = buf2;
  for (iree_uk_ssize_t i0 = 0; i0 < size0; ++i0) {
    if (memcmp(buf1_ptr, buf2_ptr, elem_size * size1)) return false;
    buf1_ptr += elem_size * stride0;
    buf2_ptr += elem_size * stride0;
  }
  return true;
}

// Parameter for locally defined lcg similar to std::minstd_rand.
#define IREE_PRNG_MULTIPLIER 48271
#define IREE_PRNG_MODULUS 2147483647

uint32_t iree_uk_random_engine_get(iree_uk_random_engine_t* e) {
  e->state = (e->state * IREE_PRNG_MULTIPLIER) % IREE_PRNG_MODULUS;
  return e->state;
}

int iree_uk_random_engine_get_0_65535(iree_uk_random_engine_t* e) {
  iree_uk_uint32_t v = iree_uk_random_engine_get(e);
  // Return the middle two out of the 4 bytes of state. It avoids
  // some mild issues with the least-significant and most-significant bytes.
  return (v >> 8) & 0xffff;
}

int iree_uk_random_engine_get_0_1(iree_uk_random_engine_t* e) {
  int v = iree_uk_random_engine_get_0_65535(e);
  return v & 1;
}

int iree_uk_random_engine_get_minus16_plus15(iree_uk_random_engine_t* e) {
  int v = iree_uk_random_engine_get_0_65535(e);
  return (v % 32) - 16;
}

void iree_uk_write_random_buffer(void* buffer, iree_uk_ssize_t size_in_bytes,
                                 iree_uk_type_t type,
                                 iree_uk_random_engine_t* engine) {
  iree_uk_ssize_t elem_size = iree_uk_type_size(type);
  iree_uk_ssize_t size_in_elems = size_in_bytes / elem_size;
  for (iree_uk_ssize_t i = 0; i < size_in_elems; ++i) {
    // Small integers, should work for now for all the types we currently have
    // and enable exact float arithmetic, allowing to keep tests simpler for
    // now. Watch out for when we'll do float16!
    int random_val = iree_uk_random_engine_get_minus16_plus15(engine);
    switch (type) {
      case IREE_UK_TYPE_FLOAT_32:
        ((float*)buffer)[i] = random_val;
        break;
      case IREE_UK_TYPE_INT_32:
        ((int32_t*)buffer)[i] = random_val;
        break;
      case IREE_UK_TYPE_INT_8:
        ((int8_t*)buffer)[i] = random_val;
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

// Just a vector of pointers to literal strings, used to hold CPU feature names.
struct iree_uk_cpu_features_list_t {
  // Number of string pointers.
  int size;
  // Number of string pointers that we have already allocated room for.
  int capacity;
  // Buffer of string pointers.
  const char** entries;
  // Optional: if not NULL, gives a shorthand name to this CPU features list.
  const char* name;
};

static iree_uk_cpu_features_list_t* iree_uk_cpu_features_list_create_empty(
    void) {
  iree_uk_cpu_features_list_t* list =
      malloc(sizeof(iree_uk_cpu_features_list_t));
  memset(list, 0, sizeof *list);
  return list;
}

void iree_uk_cpu_features_list_destroy(iree_uk_cpu_features_list_t* list) {
  free(list->entries);
  free(list);
}

static void iree_uk_cpu_features_list_append_one(
    iree_uk_cpu_features_list_t* list, const char* entry) {
  if (list->capacity == 0) {
    // TODO: Generalize if needed. Currently naive growth to fixed capacity.
    list->capacity = 64;
    IREE_UK_ASSERT(!list->entries);
    list->entries = malloc(list->capacity * sizeof list->entries[0]);
  }
  IREE_UK_ASSERT(list->size < list->capacity);
  list->entries[list->size++] = entry;
}

static void iree_uk_cpu_features_list_append_va(
    iree_uk_cpu_features_list_t* list, int count, va_list args) {
  for (int i = 0; i < count; ++i) {
    iree_uk_cpu_features_list_append_one(list, va_arg(args, const char*));
  }
}

iree_uk_cpu_features_list_t* iree_uk_cpu_features_list_create(int count, ...) {
  iree_uk_cpu_features_list_t* list = iree_uk_cpu_features_list_create_empty();
  va_list args;
  va_start(args, count);
  iree_uk_cpu_features_list_append_va(list, count, args);
  va_end(args);
  return list;
}

void iree_uk_cpu_features_list_append(iree_uk_cpu_features_list_t* list,
                                      int count, ...) {
  va_list args;
  va_start(args, count);
  iree_uk_cpu_features_list_append_va(list, count, args);
  va_end(args);
}

iree_uk_cpu_features_list_t* iree_uk_cpu_features_list_create_extend(
    iree_uk_cpu_features_list_t* list, int count, ...) {
  iree_uk_cpu_features_list_t* newlist =
      iree_uk_cpu_features_list_create_empty();
  for (int i = 0; i < list->size; ++i) {
    iree_uk_cpu_features_list_append_one(newlist, list->entries[i]);
  }
  va_list args;
  va_start(args, count);
  iree_uk_cpu_features_list_append_va(newlist, count, args);
  va_end(args);
  return newlist;
}

int iree_uk_cpu_features_list_size(const iree_uk_cpu_features_list_t* list) {
  return list->size;
};

const char* iree_uk_cpu_features_list_entry(
    const iree_uk_cpu_features_list_t* list, int index) {
  IREE_UK_ASSERT(index < list->size);
  return list->entries[index];
}

void iree_uk_cpu_features_list_set_name(iree_uk_cpu_features_list_t* list,
                                        const char* name) {
  list->name = name;
}

const char* iree_uk_cpu_features_list_get_name(
    const iree_uk_cpu_features_list_t* list) {
  return list->name;
}

iree_uk_standard_cpu_features_t* iree_uk_standard_cpu_features_create(void) {
  iree_uk_standard_cpu_features_t* cpu =
      malloc(sizeof(iree_uk_standard_cpu_features_t));
  memset(cpu, 0, sizeof *cpu);
#if defined(IREE_UK_ARCH_ARM_64)
  cpu->dotprod = iree_uk_cpu_features_list_create(1, "dotprod");
  cpu->i8mm = iree_uk_cpu_features_list_create(1, "i8mm");
#elif defined(IREE_UK_ARCH_X86_64)
  cpu->avx2_fma = iree_uk_cpu_features_list_create(3, "avx", "avx2", "fma");
  iree_uk_cpu_features_list_set_name(cpu->avx2_fma, "avx2_fma");
  cpu->avx512_base = iree_uk_cpu_features_list_create_extend(
      cpu->avx2_fma, 5, "avx512f", "avx512bw", "avx512dq", "avx512vl",
      "avx512cd");
  iree_uk_cpu_features_list_set_name(cpu->avx512_base, "avx512_base");
  cpu->avx512_vnni = iree_uk_cpu_features_list_create_extend(cpu->avx512_base,
                                                             1, "avx512vnni");
  iree_uk_cpu_features_list_set_name(cpu->avx512_vnni, "avx512_vnni");
#endif
  return cpu;
}

void iree_uk_standard_cpu_features_destroy(
    iree_uk_standard_cpu_features_t* cpu) {
#if defined(IREE_UK_ARCH_ARM_64)
  iree_uk_cpu_features_list_destroy(cpu->dotprod);
  iree_uk_cpu_features_list_destroy(cpu->i8mm);
#elif defined(IREE_UK_ARCH_X86_64)
  iree_uk_cpu_features_list_destroy(cpu->avx2_fma);
  iree_uk_cpu_features_list_destroy(cpu->avx512_base);
  iree_uk_cpu_features_list_destroy(cpu->avx512_vnni);
#endif
  free(cpu);
}

void iree_uk_make_cpu_data_for_features(
    const iree_uk_cpu_features_list_t* cpu_features,
    iree_uk_uint64_t* out_cpu_data_fields) {
  // Bit-field tracking which features exist, to diagnose misspelled features.
  uint64_t cpu_features_found = 0;
#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  if (IREE_ARCH_ENUM == IREE_ARCH_ENUM_##arch) {                              \
    for (int i = 0; i < cpu_features->size; ++i) {                            \
      if (!strcmp(cpu_features->entries[i], llvm_name)) {                     \
        out_cpu_data_fields[field_index] |= (1ull << bit_pos);                \
        cpu_features_found |= (1ull << i);                                    \
        break;                                                                \
      }                                                                       \
    }                                                                         \
  }
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT
  // Diagnose unknown (e.g. misspelled) features.
  for (int i = 0; i < cpu_features->size; ++i) {
    if (!(cpu_features_found & (1ull << i))) {
      fprintf(stderr, "CPU feature '%s' unknown on %s\n",
              cpu_features->entries[i], IREE_ARCH);
      iree_abort();
    }
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

const char* iree_uk_cpu_first_unsupported_feature(
    const iree_uk_cpu_features_list_t* cpu_features) {
  for (int i = 0; i < cpu_features->size; ++i) {
    int64_t supported = 0;
    IREE_CHECK_OK(iree_cpu_lookup_data_by_key(IREE_SV(cpu_features->entries[i]),
                                              &supported));
    if (!supported) return cpu_features->entries[i];
  }
  IREE_UK_ASSERT(false &&
                 "This function should only be called if there is an "
                 "unsupported CPU feature");
  return NULL;
}
