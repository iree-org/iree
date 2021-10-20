// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SHIMS_H_
#define IREE_VM_SHIMS_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/attributes.h"
#include "iree/base/target_platform.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"
#include "iree/vm/value.h"

//===----------------------------------------------------------------------===//
// Argument/result struct utilities
//===----------------------------------------------------------------------===//

#define IREE_VM_ABI_TYPE_NAME(types) iree_vm_abi_##types##_t

#define IREE_VM_ABI_FIXED_STRUCT(types, body) \
  IREE_VM_ABI_FIXED_STRUCT_IMPL(types, IREE_VM_ABI_TYPE_NAME(types), body)

#define IREE_VM_ABI_VLA_STRUCT(types, vla_count, vla_field, body) \
  IREE_VM_ABI_VLA_STRUCT_IMPL(types, vla_count, vla_field,        \
                              IREE_VM_ABI_TYPE_NAME(types), body)

#define IREE_VM_ABI_FIXED_STRUCT_IMPL(types, struct_type, body)        \
  typedef struct iree_vm_abi_##types##_t body IREE_ATTRIBUTE_PACKED    \
      struct_type;                                                     \
  static inline struct_type* iree_vm_abi_##types##_checked_deref(      \
      iree_byte_span_t buffer) {                                       \
    return IREE_LIKELY(buffer.data_length == sizeof(struct_type))      \
               ? (struct_type*)buffer.data                             \
               : NULL;                                                 \
  }                                                                    \
  static inline void iree_vm_abi_##types##_reset(struct_type* value) { \
    memset(value, 0, sizeof(struct_type));                             \
  }

#define IREE_VM_ABI_FIELD_SIZE(type, member) sizeof(((type*)NULL)->member)
#define IREE_VM_ABI_VLA_STRUCT_IMPL(types, vla_count, vla_field, struct_type, \
                                    body)                                     \
  typedef struct iree_vm_abi_##types##_t body IREE_ATTRIBUTE_PACKED           \
      struct_type;                                                            \
  static inline struct_type* iree_vm_abi_##types##_checked_deref(             \
      iree_byte_span_t buffer) {                                              \
    return IREE_LIKELY(buffer.data_length >= sizeof(struct_type)) &&          \
                   IREE_LIKELY(                                               \
                       buffer.data_length ==                                  \
                       sizeof(struct_type) +                                  \
                           ((const struct_type*)buffer.data)->vla_count *     \
                               IREE_VM_ABI_FIELD_SIZE(struct_type,            \
                                                      vla_field[0]))          \
               ? (struct_type*)buffer.data                                    \
               : NULL;                                                        \
  }

//===----------------------------------------------------------------------===//
// Shim function declaration/definition and accessor utilities
//===----------------------------------------------------------------------===//

typedef iree_status_t(IREE_API_PTR* iree_vm_native_function_target2_t)(
    iree_vm_stack_t* IREE_RESTRICT stack, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state, const void* IREE_RESTRICT args,
    void* IREE_RESTRICT rets);

#define IREE_VM_ABI_DECLARE_SHIM(arg_types, ret_types)                         \
  iree_status_t iree_vm_shim_##arg_types##_##ret_types(                        \
      iree_vm_stack_t* IREE_RESTRICT stack,                                    \
      const iree_vm_function_call_t* IREE_RESTRICT call,                       \
      iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module, \
      void* IREE_RESTRICT module_state,                                        \
      iree_vm_execution_result_t* IREE_RESTRICT out_result);

#define IREE_VM_ABI_DEFINE_SHIM(arg_types, ret_types)                          \
  iree_status_t iree_vm_shim_##arg_types##_##ret_types(                        \
      iree_vm_stack_t* IREE_RESTRICT stack,                                    \
      const iree_vm_function_call_t* IREE_RESTRICT call,                       \
      iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module, \
      void* IREE_RESTRICT module_state,                                        \
      iree_vm_execution_result_t* IREE_RESTRICT out_result) {                  \
    const IREE_VM_ABI_TYPE_NAME(arg_types)* args =                             \
        iree_vm_abi_##arg_types##_checked_deref(call->arguments);              \
    IREE_VM_ABI_TYPE_NAME(ret_types)* rets =                                   \
        iree_vm_abi_##ret_types##_checked_deref(call->results);                \
    if (IREE_UNLIKELY(!args || !rets)) {                                       \
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                    \
                              "argument/result signature mismatch");           \
    }                                                                          \
    iree_vm_abi_##ret_types##_reset(rets);                                     \
    return target_fn(stack, module, module_state, args, rets);                 \
  }

#define IREE_VM_ABI_EXPORT(function_name, module_state, arg_types, ret_types) \
  static iree_status_t function_name(                                         \
      iree_vm_stack_t* IREE_RESTRICT stack, void* IREE_RESTRICT module,       \
      module_state* IREE_RESTRICT state,                                      \
      IREE_VM_ABI_TYPE_NAME(arg_types) * IREE_RESTRICT args,                  \
      IREE_VM_ABI_TYPE_NAME(ret_types) * IREE_RESTRICT rets)

// TODO(benvanik): special case when source type and target type match.
#define IREE_VM_ABI_VLA_STACK_CAST(args, vla_count, vla_field, target_type, \
                                   max_count, out_count, out_ptrs)          \
  *(out_count) = (args)->vla_count;                                         \
  if (IREE_UNLIKELY((args)->vla_count > (max_count))) {                     \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "count %u > %u",      \
                            (args)->vla_count, (uint32_t)(max_count));      \
  }                                                                         \
  *(out_ptrs) =                                                             \
      (target_type*)iree_alloca((args)->vla_count * sizeof(target_type));   \
  for (iree_host_size_t i = 0; i < (args)->vla_count; ++i) {                \
    (*(out_ptrs))[i] = (target_type)((args)->vla_field[i].i0);              \
  }

#define IREE_VM_ABI_VLA_STACK_DEREF(args, vla_count, vla_field, ref_type,     \
                                    max_count, out_count, out_ptrs)           \
  *(out_count) = (args)->vla_count;                                           \
  if (IREE_UNLIKELY((args)->vla_count > (max_count))) {                       \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                         \
                            "count %u of " #ref_type " > %u",                 \
                            (args)->vla_count, (uint32_t)(max_count));        \
  }                                                                           \
  *(out_ptrs) =                                                               \
      (ref_type##_t**)iree_alloca((args)->vla_count * sizeof(ref_type##_t*)); \
  for (iree_host_size_t i = 0; i < (args)->vla_count; ++i) {                  \
    IREE_RETURN_IF_ERROR(                                                     \
        ref_type##_check_deref((args)->vla_field[i].r0, &(*(out_ptrs))[i]));  \
  }

#define IREE_VM_ABI_VLA_HEAP_DEREF(args, vla_count, vla_field, ref_type,         \
                                   host_allocator, out_count, out_ptrs)          \
  *(out_count) = (args)->vla_count;                                              \
  IREE_RETURN_IF_ERROR(iree_alloca((args)->vla_count * sizeof(ref_type##_t*));  \
  for (iree_host_size_t i = 0; i < (args)->vla_count; ++i) {                   \
    IREE_RETURN_IF_ERROR(                                                      \
        ref_type##_check_deref((args)->vla_field[i].r0, &(*(out_ptrs))[i]));  \
  }

//===----------------------------------------------------------------------===//
// Structures used for arguments and results.
//===----------------------------------------------------------------------===//

#if defined(IREE_COMPILER_MSVC)
#pragma pack(push, 1)
#endif  // IREE_COMPILER_MSVC

// Special case for void (empty args/rets) as C structs can't have a 0 length.
typedef struct iree_vm_abi_v_t {
  int unused;
} iree_vm_abi_v_t;
static inline iree_vm_abi_v_t* iree_vm_abi_v_checked_deref(
    iree_byte_span_t buffer) {
  return (iree_vm_abi_v_t*)buffer.data;
}
static inline void iree_vm_abi_v_reset(iree_vm_abi_v_t* value) {}

IREE_VM_ABI_FIXED_STRUCT(i, { int32_t i0; });

IREE_VM_ABI_FIXED_STRUCT(ii, {
  int32_t i0;
  int32_t i1;
});

IREE_VM_ABI_FIXED_STRUCT(iii, {
  int32_t i0;
  int32_t i1;
  int32_t i2;
});

IREE_VM_ABI_FIXED_STRUCT(iiii, {
  int32_t i0;
  int32_t i1;
  int32_t i2;
  int32_t i3;
});

IREE_VM_ABI_FIXED_STRUCT(irii, {
  int32_t i0;
  iree_vm_ref_t r1;
  int32_t i2;
  int32_t i3;
});

IREE_VM_ABI_FIXED_STRUCT(r, { iree_vm_ref_t r0; });

IREE_VM_ABI_FIXED_STRUCT(rr, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
});

IREE_VM_ABI_FIXED_STRUCT(rrr, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  iree_vm_ref_t r2;
});

IREE_VM_ABI_FIXED_STRUCT(ri, {
  iree_vm_ref_t r0;
  int32_t i1;
});

IREE_VM_ABI_FIXED_STRUCT(ririi, {
  iree_vm_ref_t r0;
  int32_t i1;
  iree_vm_ref_t r2;
  int32_t i3;
  int32_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(rii, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
});

IREE_VM_ABI_FIXED_STRUCT(rif, {
  iree_vm_ref_t r0;
  int32_t i1;
  float f2;
});

IREE_VM_ABI_FIXED_STRUCT(riii, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
  int32_t i3;
});

IREE_VM_ABI_FIXED_STRUCT(riirii, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
  iree_vm_ref_t r3;
  int32_t i4;
  int32_t i5;
});

IREE_VM_ABI_FIXED_STRUCT(riiirii, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
  int32_t i3;
  iree_vm_ref_t r4;
  int32_t i5;
  int32_t i6;
});

IREE_VM_ABI_FIXED_STRUCT(rriii, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  int32_t i3;
  int32_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(rriiii, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  int32_t i3;
  int32_t i4;
  int32_t i5;
});

IREE_VM_ABI_FIXED_STRUCT(rriri, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_ref_t r3;
  int32_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(rririi, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_ref_t r3;
  int32_t i4;
  int32_t i5;
});

IREE_VM_ABI_VLA_STRUCT(rCiD, a1_count, a1, {
  iree_vm_ref_t r0;
  iree_vm_size_t a1_count;
  iree_vm_abi_i_t a1[0];
});

IREE_VM_ABI_VLA_STRUCT(rCrD, a1_count, a1, {
  iree_vm_ref_t r0;
  iree_vm_size_t a1_count;
  iree_vm_abi_r_t a1[0];
});

IREE_VM_ABI_VLA_STRUCT(riCiD, a2_count, a2, {
  iree_vm_ref_t r0;
  int32_t i1;
  iree_vm_size_t a2_count;
  iree_vm_abi_i_t a2[0];
});

IREE_VM_ABI_VLA_STRUCT(riiCiD, a3_count, a3, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
  iree_vm_size_t a3_count;
  iree_vm_abi_i_t a3[0];
});

IREE_VM_ABI_VLA_STRUCT(riCrD, a2_count, a2, {
  iree_vm_ref_t r0;
  int32_t i1;
  iree_vm_size_t a2_count;
  iree_vm_abi_r_t a2[0];
});

IREE_VM_ABI_VLA_STRUCT(riiCriD, a3_count, a3, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
  iree_vm_size_t a3_count;
  iree_vm_abi_ri_t a3[0];
});

IREE_VM_ABI_VLA_STRUCT(rirCrD, a3_count, a3, {
  iree_vm_ref_t r0;
  int32_t i1;
  iree_vm_ref_t r2;
  iree_vm_size_t a3_count;
  iree_vm_abi_r_t a3[0];
});

IREE_VM_ABI_VLA_STRUCT(rrrCrD, a3_count, a3, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  iree_vm_ref_t r2;
  iree_vm_size_t a3_count;
  iree_vm_abi_r_t a3[0];
});

IREE_VM_ABI_VLA_STRUCT(rriCiD, a3_count, a3, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_size_t a3_count;
  iree_vm_abi_i_t a3[0];
});

IREE_VM_ABI_VLA_STRUCT(rrirCiD, a4_count, a4, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_ref_t r3;
  iree_vm_size_t a4_count;
  iree_vm_abi_i_t a4[0];
});

IREE_VM_ABI_VLA_STRUCT(riCiiiD, a2_count, a2, {
  iree_vm_ref_t r0;
  int32_t i1;
  iree_vm_size_t a2_count;
  iree_vm_abi_iii_t a2[0];
});

IREE_VM_ABI_VLA_STRUCT(rrCiriiD, a2_count, a2, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  iree_vm_size_t a2_count;
  iree_vm_abi_irii_t a2[0];
});

IREE_VM_ABI_VLA_STRUCT(rriCiriiD, a3_count, a3, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_size_t a3_count;
  iree_vm_abi_irii_t a3[0];
});

#if defined(IREE_COMPILER_MSVC)
#pragma pack(pop)
#endif  // IREE_COMPILER_MSVC

//===----------------------------------------------------------------------===//
// Shims for marshaling arguments and results
//===----------------------------------------------------------------------===//

IREE_VM_ABI_DECLARE_SHIM(irii, v);
IREE_VM_ABI_DECLARE_SHIM(r, i);
IREE_VM_ABI_DECLARE_SHIM(r, ii);
IREE_VM_ABI_DECLARE_SHIM(r, iii);
IREE_VM_ABI_DECLARE_SHIM(r, iiii);
IREE_VM_ABI_DECLARE_SHIM(r, r);
IREE_VM_ABI_DECLARE_SHIM(r, v);
IREE_VM_ABI_DECLARE_SHIM(rCiD, i);
IREE_VM_ABI_DECLARE_SHIM(rCrD, v);
IREE_VM_ABI_DECLARE_SHIM(ri, i);
IREE_VM_ABI_DECLARE_SHIM(ri, f);
IREE_VM_ABI_DECLARE_SHIM(ri, r);
IREE_VM_ABI_DECLARE_SHIM(ri, v);
IREE_VM_ABI_DECLARE_SHIM(riCiD, r);
IREE_VM_ABI_DECLARE_SHIM(riiCiD, r);
IREE_VM_ABI_DECLARE_SHIM(riCiiiD, r);
IREE_VM_ABI_DECLARE_SHIM(riCrD, r);
IREE_VM_ABI_DECLARE_SHIM(rii, i);
IREE_VM_ABI_DECLARE_SHIM(rii, r);
IREE_VM_ABI_DECLARE_SHIM(rii, v);
IREE_VM_ABI_DECLARE_SHIM(rif, v);
IREE_VM_ABI_DECLARE_SHIM(riii, r);
IREE_VM_ABI_DECLARE_SHIM(riii, v);
IREE_VM_ABI_DECLARE_SHIM(riirii, r);
IREE_VM_ABI_DECLARE_SHIM(riiirii, r);
IREE_VM_ABI_DECLARE_SHIM(rrrCrD, r);
IREE_VM_ABI_DECLARE_SHIM(ririi, v);
IREE_VM_ABI_DECLARE_SHIM(rr, i);
IREE_VM_ABI_DECLARE_SHIM(rr, r);
IREE_VM_ABI_DECLARE_SHIM(rr, v);
IREE_VM_ABI_DECLARE_SHIM(rr, ii);
IREE_VM_ABI_DECLARE_SHIM(rrr, ii);
IREE_VM_ABI_DECLARE_SHIM(rrCiriiD, r);
IREE_VM_ABI_DECLARE_SHIM(rriCiD, v);
IREE_VM_ABI_DECLARE_SHIM(rriCiriiD, v);
IREE_VM_ABI_DECLARE_SHIM(rriii, v);
IREE_VM_ABI_DECLARE_SHIM(rriiii, v);
IREE_VM_ABI_DECLARE_SHIM(rrirCiD, v);
IREE_VM_ABI_DECLARE_SHIM(rriri, v);
IREE_VM_ABI_DECLARE_SHIM(rririi, v);
IREE_VM_ABI_DECLARE_SHIM(v, i);
IREE_VM_ABI_DECLARE_SHIM(v, r);
IREE_VM_ABI_DECLARE_SHIM(v, v);

#endif  // IREE_VM_SHIMS_H_
