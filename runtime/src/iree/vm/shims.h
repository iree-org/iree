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
#include "iree/vm/module.h"
#include "iree/vm/native_module.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"
#include "iree/vm/value.h"

//===----------------------------------------------------------------------===//
// Argument/result struct utilities
//===----------------------------------------------------------------------===//

#define IREE_VM_ABI_TYPE_NAME(types) iree_vm_abi_##types##_t

#define IREE_VM_ABI_FIXED_STRUCT(types, body) \
  IREE_VM_ABI_FIXED_STRUCT_IMPL(IREE_VM_ABI_TYPE_NAME(types), types, body)

#define IREE_VM_ABI_VLA_STRUCT(types, vla_count, vla_field, body) \
  IREE_VM_ABI_VLA_STRUCT_IMPL(types, vla_count, vla_field,        \
                              IREE_VM_ABI_TYPE_NAME(types), body)

#define IREE_VM_ABI_FIXED_STRUCT_IMPL(struct_type, types, body)        \
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
      iree_vm_native_function_flags_t flags, iree_byte_span_t args_storage,    \
      iree_byte_span_t rets_storage,                                           \
      iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module, \
      void* IREE_RESTRICT module_state);

#define IREE_VM_ABI_DEFINE_SHIM(arg_types, ret_types)                          \
  iree_status_t iree_vm_shim_##arg_types##_##ret_types(                        \
      iree_vm_stack_t* IREE_RESTRICT stack,                                    \
      iree_vm_native_function_flags_t flags, iree_byte_span_t args_storage,    \
      iree_byte_span_t rets_storage,                                           \
      iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module, \
      void* IREE_RESTRICT module_state) {                                      \
    const IREE_VM_ABI_TYPE_NAME(arg_types)* args =                             \
        iree_vm_abi_##arg_types##_checked_deref(args_storage);                 \
    IREE_VM_ABI_TYPE_NAME(ret_types)* rets =                                   \
        iree_vm_abi_##ret_types##_checked_deref(rets_storage);                 \
    if (IREE_UNLIKELY(                                                         \
            !((flags & IREE_VM_NATIVE_FUNCTION_CALL_RESUME) || args) ||        \
            !rets)) {                                                          \
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

#define IREE_VM_ABI_VLA_STACK_DEREF_OR_NULL(                                  \
    args, vla_count, vla_field, ref_type, max_count, out_count, out_ptrs)     \
  *(out_count) = (args)->vla_count;                                           \
  if (IREE_UNLIKELY((args)->vla_count > (max_count))) {                       \
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,                         \
                            "count %u of " #ref_type " > %u",                 \
                            (args)->vla_count, (uint32_t)(max_count));        \
  }                                                                           \
  *(out_ptrs) =                                                               \
      (ref_type##_t**)iree_alloca((args)->vla_count * sizeof(ref_type##_t*)); \
  for (iree_host_size_t i = 0; i < (args)->vla_count; ++i) {                  \
    IREE_RETURN_IF_ERROR(ref_type##_check_deref_or_null(                      \
        (args)->vla_field[i].r0, &(*(out_ptrs))[i]));                         \
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

IREE_VM_ABI_FIXED_STRUCT(I, { int64_t i0; });

IREE_VM_ABI_FIXED_STRUCT(f, { float f0; });

IREE_VM_ABI_FIXED_STRUCT(ii, {
  int32_t i0;
  int32_t i1;
});

IREE_VM_ABI_FIXED_STRUCT(iI, {
  int32_t i0;
  int64_t i1;
});

IREE_VM_ABI_FIXED_STRUCT(II, {
  int64_t i0;
  int64_t i1;
});

IREE_VM_ABI_FIXED_STRUCT(IIi, {
  int64_t i0;
  int64_t i1;
  int32_t i2;
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

IREE_VM_ABI_FIXED_STRUCT(irIi, {
  int32_t i0;
  iree_vm_ref_t r1;
  int64_t i2;
  int32_t i3;
});

IREE_VM_ABI_FIXED_STRUCT(irII, {
  int32_t i0;
  iree_vm_ref_t r1;
  int64_t i2;
  int64_t i3;
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

IREE_VM_ABI_FIXED_STRUCT(rI, {
  iree_vm_ref_t r0;
  int64_t i1;
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

IREE_VM_ABI_FIXED_STRUCT(rIi, {
  iree_vm_ref_t r0;
  int64_t i1;
  int32_t i2;
});

IREE_VM_ABI_FIXED_STRUCT(rIirrii, {
  iree_vm_ref_t r0;
  int64_t i1;
  int32_t i2;
  iree_vm_ref_t r3;
  iree_vm_ref_t r4;
  int32_t i5;
  int32_t i6;
});

IREE_VM_ABI_FIXED_STRUCT(rIirIIi, {
  iree_vm_ref_t r0;
  int64_t i1;
  int32_t i2;
  iree_vm_ref_t r3;
  int64_t i4;
  int64_t i5;
  int32_t i6;
});

IREE_VM_ABI_FIXED_STRUCT(rII, {
  iree_vm_ref_t r0;
  int64_t i1;
  int64_t i2;
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

IREE_VM_ABI_FIXED_STRUCT(riiii, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
  int32_t i3;
  int32_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(riiI, {
  iree_vm_ref_t r0;
  int32_t i1;
  int32_t i2;
  int64_t i3;
});

IREE_VM_ABI_FIXED_STRUCT(iirII, {
  int32_t i0;
  int32_t i1;
  iree_vm_ref_t r2;
  int64_t i3;
  int64_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(rIiiI, {
  iree_vm_ref_t r0;
  int64_t i1;
  int32_t i2;
  int32_t i3;
  int64_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(riIiirII, {
  iree_vm_ref_t r0;
  int32_t i1;
  int64_t i2;
  int32_t i3;
  int32_t i4;
  iree_vm_ref_t r5;
  int64_t i6;
  int64_t i7;
});

IREE_VM_ABI_FIXED_STRUCT(rriirIIrIII, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  int32_t i3;
  iree_vm_ref_t r4;
  int64_t i5;
  int64_t i6;
  iree_vm_ref_t r7;
  int64_t i8;
  int64_t i9;
  int64_t i10;
});

IREE_VM_ABI_FIXED_STRUCT(rriiii, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  int32_t i3;
  int32_t i4;
  int32_t i5;
});

IREE_VM_ABI_FIXED_STRUCT(rrIIii, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int64_t i2;
  int64_t i3;
  int32_t i4;
  int32_t i5;
});

IREE_VM_ABI_FIXED_STRUCT(rrirI, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_ref_t r3;
  int64_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(rrIrII, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int64_t i2;
  iree_vm_ref_t r3;
  int64_t i4;
  int64_t i5;
});

IREE_VM_ABI_FIXED_STRUCT(rrIii, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int64_t i2;
  int32_t i3;
  int32_t i4;
});

IREE_VM_ABI_FIXED_STRUCT(rrrIii, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  iree_vm_ref_t r2;
  int64_t i3;
  int32_t i4;
  int32_t i5;
});

IREE_VM_ABI_FIXED_STRUCT(rIrriiiI, {
  iree_vm_ref_t r0;
  int64_t i1;
  iree_vm_ref_t r2;
  iree_vm_ref_t r3;
  int32_t i4;
  int32_t i5;
  int32_t i6;
  int64_t i7;
});

IREE_VM_ABI_FIXED_STRUCT(rIrrrIrIIi, {
  iree_vm_ref_t r0;
  int64_t i1;
  iree_vm_ref_t r2;
  iree_vm_ref_t r3;
  iree_vm_ref_t r4;
  int64_t i5;
  iree_vm_ref_t r6;
  int64_t i7;
  int64_t i8;
  int32_t i9;
});

IREE_VM_ABI_FIXED_STRUCT(rIrrrrrrr, {
  iree_vm_ref_t r0;
  int64_t i1;
  iree_vm_ref_t r2;
  iree_vm_ref_t r3;
  iree_vm_ref_t r4;
  iree_vm_ref_t r5;
  iree_vm_ref_t r6;
  iree_vm_ref_t r7;
  iree_vm_ref_t r8;
});

IREE_VM_ABI_FIXED_STRUCT(rIrrrrIIiiI, {
  iree_vm_ref_t r0;
  int64_t i1;
  iree_vm_ref_t r2;
  iree_vm_ref_t r3;
  iree_vm_ref_t r4;
  iree_vm_ref_t r5;
  int64_t i6;
  int64_t i7;
  int32_t i8;
  int32_t i9;
  int64_t i10;
});

IREE_VM_ABI_FIXED_STRUCT(rIrrr, {
  iree_vm_ref_t r0;
  int64_t i1;
  iree_vm_ref_t r2;
  iree_vm_ref_t r3;
  iree_vm_ref_t r4;
});

IREE_VM_ABI_VLA_STRUCT(rIrrCrD, a4_count, a4, {
  iree_vm_ref_t r0;
  int64_t i1;
  iree_vm_ref_t r2;
  iree_vm_ref_t r3;
  iree_vm_size_t a4_count;
  iree_vm_abi_r_t a4[0];
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

IREE_VM_ABI_VLA_STRUCT(rIIiiCID, a5_count, a5, {
  iree_vm_ref_t r0;
  int64_t i1;
  int64_t i2;
  int32_t i3;
  int32_t i4;
  iree_vm_size_t a5_count;
  iree_vm_abi_I_t a5[0];
});

IREE_VM_ABI_VLA_STRUCT(rriiCID, a4_count, a4, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  int32_t i3;
  iree_vm_size_t a4_count;
  iree_vm_abi_I_t a4[0];
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

IREE_VM_ABI_VLA_STRUCT(rrrrCrD, a4_count, a4, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  iree_vm_ref_t r2;
  iree_vm_ref_t r3;
  iree_vm_size_t a4_count;
  iree_vm_abi_r_t a4[0];
});

IREE_VM_ABI_VLA_STRUCT(rriCiD, a3_count, a3, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_size_t a3_count;
  iree_vm_abi_i_t a3[0];
});

IREE_VM_ABI_VLA_STRUCT(rrirCID, a4_count, a4, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_ref_t r3;
  iree_vm_size_t a4_count;
  iree_vm_abi_I_t a4[0];
});

IREE_VM_ABI_VLA_STRUCT(riCiiiD, a2_count, a2, {
  iree_vm_ref_t r0;
  int32_t i1;
  iree_vm_size_t a2_count;
  iree_vm_abi_iii_t a2[0];
});

IREE_VM_ABI_VLA_STRUCT(rrCrIID, a2_count, a2, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  iree_vm_size_t a2_count;
  iree_vm_abi_rII_t a2[0];
});

IREE_VM_ABI_VLA_STRUCT(rriCiirIID, a3_count, a3, {
  iree_vm_ref_t r0;
  iree_vm_ref_t r1;
  int32_t i2;
  iree_vm_size_t a3_count;
  iree_vm_abi_iirII_t a3[0];
});

IREE_VM_ABI_VLA_STRUCT(CrD, a0_count, a0, {
  iree_vm_size_t a0_count;
  iree_vm_abi_r_t a0[0];
});

IREE_VM_ABI_VLA_STRUCT(CrID, a0_count, a0, {
  iree_vm_size_t a0_count;
  iree_vm_abi_rI_t a0[0];
});

IREE_VM_ABI_VLA_STRUCT(iCrD, a1_count, a1, {
  int32_t i0;
  iree_vm_size_t a1_count;
  iree_vm_abi_r_t a1[0];
});

#if defined(IREE_COMPILER_MSVC)
#pragma pack(pop)
#endif  // IREE_COMPILER_MSVC

//===----------------------------------------------------------------------===//
// Shims for marshaling arguments and results
//===----------------------------------------------------------------------===//

IREE_VM_ABI_DECLARE_SHIM(irIi, v);
IREE_VM_ABI_DECLARE_SHIM(r, i);
IREE_VM_ABI_DECLARE_SHIM(r, I);
IREE_VM_ABI_DECLARE_SHIM(r, ii);
IREE_VM_ABI_DECLARE_SHIM(r, iI);
IREE_VM_ABI_DECLARE_SHIM(r, iii);
IREE_VM_ABI_DECLARE_SHIM(r, iiii);
IREE_VM_ABI_DECLARE_SHIM(r, r);
IREE_VM_ABI_DECLARE_SHIM(r, rI);
IREE_VM_ABI_DECLARE_SHIM(r, v);
IREE_VM_ABI_DECLARE_SHIM(rCiD, i);
IREE_VM_ABI_DECLARE_SHIM(rCrD, v);
IREE_VM_ABI_DECLARE_SHIM(ri, i);
IREE_VM_ABI_DECLARE_SHIM(ri, ii);
IREE_VM_ABI_DECLARE_SHIM(ri, I);
IREE_VM_ABI_DECLARE_SHIM(ri, f);
IREE_VM_ABI_DECLARE_SHIM(ri, r);
IREE_VM_ABI_DECLARE_SHIM(ri, v);
IREE_VM_ABI_DECLARE_SHIM(rI, i);
IREE_VM_ABI_DECLARE_SHIM(rI, r);
IREE_VM_ABI_DECLARE_SHIM(rI, v);
IREE_VM_ABI_DECLARE_SHIM(riCiD, r);
IREE_VM_ABI_DECLARE_SHIM(rIIiiCID, r);
IREE_VM_ABI_DECLARE_SHIM(riCiiiD, r);
IREE_VM_ABI_DECLARE_SHIM(riCrD, r);
IREE_VM_ABI_DECLARE_SHIM(rIi, i);
IREE_VM_ABI_DECLARE_SHIM(rIirrii, r);
IREE_VM_ABI_DECLARE_SHIM(rIirIIi, r);
IREE_VM_ABI_DECLARE_SHIM(rii, r);
IREE_VM_ABI_DECLARE_SHIM(rII, r);
IREE_VM_ABI_DECLARE_SHIM(rii, v);
IREE_VM_ABI_DECLARE_SHIM(rif, v);
IREE_VM_ABI_DECLARE_SHIM(riii, r);
IREE_VM_ABI_DECLARE_SHIM(riiI, r);
IREE_VM_ABI_DECLARE_SHIM(riii, v);
IREE_VM_ABI_DECLARE_SHIM(rIiiI, r);
IREE_VM_ABI_DECLARE_SHIM(riIiirII, r);
IREE_VM_ABI_DECLARE_SHIM(rriirIIrIII, v);
IREE_VM_ABI_DECLARE_SHIM(rrrrCrD, r);
IREE_VM_ABI_DECLARE_SHIM(ririi, v);
IREE_VM_ABI_DECLARE_SHIM(rr, i);
IREE_VM_ABI_DECLARE_SHIM(rr, r);
IREE_VM_ABI_DECLARE_SHIM(rr, v);
IREE_VM_ABI_DECLARE_SHIM(rr, ii);
IREE_VM_ABI_DECLARE_SHIM(rr, iI);
IREE_VM_ABI_DECLARE_SHIM(rrr, iI);
IREE_VM_ABI_DECLARE_SHIM(rrr, r);
IREE_VM_ABI_DECLARE_SHIM(rrCrIID, v);
IREE_VM_ABI_DECLARE_SHIM(rriCiD, v);
IREE_VM_ABI_DECLARE_SHIM(rriiCID, v);
IREE_VM_ABI_DECLARE_SHIM(rriCiirIID, v);
IREE_VM_ABI_DECLARE_SHIM(rriiii, v);
IREE_VM_ABI_DECLARE_SHIM(rrIIii, v);
IREE_VM_ABI_DECLARE_SHIM(rrirCID, v);
IREE_VM_ABI_DECLARE_SHIM(rrirI, v);
IREE_VM_ABI_DECLARE_SHIM(rrIrII, v);
IREE_VM_ABI_DECLARE_SHIM(rrIii, v);
IREE_VM_ABI_DECLARE_SHIM(rrrIii, v);
IREE_VM_ABI_DECLARE_SHIM(rIrriiiI, r);
IREE_VM_ABI_DECLARE_SHIM(rIrrrIrIIi, v);
IREE_VM_ABI_DECLARE_SHIM(rIrrrrrrr, v);
IREE_VM_ABI_DECLARE_SHIM(rIrrrrIIiiI, r);
IREE_VM_ABI_DECLARE_SHIM(rIrrr, v);
IREE_VM_ABI_DECLARE_SHIM(rIrrCrD, v);
IREE_VM_ABI_DECLARE_SHIM(CrID, r);
IREE_VM_ABI_DECLARE_SHIM(CrD, r);
IREE_VM_ABI_DECLARE_SHIM(iCrD, i);
IREE_VM_ABI_DECLARE_SHIM(iI, rr);
IREE_VM_ABI_DECLARE_SHIM(irII, rr);
IREE_VM_ABI_DECLARE_SHIM(v, i);
IREE_VM_ABI_DECLARE_SHIM(v, r);
IREE_VM_ABI_DECLARE_SHIM(v, v);

#endif  // IREE_VM_SHIMS_H_
