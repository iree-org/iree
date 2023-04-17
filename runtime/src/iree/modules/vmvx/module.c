// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "iree/modules/vmvx/module.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/vm/api.h"

// Include the ukernel support library so that we can use its implementations
// as fixed-function components of the runtime.
#include "iree/base/internal/cpu.h"
#include "iree/builtins/ukernel/api.h"

#define IREE_VMVX_MODULE_VERSION_0_0 0x00000000u
#define IREE_VMVX_MODULE_VERSION_LATEST IREE_VMVX_MODULE_VERSION_0_0

// Implementation of iree_uk_assert_fail failure is deferred to users code, i.e.
// to us here, as core ukernel/ code can't do I/O or depend on anything.
void iree_uk_assert_fail(const char* file, int line, const char* function,
                         const char* condition) {
#if (!defined(NDEBUG)) && (IREE_FILE_IO_ENABLE == 1)
  fflush(stdout);
  // Must be a single fprintf call (which must make a single write) - typically
  // called from multiple worker threads concurrently.
  fprintf(stderr, "%s:%d: %s: assertion failed: %s\n", file, line, function,
          condition);
  fflush(stderr);
#endif  // (!defined(NDEBUG)) && (IREE_FILE_IO_ENABLE == 1)
  IREE_ASSERT(false);
}

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

typedef struct iree_vmvx_module_t {
  iree_allocator_t host_allocator;
  // TODO(benvanik): types when we are not registering them globally.
} iree_vmvx_module_t;

#define IREE_VMVX_MODULE_CAST(module) \
  (iree_vmvx_module_t*)((uint8_t*)(module) + iree_vm_native_module_size());

typedef struct iree_vmvx_module_state_t {
  iree_allocator_t host_allocator;

  // Logical processor identifier used to index into processor info fields.
  // Depending on the implementation this may be an ordinal, a bitfield, or an
  // opaque unique identifier.
  uint32_t processor_id;

  // If we have any external libraries we want to interact with that are
  // stateful we could store their state here. Note that VMVX invocations may
  // happen from any thread and concurrently and if the state is not thread-safe
  // we'll have to perform the synchronization ourselves here. That'd be bad,
  // of course, and an indication that whatever is being called is not suited
  // for this use.
} iree_vmvx_module_state_t;

static void IREE_API_PTR iree_vmvx_module_destroy(void* base_module) {
  // No state to clean up (yet).
}

static iree_status_t IREE_API_PTR
iree_vmvx_module_alloc_state(void* self, iree_allocator_t host_allocator,
                             iree_vm_module_state_t** out_module_state) {
  iree_vmvx_module_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->host_allocator = host_allocator;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR
iree_vmvx_module_free_state(void* self, iree_vm_module_state_t* module_state) {
  iree_vmvx_module_state_t* state = (iree_vmvx_module_state_t*)module_state;
  iree_allocator_free(state->host_allocator, state);
}

//===----------------------------------------------------------------------===//
// Argument validation and marshalling
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_vmvx_2d_length_bound(
    iree_host_size_t element_size, uint64_t size0, uint64_t size1,
    uint64_t stride0, uint64_t stride1, uint64_t* overflow) {
  // Check for 2d size/stride overflow conditions for the equation:
  //   (size0 - 1) * stride0 + (size1 - 1) * stride1
  // This limits each (multiplicand + 1) to the 32bit range. We can get
  // smarter about this later or when scaling to >2D, but while limited, this
  // is easy and correct.
  *overflow |= (size0 & 0xffffffff00000000) | (size1 & 0xffffffff00000000) |
               ((stride0 + 1) & 0xffffffff00000000) |
               ((stride1 + 1) & 0xffffffff00000000);

  uint64_t last_index = (size0 - 1) * stride0 + (size1 - 1) * stride1;
  uint64_t max_size = (last_index + 1) * element_size;
  iree_host_size_t max_size_size_t = (iree_host_size_t)max_size;
  *overflow |= (max_size_size_t != max_size);  // No-op for 64bit size_t.
  return max_size_size_t;
}

static iree_host_size_t iree_vmvx_cast_host_size(int64_t value,
                                                 uint64_t* overflow) {
  if (sizeof(iree_host_size_t) == 4) {
    *overflow |= (uint64_t)value & 0xffffffff00000000ul;
  }
  return (iree_host_size_t)value;
}

#define BUFFER_2D_DECLS(name, dtype_size, offset, stride0, stride1, size0,    \
                        size1)                                                \
  uint64_t name##_overflow = 0;                                               \
  iree_host_size_t name##_size0 =                                             \
      iree_vmvx_cast_host_size(size0, &name##_overflow);                      \
  iree_host_size_t name##_size1 =                                             \
      iree_vmvx_cast_host_size(size1, &name##_overflow);                      \
  iree_host_size_t name##_stride0 =                                           \
      iree_vmvx_cast_host_size(stride0, &name##_overflow);                    \
  iree_host_size_t name##_stride1 =                                           \
      iree_vmvx_cast_host_size(stride1, &name##_overflow);                    \
  iree_host_size_t name##_length_bound = iree_vmvx_2d_length_bound(           \
      dtype_size, name##_size0, name##_size1, name##_stride0, name##_stride1, \
      &name##_overflow);                                                      \
  iree_host_size_t name##_offset =                                            \
      dtype_size * iree_vmvx_cast_host_size(offset, &name##_overflow);        \
  if (name##_overflow) {                                                      \
    IREE_TRACE_ZONE_END(z0);                                                  \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                     \
                            "buffer overflow for " #name);                    \
  }

#define MAP_BUFFER_2D_IMPL(mode, ptr_type, span_type, name, dtype_size,        \
                           buffer_ref, offset, stride0, stride1, size0, size1) \
  iree_vm_buffer_t* name##_buffer;                                             \
  span_type name##_span;                                                       \
  BUFFER_2D_DECLS(name, dtype_size, offset, stride0, stride1, size0, size1);   \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                           \
      z0, iree_vm_buffer_check_deref(buffer_ref, &name##_buffer))              \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                           \
      z0, iree_vm_buffer_map_##mode(name##_buffer,       /*offset=*/           \
                                    name##_offset,       /*length=*/           \
                                    name##_length_bound, /*alignment=*/        \
                                    dtype_size, &name##_span));                \
  ptr_type name = (ptr_type)name##_span.data

#define MAP_BUFFER_2D_UNTYPED_RO(name, dtype_size, ...)             \
  MAP_BUFFER_2D_IMPL(ro, const void*, iree_const_byte_span_t, name, \
                     dtype_size, __VA_ARGS__)
#define MAP_BUFFER_2D_UNTYPED_RW(name, dtype_size, ...) \
  MAP_BUFFER_2D_IMPL(rw, void*, iree_byte_span_t, name, dtype_size, __VA_ARGS__)
#define MAP_BUFFER_2D_RO(name, dtype, ...)                           \
  MAP_BUFFER_2D_IMPL(ro, const dtype*, iree_const_byte_span_t, name, \
                     sizeof(dtype), __VA_ARGS__)
#define MAP_BUFFER_2D_RW(name, dtype, ...)                              \
  MAP_BUFFER_2D_IMPL(rw, dtype*, iree_byte_span_t, name, sizeof(dtype), \
                     __VA_ARGS__)

//===----------------------------------------------------------------------===//
// Shared argument shims
//===----------------------------------------------------------------------===//

#define IREE_VMVX_ABI_EXPORT(function_name, arg_types, ret_types)        \
  IREE_VM_ABI_EXPORT(function_name, iree_vmvx_module_state_t, arg_types, \
                     ret_types)
#define IREE_VMVX_ABI_FIXED_STRUCT(name, types, body) \
  IREE_VM_ABI_FIXED_STRUCT(name, body)
#define IREE_VMVX_ABI_DEFINE_SHIM(arg_types, ret_types) \
  static IREE_VM_ABI_DEFINE_SHIM(arg_types, ret_types)

IREE_VMVX_ABI_FIXED_STRUCT(unary2d, rIrIIIIIII, {
  iree_vm_ref_t in_ref;
  int64_t in_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t in_stride0;
  int64_t in_stride1;
  int64_t out_stride0;
  int64_t out_stride1;
  int64_t size0;
  int64_t size1;
});
IREE_VMVX_ABI_DEFINE_SHIM(unary2d, v);

IREE_VMVX_ABI_FIXED_STRUCT(binary2d, rIrIrIIIIIIIII, {
  iree_vm_ref_t lhs_ref;
  int64_t lhs_offset;
  iree_vm_ref_t rhs_ref;
  int64_t rhs_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t lhs_stride0;
  int64_t lhs_stride1;
  int64_t rhs_stride0;
  int64_t rhs_stride1;
  int64_t out_stride0;
  int64_t out_stride1;
  int64_t size0;
  int64_t size1;
});
IREE_VMVX_ABI_DEFINE_SHIM(binary2d, v);

//===----------------------------------------------------------------------===//
// Ukernel shims. These shims are a bit different in that they directly marshal
// to a low level ukernel target function.
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_FIXED_STRUCT(ukernel_x32b_2d, rIrIrIIIIIIIII, {
  iree_vm_ref_t lhs_ref;
  int64_t lhs_offset;
  iree_vm_ref_t rhs_ref;
  int64_t rhs_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t lhs_stride0;
  int64_t lhs_stride1;
  int64_t rhs_stride0;
  int64_t rhs_stride1;
  int64_t out_stride0;
  int64_t out_stride1;
  int64_t size0;
  int64_t size1;
});

static iree_status_t iree_vm_shim_ukernel_x32b_2d_v(
    iree_vm_stack_t* IREE_RESTRICT stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state) {
  // TODO: Figure out how to identify this with the actual target fn.
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_vm_abi_ukernel_x32b_2d_t* args =
      iree_vm_abi_ukernel_x32b_2d_checked_deref(args_storage);
  if (IREE_UNLIKELY(!((flags & IREE_VM_NATIVE_FUNCTION_CALL_RESUME) || args))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }

  MAP_BUFFER_2D_RO(lhs, uint32_t,
                   /*buffer_ref=*/args->lhs_ref,
                   /*offset=*/args->lhs_offset,
                   /*stride0=*/args->lhs_stride0,
                   /*stride1=*/args->lhs_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);
  MAP_BUFFER_2D_RO(rhs, uint32_t,
                   /*buffer_ref=*/args->rhs_ref,
                   /*offset=*/args->rhs_offset,
                   /*stride0=*/args->rhs_stride0,
                   /*stride1=*/args->rhs_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);
  MAP_BUFFER_2D_RW(out, uint32_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_stride0,
                   /*stride1=*/args->out_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);

  iree_uk_x32b_2d_func_t ukernel_func = (iree_uk_x32b_2d_func_t)target_fn;

  int ret = ukernel_func(
      // LHS
      lhs, lhs_offset, lhs_stride0, lhs_stride1,
      // RHS
      rhs, rhs_offset, rhs_stride0, rhs_stride1,
      // OUT
      out, out_offset, out_stride0, out_stride1,
      // SIZE
      out_size0, out_size1);

  IREE_TRACE_ZONE_END(z0);
  return ret == 0
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "illegal x32b ukernel return code (%d)", ret);
}

IREE_VMVX_ABI_FIXED_STRUCT(ukernel_x32u_2d, rIrIIIIIII, {
  iree_vm_ref_t in_ref;
  int64_t in_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t in_stride0;
  int64_t in_stride1;
  int64_t out_stride0;
  int64_t out_stride1;
  int64_t size0;
  int64_t size1;
});

static iree_status_t iree_vm_shim_ukernel_x32u_2d_v(
    iree_vm_stack_t* IREE_RESTRICT stack, iree_vm_native_function_flags_t flags,
    iree_byte_span_t args_storage, iree_byte_span_t rets_storage,
    iree_vm_native_function_target2_t target_fn, void* IREE_RESTRICT module,
    void* IREE_RESTRICT module_state) {
  // TODO: Figure out how to identify this with the actual target fn.
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_vm_abi_ukernel_x32u_2d_t* args =
      iree_vm_abi_ukernel_x32u_2d_checked_deref(args_storage);
  if (IREE_UNLIKELY(!((flags & IREE_VM_NATIVE_FUNCTION_CALL_RESUME) || args))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "argument/result signature mismatch");
  }

  MAP_BUFFER_2D_RO(in, uint32_t,
                   /*buffer_ref=*/args->in_ref,
                   /*offset=*/args->in_offset,
                   /*stride0=*/args->in_stride0,
                   /*stride1=*/args->in_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);
  MAP_BUFFER_2D_RW(out, uint32_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_stride0,
                   /*stride1=*/args->out_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);

  iree_uk_x32u_2d_func_t ukernel_func = (iree_uk_x32u_2d_func_t)target_fn;

  int ret = ukernel_func(
      // IN
      in, in_offset, in_stride0, in_stride1,
      // OUT
      out, out_offset, out_stride0, out_stride1,
      // SIZE
      out_size0, out_size1);

  IREE_TRACE_ZONE_END(z0);
  return ret == 0
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "illegal x32u ukernel return code (%d)", ret);
}

//===----------------------------------------------------------------------===//
// Exported copy function definitions
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_EXPORT(iree_vmvx_copy2d_x8, unary2d, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(in, int8_t,
                   /*buffer_ref=*/args->in_ref,
                   /*offset=*/args->in_offset,
                   /*stride0=*/args->in_stride0,
                   /*stride1=*/args->in_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);
  MAP_BUFFER_2D_RW(out, int8_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_stride0,
                   /*stride1=*/args->out_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);

  for (iree_host_size_t j = 0; j < out_size0; ++j) {
    for (iree_host_size_t i = 0; i < out_size1; ++i) {
      out[j * out_stride0 + i * out_stride1] =
          in[j * in_stride0 + i * in_stride1];
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_copy2d_x16, unary2d, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(in, int16_t,
                   /*buffer_ref=*/args->in_ref,
                   /*offset=*/args->in_offset,
                   /*stride0=*/args->in_stride0,
                   /*stride1=*/args->in_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);
  MAP_BUFFER_2D_RW(out, int16_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_stride0,
                   /*stride1=*/args->out_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);

  for (iree_host_size_t j = 0; j < out_size0; ++j) {
    for (iree_host_size_t i = 0; i < out_size1; ++i) {
      out[j * out_stride0 + i * out_stride1] =
          in[j * in_stride0 + i * in_stride1];
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_copy2d_x32, unary2d, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(in, int32_t,
                   /*buffer_ref=*/args->in_ref,
                   /*offset=*/args->in_offset,
                   /*stride0=*/args->in_stride0,
                   /*stride1=*/args->in_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);
  MAP_BUFFER_2D_RW(out, int32_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_stride0,
                   /*stride1=*/args->out_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);

  for (iree_host_size_t j = 0; j < out_size0; ++j) {
    for (iree_host_size_t i = 0; i < out_size1; ++i) {
      out[j * out_stride0 + i * out_stride1] =
          in[j * in_stride0 + i * in_stride1];
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_copy2d_x64, unary2d, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(in, int64_t,
                   /*buffer_ref=*/args->in_ref,
                   /*offset=*/args->in_offset,
                   /*stride0=*/args->in_stride0,
                   /*stride1=*/args->in_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);
  MAP_BUFFER_2D_RW(out, int64_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_stride0,
                   /*stride1=*/args->out_stride1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);

  for (iree_host_size_t j = 0; j < out_size0; ++j) {
    for (iree_host_size_t i = 0; i < out_size1; ++i) {
      out[j * out_stride0 + i * out_stride1] =
          in[j * in_stride0 + i * in_stride1];
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Exported fill function definitions
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_FIXED_STRUCT(fill2d_x32, riIIII, {
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int32_t fill_value;
  int64_t out_row_stride;
  int64_t size0;
  int64_t size1;
});
IREE_VMVX_ABI_DEFINE_SHIM(fill2d_x32, v);
IREE_VMVX_ABI_EXPORT(iree_vmvx_fill2d_x32, fill2d_x32, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RW(out, int32_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_row_stride,
                   /*stride1=*/1,
                   /*size0=*/args->size0,
                   /*size1=*/args->size1);

  for (iree_host_size_t i = 0; i < out_size0; ++i) {
    for (iree_host_size_t j = 0; j < out_size1; ++j) {
      out[i * out_stride0 + j] = args->fill_value;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Exported matmul function definitions
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_FIXED_STRUCT(matmul, rIIrIIrIIIIIi, {
  iree_vm_ref_t lhs_ref;
  int64_t lhs_offset;
  int64_t lhs_row_stride;
  iree_vm_ref_t rhs_ref;
  int64_t rhs_offset;
  int64_t rhs_row_stride;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t out_row_stride;
  int64_t m;
  int64_t n;
  int64_t k;
  int32_t flags;
});
IREE_VMVX_ABI_DEFINE_SHIM(matmul, v);

IREE_VMVX_ABI_EXPORT(iree_vmvx_matmul_f32f32f32, matmul, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(lhs, float,
                   /*buffer_ref=*/args->lhs_ref,
                   /*offset=*/args->lhs_offset,
                   /*stride0=*/args->lhs_row_stride,
                   /*stride1=*/1,
                   /*size0=*/args->m,
                   /*size1=*/args->k);
  MAP_BUFFER_2D_RO(rhs, float,
                   /*buffer_ref=*/args->rhs_ref,
                   /*offset=*/args->rhs_offset,
                   /*stride0=*/args->rhs_row_stride,
                   /*stride1=*/1,
                   /*size0=*/args->k,
                   /*size1=*/args->n);
  MAP_BUFFER_2D_RW(out, float,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_row_stride,
                   /*stride1=*/1,
                   /*size0=*/args->m,
                   /*size1=*/args->n);

  iree_host_size_t M = (iree_host_size_t)args->m;
  iree_host_size_t N = (iree_host_size_t)args->n;
  iree_host_size_t K = (iree_host_size_t)args->k;

  // TODO: define flags more robustly
  unsigned accumulate_flag = args->flags & IREE_UK_FLAG_ACCUMULATE;
  unsigned unhandled_flags = args->flags ^ accumulate_flag;
  if (unhandled_flags) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported matmul flags: 0x%x", unhandled_flags);
  }
  for (iree_host_size_t i = 0; i < M; ++i) {
    for (iree_host_size_t j = 0; j < N; ++j) {
      float* out_ptr = out + i * out_stride0 + j;
      float acc = accumulate_flag ? *out_ptr : 0.f;
      for (iree_host_size_t k = 0; k < K; ++k) {
        acc += lhs[i * lhs_stride0 + k] * rhs[k * rhs_stride0 + j];
      }
      *out_ptr = acc;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_matmul_i8i8i32, matmul, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(lhs, int8_t,
                   /*buffer_ref=*/args->lhs_ref,
                   /*offset=*/args->lhs_offset,
                   /*stride0=*/args->lhs_row_stride,
                   /*stride1=*/1,
                   /*size0=*/args->m,
                   /*size1=*/args->k);
  MAP_BUFFER_2D_RO(rhs, int8_t,
                   /*buffer_ref=*/args->rhs_ref,
                   /*offset=*/args->rhs_offset,
                   /*stride0=*/args->rhs_row_stride,
                   /*stride1=*/1,
                   /*size0=*/args->k,
                   /*size1=*/args->n);
  MAP_BUFFER_2D_RW(out, int32_t,
                   /*buffer_ref=*/args->out_ref,
                   /*offset=*/args->out_offset,
                   /*stride0=*/args->out_row_stride,
                   /*stride1=*/1,
                   /*size0=*/args->m,
                   /*size1=*/args->n);

  iree_host_size_t M = (iree_host_size_t)args->m;
  iree_host_size_t N = (iree_host_size_t)args->n;
  iree_host_size_t K = (iree_host_size_t)args->k;

  // TODO: define flags more robustly
  unsigned accumulate_flag = args->flags & IREE_UK_FLAG_ACCUMULATE;
  unsigned unhandled_flags = args->flags ^ accumulate_flag;
  if (unhandled_flags) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported matmul flags: 0x%x", unhandled_flags);
  }
  for (iree_host_size_t i = 0; i < M; ++i) {
    for (iree_host_size_t j = 0; j < N; ++j) {
      int32_t* out_ptr = out + i * out_stride0 + j;
      int32_t acc = accumulate_flag ? *out_ptr : 0.f;
      for (iree_host_size_t k = 0; k < K; ++k) {
        // C's implicit promotion to int saves skin, but let's be explicit.
        int32_t lhs_val_int32 = lhs[i * lhs_stride0 + k];
        int32_t rhs_val_int32 = rhs[k * rhs_stride0 + j];
        acc += lhs_val_int32 * rhs_val_int32;
      }
      *out_ptr = acc;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Exported mmt4d function definitions
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_FIXED_STRUCT(mmt4d, rIrIrIIIIIIIiiii, {
  iree_vm_ref_t lhs_ref;
  int64_t lhs_offset;
  iree_vm_ref_t rhs_ref;
  int64_t rhs_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t lhs_row_stride;
  int64_t rhs_row_stride;
  int64_t out_row_stride;
  int64_t m;
  int64_t n;
  int64_t k;
  int32_t m0;
  int32_t n0;
  int32_t k0;
  uint32_t flags;
});
IREE_VMVX_ABI_DEFINE_SHIM(mmt4d, v);

static iree_status_t iree_vmvx_mmt4d(iree_uk_mmt4d_type_t type,
                                     const iree_vm_abi_mmt4d_t* args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t M = (iree_host_size_t)args->m;
  iree_host_size_t N = (iree_host_size_t)args->n;
  iree_host_size_t K = (iree_host_size_t)args->k;
  iree_host_size_t M0 = (iree_host_size_t)args->m0;
  iree_host_size_t N0 = (iree_host_size_t)args->n0;
  iree_host_size_t K0 = (iree_host_size_t)args->k0;
  iree_host_size_t lhs_tile_size = M0 * K0;
  iree_host_size_t rhs_tile_size = N0 * K0;
  iree_host_size_t out_tile_size = M0 * N0;
  int lhs_elem_size = iree_uk_type_size(iree_uk_mmt4d_lhs_type(type));
  int rhs_elem_size = iree_uk_type_size(iree_uk_mmt4d_rhs_type(type));
  int out_elem_size = iree_uk_type_size(iree_uk_mmt4d_out_type(type));
  // Here are abusing the 2D-specific macros MAP_BUFFER_2D_* to query 4D arrays.
  // Thanks to the requirement that all dimensions but the outer-most one are
  // contiguous row-major, the outer-most stride is the only nontrivial stride,
  // we can correctly coalesce the inner 3 dimensions without changing the
  // mapped span.
  MAP_BUFFER_2D_UNTYPED_RO(lhs,
                           /*dtype_size=*/lhs_elem_size,
                           /*buffer_ref=*/args->lhs_ref,
                           /*offset=*/args->lhs_offset,
                           /*stride0=*/args->lhs_row_stride,
                           /*stride1=*/1,
                           /*size0=*/M,
                           /*size1=*/K * lhs_tile_size);
  MAP_BUFFER_2D_UNTYPED_RO(rhs, /*dtype_size=*/rhs_elem_size,
                           /*buffer_ref=*/args->rhs_ref,
                           /*offset=*/args->rhs_offset,
                           /*stride0=*/args->rhs_row_stride,
                           /*stride1=*/1,
                           /*size0=*/N,
                           /*size1=*/K * rhs_tile_size);
  MAP_BUFFER_2D_UNTYPED_RW(out, /*dtype_size=*/out_elem_size,
                           /*buffer_ref=*/args->out_ref,
                           /*offset=*/args->out_offset,
                           /*stride0=*/args->out_row_stride,
                           /*stride1=*/1,
                           /*size0=*/M,
                           /*size1=*/N * out_tile_size);
  iree_uk_mmt4d_params_t ukernel_params = {
      .type = type,
      .flags = args->flags,
      .lhs_buffer = lhs,
      .rhs_buffer = rhs,
      .out_buffer = out,
      .lhs_stride = lhs_stride0,
      .rhs_stride = rhs_stride0,
      .out_stride = out_stride0,
      .M = M,
      .N = N,
      .K = K,
      .M0 = M0,
      .N0 = N0,
      .K0 = K0,
      .cpu_data = (const iree_uk_uint64_t*)iree_cpu_data_fields(),
  };
  iree_uk_mmt4d(&ukernel_params);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_mmt4d_f32f32f32, mmt4d, v) {
  return iree_vmvx_mmt4d(iree_uk_mmt4d_type_f32f32f32, args);
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_mmt4d_i8i8i32, mmt4d, v) {
  return iree_vmvx_mmt4d(iree_uk_mmt4d_type_i8i8i32, args);
}

//===----------------------------------------------------------------------===//
// Exported pack function definitions
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_FIXED_STRUCT(pack_f, rIrIIIIIIIIIIfi, {
  iree_vm_ref_t in_ref;
  int64_t in_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t in_stride0;
  int64_t out_stride0;
  int64_t in_size0;
  int64_t in_size1;
  int64_t out_size0;
  int64_t out_size1;
  int64_t out_size2;
  int64_t out_size3;
  float padding_value;
  uint32_t flags;
});
IREE_VMVX_ABI_DEFINE_SHIM(pack_f, v);

IREE_VMVX_ABI_FIXED_STRUCT(pack_i, rIrIIIIIIIIIIii, {
  iree_vm_ref_t in_ref;
  int64_t in_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t in_stride0;
  int64_t out_stride0;
  int64_t in_size0;
  int64_t in_size1;
  int64_t out_size0;
  int64_t out_size1;
  int64_t out_size2;
  int64_t out_size3;
  int32_t padding_value;
  uint32_t flags;
});
IREE_VMVX_ABI_DEFINE_SHIM(pack_i, v);

static iree_status_t iree_vmvx_pack_f(iree_uk_pack_type_t type,
                                      const iree_vm_abi_pack_f_t* args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t out_tile_size = args->out_size2 * args->out_size3;
  int in_elem_size = iree_uk_type_size(iree_uk_pack_in_type(type));
  int out_elem_size = iree_uk_type_size(iree_uk_pack_out_type(type));
  MAP_BUFFER_2D_UNTYPED_RO(in,
                           /*dtype_size=*/in_elem_size,
                           /*buffer_ref=*/args->in_ref,
                           /*offset=*/args->in_offset,
                           /*stride0=*/args->in_stride0,
                           /*stride1=*/1,
                           /*size0=*/args->in_size0,
                           /*size1=*/args->in_size1);
  MAP_BUFFER_2D_UNTYPED_RW(out, /*dtype_size=*/out_elem_size,
                           /*buffer_ref=*/args->out_ref,
                           /*offset=*/args->out_offset,
                           /*stride0=*/args->out_stride0,
                           /*stride1=*/1,
                           /*size0=*/args->out_size0,
                           /*size1=*/args->out_size1 * out_tile_size);
  iree_uk_pack_params_t ukernel_params = {
      .type = type,
      .in_buffer = in,
      .out_buffer = out,
      .in_stride0 = args->in_stride0,
      .out_stride0 = args->out_stride0,
      .in_size0 = args->in_size0,
      .in_size1 = args->in_size1,
      .out_size0 = args->out_size0,
      .out_size1 = args->out_size1,
      .out_size2 = args->out_size2,
      .out_size3 = args->out_size3,
      .padding_value = &args->padding_value,
      .flags = args->flags,
      .cpu_data = (const iree_uk_uint64_t*)iree_cpu_data_fields(),
  };
  iree_uk_pack(&ukernel_params);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_vmvx_pack_i(iree_uk_pack_type_t type,
                                      const iree_vm_abi_pack_i_t* args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t out_tile_size = args->out_size2 * args->out_size3;
  int in_elem_size = iree_uk_type_size(iree_uk_pack_in_type(type));
  int out_elem_size = iree_uk_type_size(iree_uk_pack_out_type(type));
  MAP_BUFFER_2D_UNTYPED_RO(in,
                           /*dtype_size=*/in_elem_size,
                           /*buffer_ref=*/args->in_ref,
                           /*offset=*/args->in_offset,
                           /*stride0=*/args->in_stride0,
                           /*stride1=*/1,
                           /*size0=*/args->in_size0,
                           /*size1=*/args->in_size1);
  MAP_BUFFER_2D_UNTYPED_RW(out, /*dtype_size=*/out_elem_size,
                           /*buffer_ref=*/args->out_ref,
                           /*offset=*/args->out_offset,
                           /*stride0=*/args->out_stride0,
                           /*stride1=*/1,
                           /*size0=*/args->out_size0,
                           /*size1=*/args->out_size1 * out_tile_size);
  iree_uk_pack_params_t ukernel_params = {
      .type = type,
      .in_buffer = in,
      .out_buffer = out,
      .in_stride0 = args->in_stride0,
      .out_stride0 = args->out_stride0,
      .in_size0 = args->in_size0,
      .in_size1 = args->in_size1,
      .out_size0 = args->out_size0,
      .out_size1 = args->out_size1,
      .out_size2 = args->out_size2,
      .out_size3 = args->out_size3,
      .padding_value = &args->padding_value,
      .flags = args->flags,
      .cpu_data = (const iree_uk_uint64_t*)iree_cpu_data_fields(),
  };
  iree_uk_pack(&ukernel_params);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_pack_f32f32, pack_f, v) {
  return iree_vmvx_pack_f(iree_uk_pack_type_f32f32, args);
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_pack_i8i8, pack_i, v) {
  return iree_vmvx_pack_i(iree_uk_pack_type_i8i8, args);
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_pack_i32i32, pack_i, v) {
  return iree_vmvx_pack_i(iree_uk_pack_type_i32i32, args);
}

//===----------------------------------------------------------------------===//
// Exported unpack function definitions
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_FIXED_STRUCT(unpack, rIrIIIIIIIIIIi, {
  iree_vm_ref_t in_ref;
  int64_t in_offset;
  iree_vm_ref_t out_ref;
  int64_t out_offset;
  int64_t in_stride0;
  int64_t out_stride0;
  int64_t in_size0;
  int64_t in_size1;
  int64_t in_size2;
  int64_t in_size3;
  int64_t out_size0;
  int64_t out_size1;
  uint32_t flags;
});
IREE_VMVX_ABI_DEFINE_SHIM(unpack, v);

static iree_status_t iree_vmvx_unpack(iree_uk_unpack_type_t type,
                                      const iree_vm_abi_unpack_t* args) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t out_tile_size = args->in_size2 * args->in_size3;
  int in_elem_size = iree_uk_type_size(iree_uk_unpack_in_type(type));
  int out_elem_size = iree_uk_type_size(iree_uk_unpack_out_type(type));
  MAP_BUFFER_2D_UNTYPED_RO(in,
                           /*dtype_size=*/in_elem_size,
                           /*buffer_ref=*/args->in_ref,
                           /*offset=*/args->in_offset,
                           /*stride0=*/args->in_stride0,
                           /*stride1=*/1,
                           /*size0=*/args->in_size0,
                           /*size1=*/args->in_size1 * out_tile_size);
  MAP_BUFFER_2D_UNTYPED_RW(out, /*dtype_size=*/out_elem_size,
                           /*buffer_ref=*/args->out_ref,
                           /*offset=*/args->out_offset,
                           /*stride0=*/args->out_stride0,
                           /*stride1=*/1,
                           /*size0=*/args->out_size0,
                           /*size1=*/args->out_size1);
  iree_uk_unpack_params_t ukernel_params = {
      .type = type,
      .in_buffer = in,
      .out_buffer = out,
      .in_stride0 = args->in_stride0,
      .out_stride0 = args->out_stride0,
      .in_size0 = args->in_size0,
      .in_size1 = args->in_size1,
      .in_size2 = args->in_size2,
      .in_size3 = args->in_size3,
      .out_size0 = args->out_size0,
      .out_size1 = args->out_size1,
      .flags = args->flags,
      .cpu_data = (const iree_uk_uint64_t*)iree_cpu_data_fields(),
  };
  iree_uk_unpack(&ukernel_params);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_unpack_f32f32, unpack, v) {
  return iree_vmvx_unpack(iree_uk_unpack_type_f32f32, args);
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_unpack_i8i8, unpack, v) {
  return iree_vmvx_unpack(iree_uk_unpack_type_i8i8, args);
}

IREE_VMVX_ABI_EXPORT(iree_vmvx_unpack_i32i32, unpack, v) {
  return iree_vmvx_unpack(iree_uk_unpack_type_i32i32, args);
}

//===----------------------------------------------------------------------===//
// Exported query_tile_sizes function definitions
//===----------------------------------------------------------------------===//

IREE_VMVX_ABI_FIXED_STRUCT(query_tile_sizes_2d, IIi, {
  int64_t size0;
  int64_t size1;
  uint32_t flags;
});
IREE_VMVX_ABI_DEFINE_SHIM(query_tile_sizes_2d, II);

IREE_VMVX_ABI_EXPORT(iree_vmvx_query_tile_sizes_2d, query_tile_sizes_2d, II) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_uk_query_tile_sizes_2d_params_t ukernel_params = {
      .size0 = args->size0,
      .size1 = args->size1,
      .flags = args->flags,
      .cpu_data = (const iree_uk_uint64_t*)iree_cpu_data_fields(),
  };
  iree_uk_query_tile_sizes_2d_out_params_t ukernel_out_params;
  iree_uk_query_tile_sizes_2d(&ukernel_params, &ukernel_out_params);
  rets->i0 = ukernel_out_params.tile_size0;
  rets->i1 = ukernel_out_params.tile_size1;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_vmvx_module_exports_table.
static const iree_vm_native_function_ptr_t iree_vmvx_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_struct, arg_types, ret_types) \
  {                                                                  \
      .shim = (iree_vm_native_function_shim_t)                       \
          iree_vm_shim_##arg_struct##_##ret_types,                   \
      .target = (iree_vm_native_function_target_t)(target_fn),       \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length, but can't express that in C.
static const iree_vm_native_import_descriptor_t iree_vmvx_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_vmvx_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_struct, arg_types, ret_types) \
  {                                                                  \
      .local_name = iree_string_view_literal(name),                  \
      .calling_convention =                                          \
          iree_string_view_literal("0" #arg_types "_" #ret_types),   \
      .attr_count = 0,                                               \
      .attrs = NULL,                                                 \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_vmvx_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_vmvx_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t iree_vmvx_module_descriptor_ = {
    .name = iree_string_view_literal("vmvx"),
    .version = IREE_VMVX_MODULE_VERSION_LATEST,
    .attr_count = 0,
    .attrs = NULL,
    .dependency_count = 0,
    .dependencies = NULL,
    .import_count = 0,  // workaround for 0-length C struct
    .imports = iree_vmvx_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_vmvx_module_exports_),
    .exports = iree_vmvx_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_vmvx_module_funcs_),
    .functions = iree_vmvx_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_vmvx_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  static const iree_vm_module_t interface = {
      .destroy = iree_vmvx_module_destroy,
      .alloc_state = iree_vmvx_module_alloc_state,
      .free_state = iree_vmvx_module_free_state,
  };

  // Allocate shared module state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_vmvx_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_vmvx_module_descriptor_, instance, host_allocator,
      base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_vmvx_module_t* module = IREE_VMVX_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;

  *out_module = base_module;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_vmvx_module_state_update_workgroup_state(
    iree_vm_module_state_t* module_state, uint32_t processor_id) {
  iree_vmvx_module_state_t* state = (iree_vmvx_module_state_t*)module_state;
  state->processor_id = processor_id;
}
