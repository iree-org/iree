// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/vmvx/module.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/vm/api.h"

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
// Private shims not needed by anything else.
//===----------------------------------------------------------------------===//

IREE_VM_ABI_FIXED_STRUCT(irIIII, {
  union {
    struct {
      int32_t fill_value;
      iree_vm_ref_t out_ref;
      int64_t out_offset;
      int64_t out_row_stride;
      int64_t size0;
      int64_t size1;
    } IREE_ATTRIBUTE_PACKED fill2d_x32;
    struct {
      int32_t i0;
      iree_vm_ref_t r1;
      int64_t i2;
      int64_t i3;
      int64_t i4;
      int64_t i5;
    } IREE_ATTRIBUTE_PACKED raw;
  };
});

IREE_VM_ABI_FIXED_STRUCT(rIIrIIrIIIIIffi, {
  union {
    struct {
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
      float alpha;
      float beta;
      int32_t flags;
    } IREE_ATTRIBUTE_PACKED matmul_f32;
    struct {
      iree_vm_ref_t r0;
      int64_t i1;
      int64_t i2;
      iree_vm_ref_t r3;
      int64_t i4;
      int64_t i5;
      iree_vm_ref_t r6;
      int64_t i7;
      int64_t i8;
      int64_t i9;
      int64_t i10;
      int64_t i11;
      float i12;
      float i13;
      int32_t flags;
    } IREE_ATTRIBUTE_PACKED raw;
  };
});

IREE_VM_ABI_FIXED_STRUCT(rIIIrIIIII, {
  union {
    struct {
      iree_vm_ref_t in_ref;
      int64_t in_offset;
      int64_t in_stride0;
      int64_t in_stride1;
      iree_vm_ref_t out_ref;
      int64_t out_offset;
      int64_t out_stride0;
      int64_t out_stride1;
      int64_t size0;
      int64_t size1;
    } IREE_ATTRIBUTE_PACKED unary2d;
    struct {
      iree_vm_ref_t r0;
      int64_t i1;
      int64_t i2;
      int64_t i3;
      iree_vm_ref_t r4;
      int64_t i5;
      int64_t i6;
      int64_t i7;
      int64_t i8;
      int64_t i9;
    } IREE_ATTRIBUTE_PACKED raw;
  };
});

// Binary elementwise ops all have the same signature:
// rIIIrIIIII
IREE_VM_ABI_FIXED_STRUCT(rIIIrIIIrIIIII, {
  union {
    struct {
      iree_vm_ref_t lhs_ref;
      int64_t lhs_offset;
      int64_t lhs_stride0;
      int64_t lhs_stride1;
      iree_vm_ref_t rhs_ref;
      int64_t rhs_offset;
      int64_t rhs_stride0;
      int64_t rhs_stride1;
      iree_vm_ref_t out_ref;
      int64_t out_offset;
      int64_t out_stride0;
      int64_t out_stride1;
      int64_t size0;
      int64_t size1;
    } IREE_ATTRIBUTE_PACKED binary2d;
    struct {
      iree_vm_ref_t r0;
      int64_t i1;
      int64_t i2;
      int64_t i3;
      iree_vm_ref_t r4;
      int64_t i5;
      int64_t i6;
      int64_t i7;
      iree_vm_ref_t r8;
      int64_t i9;
      int64_t i10;
      int64_t i11;
      int64_t i12;
      int64_t i13;
    } IREE_ATTRIBUTE_PACKED raw;
  };
});

// Since these are private/unlikely to be general, mark as static. If there
// ever becomes one of these in the common set, the compiler will complain.
static IREE_VM_ABI_DEFINE_SHIM(irIIII, v);
static IREE_VM_ABI_DEFINE_SHIM(rIIIrIIIII, v);
static IREE_VM_ABI_DEFINE_SHIM(rIIrIIrIIIIIffi, v);
static IREE_VM_ABI_DEFINE_SHIM(rIIIrIIIrIIIII, v);

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

#define BUFFER_2D_DECLS(name, dtype, offset, stride0, stride1, size0, size1) \
  uint64_t name##_overflow = 0;                                              \
  iree_host_size_t name##_size0 =                                            \
      iree_vmvx_cast_host_size(size0, &name##_overflow);                     \
  iree_host_size_t name##_size1 =                                            \
      iree_vmvx_cast_host_size(size1, &name##_overflow);                     \
  iree_host_size_t name##_stride0 =                                          \
      iree_vmvx_cast_host_size(stride0, &name##_overflow);                   \
  iree_host_size_t name##_stride1 =                                          \
      iree_vmvx_cast_host_size(stride1, &name##_overflow);                   \
  iree_host_size_t name##_length_bound = iree_vmvx_2d_length_bound(          \
      sizeof(dtype), name##_size0, name##_size1, name##_stride0,             \
      name##_stride1, &name##_overflow);                                     \
  iree_host_size_t name##_offset =                                           \
      sizeof(dtype) * iree_vmvx_cast_host_size(offset, &name##_overflow);    \
  if (name##_overflow) {                                                     \
    IREE_TRACE_ZONE_END(z0);                                                 \
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                    \
                            "buffer overflow for " #name);                   \
  }

#define MAP_BUFFER_2D_RO(name, dtype, buffer_ref, offset, stride0, stride1, \
                         size0, size1)                                      \
  iree_vm_buffer_t* name##_buffer;                                          \
  iree_const_byte_span_t name##_span;                                       \
  BUFFER_2D_DECLS(name, dtype, offset, stride0, stride1, size0, size1);     \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                        \
      z0, iree_vm_buffer_check_deref(buffer_ref, &name##_buffer))           \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                        \
      z0, iree_vm_buffer_map_ro(name##_buffer,       /*offset=*/            \
                                name##_offset,       /*length=*/            \
                                name##_length_bound, /*alignment=*/         \
                                sizeof(dtype), &name##_span));              \
  const dtype* name = (dtype*)name##_span.data

#define MAP_BUFFER_2D_RW(name, dtype, buffer_ref, offset, stride0, stride1,  \
                         size0, size1)                                       \
  iree_vm_buffer_t* name##_buffer;                                           \
  iree_byte_span_t name##_span;                                              \
  BUFFER_2D_DECLS(name, dtype, offset, stride0, stride1, size0, size1);      \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                         \
      z0, iree_vm_buffer_check_deref(buffer_ref, &name##_buffer));           \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                         \
      z0, iree_vm_buffer_map_rw(name##_buffer, /*offset=*/                   \
                                name##_offset, /*length=*/                   \
                                name##_length_bound,                         \
                                /*alignment=*/sizeof(dtype), &name##_span)); \
  dtype* name = (dtype*)name##_span.data

//===----------------------------------------------------------------------===//
// Exported
// function
// definitions
//===----------------------------------------------------------------------===//

IREE_VM_ABI_EXPORT(iree_vmvx_add2d_f32, iree_vmvx_module_state_t,
                   rIIIrIIIrIIIII, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(lhs, float,
                   /*buffer_ref=*/
                   args->binary2d.lhs_ref,
                   /*offset=*/
                   args->binary2d.lhs_offset,
                   /*stride0=*/
                   args->binary2d.lhs_stride0,
                   /*stride1=*/
                   args->binary2d.lhs_stride1,
                   /*size0=*/
                   args->binary2d.size0,
                   /*size1=*/
                   args->binary2d.size1);
  MAP_BUFFER_2D_RO(rhs, float,
                   /*buffer_ref=*/
                   args->binary2d.rhs_ref,
                   /*offset=*/
                   args->binary2d.rhs_offset,
                   /*stride0=*/
                   args->binary2d.rhs_stride0,
                   /*stride1=*/
                   args->binary2d.rhs_stride1,
                   /*size0=*/
                   args->binary2d.size0,
                   /*size1=*/
                   args->binary2d.size1);
  MAP_BUFFER_2D_RW(out, float,
                   /*buffer_ref=*/
                   args->binary2d.out_ref,
                   /*offset=*/
                   args->binary2d.out_offset,
                   /*stride0=*/
                   args->binary2d.out_stride0,
                   /*stride1=*/
                   args->binary2d.out_stride1,
                   /*size0=*/
                   args->binary2d.size0,
                   /*size1=*/
                   args->binary2d.size1);

  for (iree_host_size_t i = 0; i < out_size0; ++i) {
    for (iree_host_size_t j = 0; j < out_size1; ++j) {
      out[i * out_stride0 + j * out_stride1] =
          lhs[i * lhs_stride0 + j * lhs_stride1] +
          rhs[i * rhs_stride0 + j * rhs_stride1];
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_vmvx_copy2d_x32, iree_vmvx_module_state_t, rIIIrIIIII,
                   v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(in, int32_t,
                   /*buffer_ref=*/
                   args->unary2d.in_ref,
                   /*offset=*/
                   args->unary2d.in_offset,
                   /*stride0=*/
                   args->unary2d.in_stride0,
                   /*stride1=*/
                   args->unary2d.in_stride1,
                   /*size0=*/
                   args->unary2d.size0,
                   /*size1=*/
                   args->unary2d.size1);
  MAP_BUFFER_2D_RW(out, int32_t,
                   /*buffer_ref=*/
                   args->unary2d.out_ref,
                   /*offset=*/
                   args->unary2d.out_offset,
                   /*stride0=*/
                   args->unary2d.out_stride0,
                   /*stride1=*/
                   args->unary2d.out_stride1,
                   /*size0=*/
                   args->unary2d.size0,
                   /*size1=*/
                   args->unary2d.size1);

  for (iree_host_size_t j = 0; j < out_size0; ++j) {
    for (iree_host_size_t i = 0; i < out_size1; ++i) {
      out[j * out_stride0 + i * out_stride1] =
          in[j * in_stride0 + i * in_stride1];
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_vmvx_fill2d_x32, iree_vmvx_module_state_t, irIIII, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RW(out, int32_t,
                   /*buffer_ref=*/
                   args->fill2d_x32.out_ref,
                   /*offset=*/
                   args->fill2d_x32.out_offset,
                   /*stride0=*/
                   args->fill2d_x32.out_row_stride,
                   /*stride1=*/
                   1,
                   /*size0=*/
                   args->fill2d_x32.size0,
                   /*size1=*/
                   args->fill2d_x32.size1);

  for (iree_host_size_t i = 0; i < out_size0; ++i) {
    for (iree_host_size_t j = 0; j < out_size1; ++j) {
      out[i * out_stride0 + j] = args->fill2d_x32.fill_value;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_VM_ABI_EXPORT(iree_vmvx_matmul_f32f32f32, iree_vmvx_module_state_t,
                   rIIrIIrIIIIIffi, v) {
  IREE_TRACE_ZONE_BEGIN(z0);
  MAP_BUFFER_2D_RO(lhs, float,
                   /*buffer_ref=*/
                   args->matmul_f32.lhs_ref,
                   /*offset=*/
                   args->matmul_f32.lhs_offset,
                   /*stride0=*/
                   args->matmul_f32.lhs_row_stride,
                   /*stride1=*/
                   1,
                   /*size0=*/
                   args->matmul_f32.m,
                   /*size1=*/
                   args->matmul_f32.k);
  MAP_BUFFER_2D_RO(rhs, float,
                   /*buffer_ref=*/
                   args->matmul_f32.rhs_ref,
                   /*offset=*/
                   args->matmul_f32.rhs_offset,
                   /*stride0=*/
                   args->matmul_f32.rhs_row_stride,
                   /*stride1=*/
                   1,
                   /*size0=*/
                   args->matmul_f32.k,
                   /*size1=*/
                   args->matmul_f32.n);
  MAP_BUFFER_2D_RW(out, float,
                   /*buffer_ref=*/
                   args->matmul_f32.out_ref,
                   /*offset=*/
                   args->matmul_f32.out_offset,
                   /*stride0=*/
                   args->matmul_f32.out_row_stride,
                   /*stride1=*/
                   1,
                   /*size0=*/
                   args->matmul_f32.m,
                   /*size1=*/
                   args->matmul_f32.n);

  iree_host_size_t M = (iree_host_size_t)args->matmul_f32.m;
  iree_host_size_t N = (iree_host_size_t)args->matmul_f32.n;
  iree_host_size_t K = (iree_host_size_t)args->matmul_f32.k;

  // TODO: Define
  // flags more
  // robustly.
  if (args->matmul_f32.flags == 0) {
    // Row major.
    for (iree_host_size_t i = 0; i < M; ++i) {
      for (iree_host_size_t k = 0; k < K; ++k) {
        float apart = args->matmul_f32.alpha * lhs[i * lhs_stride0 + k];
        for (iree_host_size_t j = 0; j < N; ++j) {
          out[i * out_stride0 + j] +=
              args->matmul_f32.beta * apart * rhs[k * rhs_stride0 + j];
        }
      }
    }
  } else {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported matmul flags: %x",
                            (unsigned)args->matmul_f32.flags);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

// NOTE: this must match the ordering of the iree_vmvx_module_exports_table.
static const iree_vm_native_function_ptr_t iree_vmvx_module_funcs_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)       \
  {                                                            \
      .shim = (iree_vm_native_function_shim_t)                 \
          iree_vm_shim_##arg_types##_##ret_types,              \
      .target = (iree_vm_native_function_target_t)(target_fn), \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};

// NOTE: 0 length,
// but can't
// express that in
// C.
static const iree_vm_native_import_descriptor_t iree_vmvx_module_imports_[1];

static const iree_vm_native_export_descriptor_t iree_vmvx_module_exports_[] = {
#define EXPORT_FN(name, target_fn, arg_types, ret_types)           \
  {                                                                \
      .local_name = iree_string_view_literal(name),                \
      .calling_convention =                                        \
          iree_string_view_literal("0" #arg_types "_" #ret_types), \
      .attr_count = 0,                                             \
      .attrs = NULL,                                               \
  },
#include "iree/modules/vmvx/exports.inl"  // IWYU pragma: keep
#undef EXPORT_FN
};
static_assert(IREE_ARRAYSIZE(iree_vmvx_module_funcs_) ==
                  IREE_ARRAYSIZE(iree_vmvx_module_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t iree_vmvx_module_descriptor_ = {
    .module_name = iree_string_view_literal("vmv"
                                            "x"),
    .module_attr_count = 0,
    .module_attrs = NULL,
    .import_count = 0,  // workaround
                        // for
                        // 0-length
                        // C
                        // struct
    .imports = iree_vmvx_module_imports_,
    .export_count = IREE_ARRAYSIZE(iree_vmvx_module_exports_),
    .exports = iree_vmvx_module_exports_,
    .function_count = IREE_ARRAYSIZE(iree_vmvx_module_funcs_),
    .functions = iree_vmvx_module_funcs_,
};

IREE_API_EXPORT iree_status_t iree_vmvx_module_create(
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  // Setup the
  // interface with
  // the functions
  // we implement
  // ourselves. Any
  // function we
  // omit will be
  // handled by the
  // base native
  // module.
  static const iree_vm_module_t interface = {
      .destroy = iree_vmvx_module_destroy,
      .alloc_state = iree_vmvx_module_alloc_state,
      .free_state = iree_vmvx_module_free_state,
  };

  // Allocate
  // shared module
  // state.
  iree_host_size_t total_size =
      iree_vm_native_module_size() + sizeof(iree_vmvx_module_t);
  iree_vm_module_t* base_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&base_module));
  memset(base_module, 0, total_size);
  iree_status_t status = iree_vm_native_module_initialize(
      &interface, &iree_vmvx_module_descriptor_, host_allocator, base_module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, base_module);
    return status;
  }

  iree_vmvx_module_t* module = IREE_VMVX_MODULE_CAST(base_module);
  module->host_allocator = host_allocator;

  *out_module = base_module;
  return iree_ok_status();
}
