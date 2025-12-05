// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <memory>

#include "iree/base/api.h"
#include "iree/vm/buffer.h"
#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"
#include "iree/vm/native_module.h"
#include "iree/vm/native_module_cc.h"
#include "iree/vm/ref.h"
#include "iree/vm/shims.h"
#include "iree/vm/stack.h"

// Wrapper for calling the import functions with type (i32)->i32.
// NOTE: we should have some common ones prebuilt or can generate and rely on
// LTO to strip duplicates across the entire executable.
// TODO(benvanik): generate/export these shims/call functions in stack.h.
static iree_status_t call_import_i32_i32(iree_vm_stack_t* stack,
                                         const iree_vm_function_t* import,
                                         int32_t arg0, int32_t* out_ret0) {
  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arg0, sizeof(arg0));
  call.results = iree_make_byte_span(out_ret0, sizeof(*out_ret0));
  return import->module->begin_call(import->module, stack, call);
}

typedef iree_status_t (*call_i32_i32_t)(iree_vm_stack_t* stack,
                                        void* module_ptr, void* module_state,
                                        int32_t arg0, int32_t* out_ret0);

// Wrapper for calling a |target_fn| C function from the VM ABI.
// It's optional to bounce through like this; if the function can more
// efficiently directly access the arguments from the |call| then it can do so.
// This approach is most useful when the function may also be exported/used by
// non-VM code or may be internally referenced using a target-specific ABI.
// TODO(benvanik): generate/export these shims/call functions in stack.h.
static iree_status_t call_shim_i32_i32(iree_vm_stack_t* stack,
                                       iree_vm_native_function_flags_t flags,
                                       iree_byte_span_t args_storage,
                                       iree_byte_span_t rets_storage,
                                       call_i32_i32_t target_fn, void* module,
                                       void* module_state) {
  // We can use structs to allow compiler-controlled indexing optimizations,
  // though this won't work for variadic cases.
  // TODO(benvanik): packed attributes.
  typedef struct {
    int32_t arg0;
  } args_t;
  typedef struct {
    int32_t ret0;
  } results_t;

  const args_t* args = (const args_t*)args_storage.data;
  results_t* results = (results_t*)rets_storage.data;

  // For simple cases like this (zero or 1 result) we can tail-call.
  return target_fn(stack, module, module_state, args->arg0, &results->ret0);
}

//===----------------------------------------------------------------------===//
// module_a
//===----------------------------------------------------------------------===//
// This simple stateless module exports two functions that can be imported by
// other modules or called directly by the user. When no imports, custom types,
// or per-context state is required this simplifies module definitions.
//
// module_b below imports these functions and demonstrates a more complex module
// with state.

typedef struct module_a_t module_a_t;
typedef struct module_a_state_t module_a_state_t;

// vm.import private @module_a.add_1(%arg0 : i32) -> i32
static iree_status_t module_a_add_1(iree_vm_stack_t* stack, module_a_t* module,
                                    module_a_state_t* module_state,
                                    int32_t arg0, int32_t* out_ret0) {
  // Add 1 to arg0 and return.
  *out_ret0 = arg0 + 1;
  return iree_ok_status();
}

// vm.import private @module_a.sub_1(%arg0 : i32) -> i32
static iree_status_t module_a_sub_1(iree_vm_stack_t* stack, module_a_t* module,
                                    module_a_state_t* module_state,
                                    int32_t arg0, int32_t* out_ret0) {
  // Sub 1 to arg0 and return. Fail if < 0.
  *out_ret0 = arg0 - 1;
  return iree_ok_status();
}

static const iree_vm_native_export_descriptor_t module_a_exports_[] = {
    {IREE_SV("add_1"), IREE_SV("0i_i"), 0, NULL},
    {IREE_SV("sub_1"), IREE_SV("0i_i"), 0, NULL},
};
static const iree_vm_native_function_ptr_t module_a_funcs_[] = {
    {(iree_vm_native_function_shim_t)call_shim_i32_i32,
     (iree_vm_native_function_target_t)module_a_add_1},
    {(iree_vm_native_function_shim_t)call_shim_i32_i32,
     (iree_vm_native_function_target_t)module_a_sub_1},
};
static_assert(IREE_ARRAYSIZE(module_a_funcs_) ==
                  IREE_ARRAYSIZE(module_a_exports_),
              "function pointer table must be 1:1 with exports");
static const iree_vm_native_module_descriptor_t module_a_descriptor_ = {
    /*name=*/IREE_SV("module_a"),
    /*version=*/0,
    /*attr_count=*/0,
    /*attrs=*/NULL,
    /*dependency_count=*/0,
    /*dependencies=*/NULL,
    /*import_count=*/0,
    /*imports=*/NULL,
    /*export_count=*/IREE_ARRAYSIZE(module_a_exports_),
    /*exports=*/module_a_exports_,
    /*function_count=*/IREE_ARRAYSIZE(module_a_funcs_),
    /*functions=*/module_a_funcs_,
};

static iree_status_t module_a_create(iree_vm_instance_t* instance,
                                     iree_allocator_t allocator,
                                     iree_vm_module_t** out_module) {
  // NOTE: this module has neither shared or per-context module state.
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  return iree_vm_native_module_create(&interface, &module_a_descriptor_,
                                      instance, allocator, out_module);
}

//===----------------------------------------------------------------------===//
// module_b
//===----------------------------------------------------------------------===//
// A more complex module that holds state for resolved types (shared across
// all instances), imported functions (stored per-context), per-context user
// data, and reflection metadata.

typedef struct module_b_t module_b_t;
typedef struct module_b_state_t module_b_state_t;

// Stores shared state across all instances of the module.
// This should generally be treated as read-only and if mutation is possible
// then users must synchronize themselves.
typedef struct module_b_t {
  // Allocator the module must be freed with and that can be used for any other
  // shared dynamic allocations.
  iree_allocator_t allocator;
  // Resolved types; these never change once queried and are safe to store on
  // the shared structure to avoid needing to look them up again.
  iree_vm_ref_type_t types[1];
} module_b_t;

// Stores per-context state; at the minimum imports, but possibly other user
// state data. No synchronization is required as the VM will not call functions
// with the same state from multiple threads concurrently.
typedef struct module_b_state_t {
  // Allocator the state must be freed with and that can be used for any other
  // per-context dynamic allocations.
  iree_allocator_t allocator;
  // Resolved import functions matching 1:1 with the module import descriptors.
  iree_vm_function_t imports[2];
  // Example user data stored per-state.
  int counter;
} module_b_state_t;

// Frees the shared module; by this point all per-context states have been
// freed and no more shared data is required.
static void IREE_API_PTR module_b_destroy(void* self) {
  module_b_t* module = (module_b_t*)self;
  iree_allocator_free(module->allocator, module);
}

// Allocates per-context state, which stores resolved import functions and any
// other non-shared user state.
static iree_status_t IREE_API_PTR
module_b_alloc_state(void* self, iree_allocator_t allocator,
                     iree_vm_module_state_t** out_module_state) {
  module_b_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->allocator = allocator;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

// Frees the per-context state.
static void IREE_API_PTR
module_b_free_state(void* self, iree_vm_module_state_t* module_state) {
  module_b_state_t* state = (module_b_state_t*)module_state;
  iree_allocator_free(state->allocator, state);
}

// Clones the module state and retains resources by-reference.
static iree_status_t IREE_API_PTR module_b_fork_state(
    void* self, iree_vm_module_state_t* parent_state,
    iree_allocator_t allocator, iree_vm_module_state_t** out_child_state) {
  module_b_state_t* child_state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*child_state),
                                             (void**)&child_state));
  // Copy resolved imports and the counter value.
  memcpy(child_state, parent_state, sizeof(*child_state));
  // Reassign the allocator used.
  child_state->allocator = allocator;
  *out_child_state = (iree_vm_module_state_t*)child_state;
  return iree_ok_status();
}

// Called once per import function so the module can store the function ref.
static iree_status_t IREE_API_PTR module_b_resolve_import(
    void* self, iree_vm_module_state_t* module_state, iree_host_size_t ordinal,
    const iree_vm_function_t* function,
    const iree_vm_function_signature_t* signature) {
  module_b_state_t* state = (module_b_state_t*)module_state;
  state->imports[ordinal] = *function;
  return iree_ok_status();
}

// Our actual function. Here we directly access the registers but one could also
// use this as a trampoline into user code with a native signature (such as
// fetching the args, calling the function as a normal C function, and stashing
// back the results).
//
// vm.import private @module_b.entry(%arg0 : i32) -> i32
static iree_status_t module_b_entry(iree_vm_stack_t* stack, module_b_t* module,
                                    module_b_state_t* module_state,
                                    int32_t arg0, int32_t* out_ret0) {
  // NOTE: if we needed to use ref types here we have them under module->types.
  assert(module->types[0]);

  // Call module_a.add_1.
  IREE_RETURN_IF_ERROR(
      call_import_i32_i32(stack, &module_state->imports[0], arg0, &arg0));

  // Increment per-context state (persists across calls). No need for a mutex as
  // only one thread can be using the per-context state at a time.
  module_state->counter += arg0;
  int32_t ret0 = module_state->counter;

  // Call module_a.sub_1.
  IREE_RETURN_IF_ERROR(
      call_import_i32_i32(stack, &module_state->imports[1], ret0, &ret0));

  *out_ret0 = ret0;
  return iree_ok_status();
}

// Table of exported function pointers. Note that this table could be read-only
// (like here) or shared/per-context to allow exposing different functions based
// on versions, access rights, etc.
static const iree_vm_native_function_ptr_t module_b_funcs_[] = {
    {(iree_vm_native_function_shim_t)call_shim_i32_i32,
     (iree_vm_native_function_target_t)module_b_entry},
};

static const iree_vm_native_import_descriptor_t module_b_imports_[] = {
    {IREE_VM_NATIVE_IMPORT_REQUIRED, IREE_SV("module_a.add_1")},
    {IREE_VM_NATIVE_IMPORT_REQUIRED, IREE_SV("module_a.sub_1")},
};
static_assert(IREE_ARRAYSIZE(((module_b_state_t*)NULL)->imports) ==
                  IREE_ARRAYSIZE(module_b_imports_),
              "import storage must be able to hold all imports");
static const iree_string_pair_t module_b_entry_attrs_[] = {
    {{IREE_SV("key1")}, {IREE_SV("value1")}},
};
static const iree_vm_native_export_descriptor_t module_b_exports_[] = {
    {IREE_SV("entry"), IREE_SV("0i_i"), IREE_ARRAYSIZE(module_b_entry_attrs_),
     module_b_entry_attrs_},
};
static_assert(IREE_ARRAYSIZE(module_b_funcs_) ==
                  IREE_ARRAYSIZE(module_b_exports_),
              "function pointer table must be 1:1 with exports");
static const iree_vm_native_module_descriptor_t module_b_descriptor_ = {
    /*name=*/IREE_SV("module_b"),
    /*version=*/0,
    /*attr_count=*/0,
    /*attrs=*/NULL,
    /*dependency_count=*/0,
    /*dependencies=*/NULL,
    /*import_count=*/IREE_ARRAYSIZE(module_b_imports_),
    /*imports=*/module_b_imports_,
    /*export_count=*/IREE_ARRAYSIZE(module_b_exports_),
    /*exports=*/module_b_exports_,
    /*function_count=*/IREE_ARRAYSIZE(module_b_funcs_),
    /*functions=*/module_b_funcs_,
};

static iree_status_t module_b_create(iree_vm_instance_t* instance,
                                     iree_allocator_t allocator,
                                     iree_vm_module_t** out_module) {
  // Allocate shared module state.
  module_b_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  module->allocator = allocator;

  // Resolve types used by the module once so that we can share it across all
  // instances of the module. Depending on the types here can be somewhat risky
  // as it can lead to ordering issues. If possible resolving types on module
  // state is better as all dependent modules are guaranteed to have been
  // loaded.
  module->types[0] = iree_vm_instance_lookup_type(
      instance, iree_make_cstring_view("vm.buffer"));
  if (!module->types[0]) {
    iree_allocator_free(allocator, module);
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "required type vm.buffer not registered with the type system");
  }

  // Setup the interface with the functions we implement ourselves. Any function
  // we omit will be handled by the base native module.
  iree_vm_module_t interface;
  iree_status_t status = iree_vm_module_initialize(&interface, module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, module);
    return status;
  }
  interface.destroy = module_b_destroy;
  interface.alloc_state = module_b_alloc_state;
  interface.free_state = module_b_free_state;
  interface.fork_state = module_b_fork_state;
  interface.resolve_import = module_b_resolve_import;
  return iree_vm_native_module_create(&interface, &module_b_descriptor_,
                                      instance, allocator, out_module);
}

//===----------------------------------------------------------------------===//
// module_c_align
//===----------------------------------------------------------------------===//
// C implementation to test alignment-sensitive parameter unpacking and result
// packing patterns. This module implements the same functions as module_cpp
// using the C native module API.

typedef struct module_c_align_t {
  iree_allocator_t allocator;
} module_c_align_t;

typedef struct module_c_align_state_t {
  iree_allocator_t allocator;
  int counter;
} module_c_align_state_t;

static void IREE_API_PTR module_c_align_destroy(void* self) {
  module_c_align_t* module = (module_c_align_t*)self;
  iree_allocator_free(module->allocator, module);
}

static iree_status_t IREE_API_PTR
module_c_align_alloc_state(void* self, iree_allocator_t allocator,
                           iree_vm_module_state_t** out_module_state) {
  module_c_align_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->allocator = allocator;
  *out_module_state = (iree_vm_module_state_t*)state;
  return iree_ok_status();
}

static void IREE_API_PTR
module_c_align_free_state(void* self, iree_vm_module_state_t* module_state) {
  module_c_align_state_t* state = (module_c_align_state_t*)module_state;
  iree_allocator_free(state->allocator, state);
}

// (i32) -> i32 - basic entry point.
IREE_VM_ABI_EXPORT(module_c_align_entry, module_c_align_state_t, i, i) {
  state->counter += args->i0 + 1;
  rets->i0 = state->counter - 1;
  return iree_ok_status();
}

// (i32, ref<buffer>) -> i32 - ref after i32 triggers alignment.
IREE_VM_ABI_EXPORT(module_c_align_mixed_i32_ref, module_c_align_state_t, ir,
                   i) {
  iree_host_size_t buf_len =
      iree_vm_ref_is_null(&args->r1)
          ? 0
          : iree_vm_buffer_length((iree_vm_buffer_t*)args->r1.ptr);
  rets->i0 = args->i0 + (int32_t)buf_len;
  return iree_ok_status();
}

// (ref<buffer>, i32, ref<buffer>) -> i32 - ref/i32/ref pattern.
IREE_VM_ABI_EXPORT(module_c_align_mixed_ref_i32_ref, module_c_align_state_t,
                   rir, i) {
  iree_host_size_t len1 =
      iree_vm_ref_is_null(&args->r0)
          ? 0
          : iree_vm_buffer_length((iree_vm_buffer_t*)args->r0.ptr);
  iree_host_size_t len2 =
      iree_vm_ref_is_null(&args->r2)
          ? 0
          : iree_vm_buffer_length((iree_vm_buffer_t*)args->r2.ptr);
  rets->i0 = (int32_t)len1 + args->i1 + (int32_t)len2;
  return iree_ok_status();
}

// (i32, i64) -> i64 - i64 after i32 triggers alignment.
IREE_VM_ABI_EXPORT(module_c_align_mixed_i32_i64, module_c_align_state_t, iI,
                   I) {
  rets->i0 = args->i0 + args->i1;
  return iree_ok_status();
}

// (i32, i32, i32, ref<buffer>) -> i32 - ref after 3 i32s (12 bytes).
IREE_VM_ABI_EXPORT(module_c_align_mixed_i32x3_ref, module_c_align_state_t, iiir,
                   i) {
  iree_host_size_t buf_len =
      iree_vm_ref_is_null(&args->r3)
          ? 0
          : iree_vm_buffer_length((iree_vm_buffer_t*)args->r3.ptr);
  rets->i0 = args->i0 + args->i1 + args->i2 + (int32_t)buf_len;
  return iree_ok_status();
}

// (i64, i32) -> i32 - i32 after i64.
IREE_VM_ABI_EXPORT(module_c_align_mixed_i64_i32, module_c_align_state_t, Ii,
                   i) {
  rets->i0 = (int32_t)args->i0 + args->i1;
  return iree_ok_status();
}

// (i32, i64, i32) -> i64 - i64 sandwiched between i32s.
IREE_VM_ABI_EXPORT(module_c_align_mixed_i32_i64_i32, module_c_align_state_t,
                   iIi, I) {
  rets->i0 = args->i0 + args->i1 + args->i2;
  return iree_ok_status();
}

// Exports must be sorted alphabetically by name.
static const iree_vm_native_export_descriptor_t module_c_align_exports_[] = {
    {IREE_SV("entry"), IREE_SV("0i_i"), 0, NULL},
    {IREE_SV("mixed_i32_i64"), IREE_SV("0iI_I"), 0, NULL},
    {IREE_SV("mixed_i32_i64_i32"), IREE_SV("0iIi_I"), 0, NULL},
    {IREE_SV("mixed_i32_ref"), IREE_SV("0ir_i"), 0, NULL},
    {IREE_SV("mixed_i32x3_ref"), IREE_SV("0iiir_i"), 0, NULL},
    {IREE_SV("mixed_i64_i32"), IREE_SV("0Ii_i"), 0, NULL},
    {IREE_SV("mixed_ref_i32_ref"), IREE_SV("0rir_i"), 0, NULL},
};

static const iree_vm_native_function_ptr_t module_c_align_funcs_[] = {
    {(iree_vm_native_function_shim_t)iree_vm_shim_i_i,
     (iree_vm_native_function_target_t)module_c_align_entry},
    {(iree_vm_native_function_shim_t)iree_vm_shim_iI_I,
     (iree_vm_native_function_target_t)module_c_align_mixed_i32_i64},
    {(iree_vm_native_function_shim_t)iree_vm_shim_iIi_I,
     (iree_vm_native_function_target_t)module_c_align_mixed_i32_i64_i32},
    {(iree_vm_native_function_shim_t)iree_vm_shim_ir_i,
     (iree_vm_native_function_target_t)module_c_align_mixed_i32_ref},
    {(iree_vm_native_function_shim_t)iree_vm_shim_iiir_i,
     (iree_vm_native_function_target_t)module_c_align_mixed_i32x3_ref},
    {(iree_vm_native_function_shim_t)iree_vm_shim_Ii_i,
     (iree_vm_native_function_target_t)module_c_align_mixed_i64_i32},
    {(iree_vm_native_function_shim_t)iree_vm_shim_rir_i,
     (iree_vm_native_function_target_t)module_c_align_mixed_ref_i32_ref},
};

static_assert(IREE_ARRAYSIZE(module_c_align_funcs_) ==
                  IREE_ARRAYSIZE(module_c_align_exports_),
              "function pointer table must be 1:1 with exports");

static const iree_vm_native_module_descriptor_t module_c_align_descriptor_ = {
    /*name=*/IREE_SV("module_c"),
    /*version=*/0,
    /*attr_count=*/0,
    /*attrs=*/NULL,
    /*dependency_count=*/0,
    /*dependencies=*/NULL,
    /*import_count=*/0,
    /*imports=*/NULL,
    /*export_count=*/IREE_ARRAYSIZE(module_c_align_exports_),
    /*exports=*/module_c_align_exports_,
    /*function_count=*/IREE_ARRAYSIZE(module_c_align_funcs_),
    /*functions=*/module_c_align_funcs_,
};

static iree_status_t module_c_align_create(iree_vm_instance_t* instance,
                                           iree_allocator_t allocator,
                                           iree_vm_module_t** out_module) {
  module_c_align_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  module->allocator = allocator;

  iree_vm_module_t interface;
  iree_status_t status = iree_vm_module_initialize(&interface, module);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, module);
    return status;
  }
  interface.destroy = module_c_align_destroy;
  interface.alloc_state = module_c_align_alloc_state;
  interface.free_state = module_c_align_free_state;
  return iree_vm_native_module_create(&interface, &module_c_align_descriptor_,
                                      instance, allocator, out_module);
}

//===----------------------------------------------------------------------===//
// module_cpp
//===----------------------------------------------------------------------===//
// C++ implementation using native_module_cc.h to test alignment-sensitive
// parameter unpacking and result packing patterns. This module implements
// functions with mixed type signatures that can trigger alignment issues.

namespace {

struct ModuleCppState final {
  int counter = 0;

  // Same signature as C module_b.entry: (i32) -> i32
  iree::StatusOr<int32_t> Entry(int32_t arg0) {
    counter += arg0 + 1;  // Equivalent to add_1
    return counter - 1;   // Equivalent to sub_1
  }

  // (i32, ref<buffer>) -> i32 - tests ref after i32
  iree::StatusOr<int32_t> MixedI32Ref(int32_t arg0,
                                      iree::vm::ref<iree_vm_buffer_t> buf) {
    // Return arg0 + buffer length (or 0 if null).
    iree_host_size_t buf_len = buf ? iree_vm_buffer_length(buf.get()) : 0;
    return arg0 + static_cast<int32_t>(buf_len);
  }

  // (ref<buffer>, i32, ref<buffer>) -> i32 - tests ref after i32 after ref
  iree::StatusOr<int32_t> MixedRefI32Ref(iree::vm::ref<iree_vm_buffer_t> buf1,
                                         int32_t arg0,
                                         iree::vm::ref<iree_vm_buffer_t> buf2) {
    iree_host_size_t len1 = buf1 ? iree_vm_buffer_length(buf1.get()) : 0;
    iree_host_size_t len2 = buf2 ? iree_vm_buffer_length(buf2.get()) : 0;
    return static_cast<int32_t>(len1) + arg0 + static_cast<int32_t>(len2);
  }

  // (i32, i64) -> i64 - tests i64 after i32
  iree::StatusOr<int64_t> MixedI32I64(int32_t a, int64_t b) { return a + b; }

  // (i32, i32, i32, ref<buffer>) -> i32 - tests ref after 3 i32s (12 bytes)
  iree::StatusOr<int32_t> MixedI32x3Ref(int32_t a, int32_t b, int32_t c,
                                        iree::vm::ref<iree_vm_buffer_t> buf) {
    iree_host_size_t buf_len = buf ? iree_vm_buffer_length(buf.get()) : 0;
    return a + b + c + static_cast<int32_t>(buf_len);
  }

  // (i64, i32) -> i32 - tests i32 after i64
  iree::StatusOr<int32_t> MixedI64I32(int64_t a, int32_t b) {
    return static_cast<int32_t>(a) + b;
  }

  // (i32, i64, i32) -> i64 - tests i64 sandwiched between i32s
  iree::StatusOr<int64_t> MixedI32I64I32(int32_t a, int64_t b, int32_t c) {
    return a + b + c;
  }
};

static const iree::vm::NativeFunction<ModuleCppState> kModuleCppFunctions[] = {
    iree::vm::MakeNativeFunction("entry", &ModuleCppState::Entry),
    iree::vm::MakeNativeFunction("mixed_i32_ref", &ModuleCppState::MixedI32Ref),
    iree::vm::MakeNativeFunction("mixed_ref_i32_ref",
                                 &ModuleCppState::MixedRefI32Ref),
    iree::vm::MakeNativeFunction("mixed_i32_i64", &ModuleCppState::MixedI32I64),
    iree::vm::MakeNativeFunction("mixed_i32x3_ref",
                                 &ModuleCppState::MixedI32x3Ref),
    iree::vm::MakeNativeFunction("mixed_i64_i32", &ModuleCppState::MixedI64I32),
    iree::vm::MakeNativeFunction("mixed_i32_i64_i32",
                                 &ModuleCppState::MixedI32I64I32),
};

class ModuleCpp final : public iree::vm::NativeModule<ModuleCppState> {
 public:
  using iree::vm::NativeModule<ModuleCppState>::NativeModule;

 protected:
  iree::StatusOr<std::unique_ptr<ModuleCppState>> CreateState(
      iree_allocator_t allocator) override {
    return std::make_unique<ModuleCppState>();
  }

  iree::StatusOr<std::unique_ptr<ModuleCppState>> ForkState(
      ModuleCppState* parent_state, iree_allocator_t allocator) override {
    auto child_state = std::make_unique<ModuleCppState>();
    child_state->counter = parent_state->counter;
    return child_state;
  }
};

}  // namespace

static iree_status_t module_cpp_create(iree_vm_instance_t* instance,
                                       iree_allocator_t allocator,
                                       iree_vm_module_t** out_module) {
  auto module = std::make_unique<ModuleCpp>(
      "module_cpp", /*version=*/0, instance, allocator,
      iree::span<const iree::vm::NativeFunction<ModuleCppState>>{
          kModuleCppFunctions});
  *out_module = module.release()->interface();
  return iree_ok_status();
}
