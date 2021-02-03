// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/local/loaders/vmla_module_loader.h"

#include "iree/base/tracing.h"
#include "iree/hal/local/local_descriptor_set_layout.h"
#include "iree/hal/local/local_executable.h"
#include "iree/modules/vmla/op_module.h"
#include "iree/vm/bytecode_module.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/vmla_executable_def_reader.h"
#include "iree/schemas/vmla_executable_def_verifier.h"

//===----------------------------------------------------------------------===//
// Verification and file utilities
//===----------------------------------------------------------------------===//

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime. There are still some conditions we must be aware of (such as omitted
// names on functions with internal linkage), however we shouldn't need to
// bounds check anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_vmla_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  // Special handling for valid but mismatching flatbuffers.
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16 ||
      !flatbuffers_has_identifier(flatbuffer_data.data,
                                  iree_VMLAExecutableDef_file_identifier)) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_VMLAExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_VMLAExecutableDef_table_t executable_def =
      iree_VMLAExecutableDef_as_root(flatbuffer_data.data);

  if (flatbuffers_uint8_vec_len(
          iree_VMLAExecutableDef_bytecode_module_get(executable_def)) < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable bytecode_module is missing/empty");
  }

  // NOTE: we don't check the actual bytecode module contents here; it's opaque
  // to us and passed on to the VM.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_vmla_executable_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_local_executable_t base;

  // Context containing both the VMLA module and the loaded executable.
  iree_vm_context_t* context;

  // Resolved entry functions from the module.
  iree_host_size_t entry_fn_count;
  iree_vm_function_t entry_fns[];
} iree_hal_vmla_executable_t;

extern const iree_hal_local_executable_vtable_t iree_hal_vmla_executable_vtable;

static iree_status_t iree_hal_vmla_executable_create(
    iree_vm_context_t* context, iree_vm_module_t* bytecode_module,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(bytecode_module);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t entry_count =
      iree_vm_module_signature(bytecode_module).export_function_count;
  if (entry_count != executable_layout_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable provides %zu entry points but caller "
                            "provided %zu; must match",
                            entry_count, executable_layout_count);
  }

  iree_hal_vmla_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) + entry_count * sizeof(*executable->entry_fns) +
      executable_layout_count * sizeof(iree_hal_local_executable_layout_t);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_layout_t** executable_layouts_ptr =
        (iree_hal_local_executable_layout_t**)(((uint8_t*)executable) +
                                               sizeof(*executable) +
                                               entry_count *
                                                   sizeof(
                                                       *executable->entry_fns));
    iree_hal_local_executable_initialize(
        &iree_hal_vmla_executable_vtable, executable_layout_count,
        executable_layouts, executable_layouts_ptr, host_allocator,
        &executable->base);
    executable->context = context;
    iree_vm_context_retain(executable->context);

    executable->entry_fn_count = entry_count;
    for (iree_host_size_t i = 0; i < executable->entry_fn_count; ++i) {
      status = iree_vm_module_lookup_function_by_ordinal(
          bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i,
          &executable->entry_fns[i], NULL);
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vmla_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_vmla_executable_t* executable =
      (iree_hal_vmla_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_context_release(executable->context);
  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vmla_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_local_executable_call_t* call) {
  iree_hal_vmla_executable_t* executable =
      (iree_hal_vmla_executable_t*)base_executable;

  if (IREE_UNLIKELY(ordinal >= executable->entry_fn_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_string_view_t entry_point_name =
      iree_vm_function_name(&executable->entry_fns[ordinal]);
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_vmla_call");
  }
  IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, entry_point_name.data,
                                      entry_point_name.size);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  // We churn memory here but I don't rightly care: this entire VMLA approach is
  // deprecated and will be going away at some point. There's about 100
  // low-hanging branches we can hack at in the compiler before this extra
  // allocation matters :)
  iree_allocator_t host_allocator = executable->base.host_allocator;
  iree::hal::vmla::Interface interface;
  iree_vm_ref_t interface_ref = Interface_retain_ref(&interface);
  iree_host_size_t input_list_size = iree_vm_list_storage_size(
      /*element_type=*/NULL, /*interface*/ 1 + /*workgroup_xyz[3]*/ 3);
  void* input_list_storage = iree_alloca(input_list_size);
  iree_vm_list_t* input_list = NULL;
  IREE_CHECK_OK(iree_vm_list_initialize(
      iree_make_byte_span(input_list_storage, input_list_size),
      /*element_type=*/NULL,
      /*interface*/ 1 + /*workgroup_xyz[3]*/ 3, &input_list));
  iree_vm_list_push_ref_retain(input_list, &interface_ref);
  iree_vm_value_t workgroup_id_x = iree_vm_value_make_i32(call->workgroup_id.x);
  iree_vm_value_t workgroup_id_y = iree_vm_value_make_i32(call->workgroup_id.y);
  iree_vm_value_t workgroup_id_z = iree_vm_value_make_i32(call->workgroup_id.z);
  iree_vm_list_push_value(input_list, &workgroup_id_x);
  iree_vm_list_push_value(input_list, &workgroup_id_y);
  iree_vm_list_push_value(input_list, &workgroup_id_z);

  iree_hal_local_executable_layout_t* local_layout =
      executable->base.executable_layouts[ordinal];
  IREE_CHECK_OK(interface.SetConstants(
      absl::MakeConstSpan(call->push_constants, local_layout->push_constants)));

  for (iree_host_size_t set_ordinal = 0;
       set_ordinal < local_layout->set_layout_count; ++set_ordinal) {
    iree_hal_local_descriptor_set_layout_t* local_set_layout =
        iree_hal_local_descriptor_set_layout_cast(
            local_layout->set_layouts[set_ordinal]);
    for (iree_host_size_t i = 0; i < local_set_layout->binding_count; ++i) {
      auto buffer_or = iree::hal::vmla::Buffer::WrapMutable(
          call->bindings[i], call->binding_lengths[i], iree_allocator_null());
      if (!buffer_or.ok()) {
        IREE_CHECK_OK(std::move(buffer_or).status());
      }
      IREE_CHECK_OK(interface.SetBinding(set_ordinal,
                                         local_set_layout->bindings[i].binding,
                                         {std::move(buffer_or.value())}));
    }
  }

  iree_status_t status =
      iree_vm_invoke(executable->context, executable->entry_fns[ordinal],
                     /*policy=*/NULL, input_list,
                     /*outputs=*/NULL, host_allocator);

  iree_vm_list_deinitialize(input_list);
  iree_vm_ref_release(&interface_ref);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_local_executable_vtable_t iree_hal_vmla_executable_vtable = {
    /*.base=*/
    {
        /*.destroy=*/iree_hal_vmla_executable_destroy,
    },
    /*.issue_call=*/iree_hal_vmla_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_vmla_module_loader_t
//===----------------------------------------------------------------------===//

typedef struct {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
  iree_vm_instance_t* instance;
  iree_vm_module_t* vmla_module;
} iree_hal_vmla_module_loader_t;

extern const iree_hal_executable_loader_vtable_t
    iree_hal_vmla_module_loader_vtable;

iree_status_t iree_hal_vmla_module_loader_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // A single VMLA module is shared across all loaded executables.
  IREE_RETURN_IF_ERROR(iree::hal::vmla::ModuleRegisterTypes());
  iree_vm_module_t* vmla_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree::hal::vmla::ModuleCreate(host_allocator, &vmla_module));

  iree_hal_vmla_module_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(&iree_hal_vmla_module_loader_vtable,
                                          &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    executable_loader->instance = instance;
    iree_vm_instance_retain(executable_loader->instance);
    executable_loader->vmla_module = vmla_module;
    iree_vm_module_retain(executable_loader->vmla_module);
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  iree_vm_module_release(vmla_module);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vmla_module_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_vmla_module_loader_t* executable_loader =
      (iree_hal_vmla_module_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_vm_module_release(executable_loader->vmla_module);
  iree_vm_instance_release(executable_loader->instance);
  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_vmla_module_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_hal_executable_format_t executable_format) {
  return executable_format == iree_hal_make_executable_format("VMLA");
}

static iree_status_t iree_hal_vmla_module_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  iree_hal_vmla_module_loader_t* executable_loader =
      (iree_hal_vmla_module_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify that we have a valid flatbuffer that contains a VMLA executable.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_vmla_executable_flatbuffer_verify(
                                        executable_spec->executable_data));
  iree_VMLAExecutableDef_table_t executable_def =
      iree_VMLAExecutableDef_as_root(executable_spec->executable_data.data);
  flatbuffers_uint8_vec_t bytecode_module_vec =
      iree_VMLAExecutableDef_bytecode_module_get(executable_def);
  iree_const_byte_span_t bytecode_module_data = iree_make_const_byte_span(
      bytecode_module_vec, flatbuffers_uint8_vec_len(bytecode_module_vec));

  // If the caching mode allows for aliasing the existing flatbuffer data then
  // we avoid allocations and just pass the pointer on through. The caller
  // ensures that the data remains valid for the duration the executable is
  // loaded. Otherwise, we clone it and let the bytecode module take ownership.
  iree_allocator_t bytecode_module_allocator;
  if (iree_all_bits_set(executable_spec->caching_mode,
                        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA)) {
    // Zero-copy route.
    bytecode_module_allocator = iree_allocator_null();
  } else {
    bytecode_module_allocator = executable_loader->host_allocator;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_clone(executable_loader->host_allocator,
                                 bytecode_module_data,
                                 (void**)&bytecode_module_data.data));
  }

  // Load the user-provided bytecode module. We pass ownership of the data (if
  // we have it) to the module to manage.
  iree_vm_module_t* bytecode_module = NULL;
  iree_status_t status = iree_vm_bytecode_module_create(
      bytecode_module_data, bytecode_module_allocator,
      executable_loader->host_allocator, &bytecode_module);

  // Create the context tying together the shared VMLA module and the
  // user-provided module that references it. If we wanted to allow custom
  // modules here for user-provided functions we'd mix them in here.
  iree_vm_context_t* context = NULL;
  if (iree_status_is_ok(status)) {
    iree_vm_module_t* modules[2] = {
        executable_loader->vmla_module,
        bytecode_module,
    };
    status = iree_vm_context_create_with_modules(
        executable_loader->instance, modules, IREE_ARRAYSIZE(modules),
        executable_loader->host_allocator, &context);
  }

  // Executable takes ownership of the entire context (including the bytecode
  // module, which itself may own the underlying allocation).
  if (iree_status_is_ok(status)) {
    status = iree_hal_vmla_executable_create(
        context, bytecode_module, executable_spec->executable_layout_count,
        executable_spec->executable_layouts, executable_loader->host_allocator,
        out_executable);
  }

  iree_vm_context_release(context);
  iree_vm_module_release(bytecode_module);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_executable_loader_vtable_t iree_hal_vmla_module_loader_vtable = {
    /*.destroy=*/iree_hal_vmla_module_loader_destroy,
    /*.query_support=*/iree_hal_vmla_module_loader_query_support,
    /*.try_load=*/iree_hal_vmla_module_loader_try_load,
};
