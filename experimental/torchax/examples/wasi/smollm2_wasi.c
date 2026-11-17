// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Executes prefill/decode with functional f32 state and prepared Torch results.
// Inputs are concatenated NPY arrays; see smollm2_wasm.py for their
// preparation.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/io/file_contents.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/parameter_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#if defined(IREE_TORCHAX_STATIC_MODEL)
#include "iree/hal/local/loaders/static_library_loader.h"
// Generated from the compiler-produced headers for this set of model objects.
#include "static_libraries.h"
#else
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#endif

IREE_FLAG(string, prefill, "", "Prefill VMFB path.");
IREE_FLAG(string, decode, "", "Decode VMFB path.");
IREE_FLAG(string, case_directory, "",
          "Directory containing prepared NPY arrays.");
IREE_FLAG(int32_t, state_count, 60,
          "Number of functional cache arguments (1-64).");
IREE_FLAG(int32_t, steps, 3, "Prefill plus decode invocation count.");
IREE_FLAG(bool, retain_state, false,
          "Check that physical input cache buffers remain unchanged.");

enum { IREE_TORCHAX_MAX_STATE_COUNT = 64 };

typedef struct iree_torchax_snapshot_t {
  void* data;
  iree_host_size_t length;
} iree_torchax_snapshot_t;

static iree_status_t load_module(iree_vm_instance_t* instance, const char* path,
                                 iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_read(
      iree_make_cstring_view(path), iree_allocator_system(), &contents));
  iree_status_t status = iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE, contents->const_buffer,
      iree_io_file_contents_deallocator(contents), iree_allocator_system(),
      out_module);
  if (!iree_status_is_ok(status)) {
    iree_io_file_contents_free(contents);
  }
  return status;
}

// The parser owns the new list and its buffers; no NPY file storage is
// borrowed.
static iree_status_t load_arrays(iree_hal_device_t* device, const char* name,
                                 int count, iree_vm_list_t** out_values) {
  *out_values = NULL;
  char path[4096];
  int length =
      snprintf(path, sizeof(path), "*%s/%s", FLAG_case_directory, name);
  if (length < 0 || (iree_host_size_t)length >= sizeof(path)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "NPY path is too long");
  }
  char refs[IREE_TORCHAX_MAX_STATE_COUNT + 2];
  memset(refs, 'r', count);
  iree_string_view_t spec = iree_make_cstring_view(path);
  IREE_RETURN_IF_ERROR(iree_tooling_parse_variants(
      iree_make_string_view(refs, count),
      (iree_string_view_list_t){.count = 1, .values = &spec}, device,
      iree_hal_device_allocator(device), iree_allocator_system(), out_values));
  if (iree_vm_list_size(*out_values) != (iree_host_size_t)count) {
    iree_vm_list_release(*out_values);
    *out_values = NULL;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected %d arrays in %s", count, name);
  }
  return iree_ok_status();
}

static iree_status_t append_values(iree_vm_list_t* source, int count,
                                   iree_vm_list_t* destination) {
  for (int i = 0; i < count; ++i) {
    iree_vm_variant_t value;
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant_assign(source, i, &value));
    IREE_RETURN_IF_ERROR(iree_vm_list_push_variant_retain(destination, &value));
  }
  return iree_ok_status();
}

static iree_status_t copy_view_to_host(iree_hal_buffer_view_t* view,
                                       iree_torchax_snapshot_t* out_snapshot) {
  *out_snapshot = (iree_torchax_snapshot_t){0};
  if (!view) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a HAL buffer view");
  }
  iree_device_size_t length = iree_hal_buffer_view_byte_length(view);
  if (length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "buffer does not fit in host memory");
  }
  void* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), length ? (iree_host_size_t)length : 1, &data));
  iree_status_t status = iree_hal_buffer_map_read(
      iree_hal_buffer_view_buffer(view), 0, data, length);
  if (iree_status_is_ok(status)) {
    out_snapshot->data = data;
    out_snapshot->length = (iree_host_size_t)length;
  } else {
    iree_allocator_free(iree_allocator_system(), data);
  }
  return status;
}

static void release_snapshots(int count, iree_torchax_snapshot_t* snapshots) {
  for (int i = 0; i < count; ++i) {
    iree_allocator_free(iree_allocator_system(), snapshots[i].data);
  }
}

static iree_status_t snapshot_caches(iree_vm_list_t* state,
                                     iree_torchax_snapshot_t* snapshots) {
  for (int i = 0; i < FLAG_state_count; ++i) {
    IREE_RETURN_IF_ERROR(copy_view_to_host(
        iree_vm_list_get_buffer_view_assign(state, i), &snapshots[i]));
  }
  return iree_ok_status();
}

static iree_status_t compare_retained_caches(
    iree_vm_list_t* state, const iree_torchax_snapshot_t* snapshots) {
  for (int i = 0; i < FLAG_state_count; ++i) {
    iree_torchax_snapshot_t after = {0};
    IREE_RETURN_IF_ERROR(copy_view_to_host(
        iree_vm_list_get_buffer_view_assign(state, i), &after));
    bool unchanged = after.length == snapshots[i].length &&
                     memcmp(after.data, snapshots[i].data, after.length) == 0;
    iree_allocator_free(iree_allocator_system(), after.data);
    if (!unchanged) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "retained input cache %d changed", i);
    }
  }
  return iree_ok_status();
}

static iree_status_t validate_output_views(iree_hal_buffer_view_t* actual,
                                           iree_hal_buffer_view_t* expected) {
  if (!actual || !expected ||
      iree_hal_buffer_view_element_type(actual) !=
          IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
      iree_hal_buffer_view_element_type(expected) !=
          IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
      iree_hal_buffer_view_encoding_type(actual) !=
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR ||
      iree_hal_buffer_view_encoding_type(expected) !=
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR ||
      iree_hal_buffer_view_shape_rank(actual) !=
          iree_hal_buffer_view_shape_rank(expected) ||
      memcmp(
          iree_hal_buffer_view_shape_dims(actual),
          iree_hal_buffer_view_shape_dims(expected),
          iree_hal_buffer_view_shape_rank(actual) * sizeof(iree_hal_dim_t)) ||
      iree_hal_buffer_view_byte_length(actual) !=
          iree_hal_buffer_view_byte_length(expected)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "output type, shape, or encoding mismatch");
  }
  return iree_ok_status();
}

static iree_host_size_t find_argmax(const float* values,
                                    iree_host_size_t count) {
  iree_host_size_t best = 0;
  for (iree_host_size_t i = 1; i < count; ++i) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

static iree_status_t compare_logits(iree_hal_buffer_view_t* view,
                                    const float* actual, const float* expected,
                                    int* out_token) {
  if (iree_hal_buffer_view_shape_rank(view) != 3 ||
      iree_hal_buffer_view_shape_dim(view, 0) != 1 ||
      iree_hal_buffer_view_shape_dim(view, 1) == 0 ||
      iree_hal_buffer_view_shape_dim(view, 2) == 0 ||
      iree_hal_buffer_view_shape_dim(view, 2) > INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "logits must have shape 1 x rows x vocabulary");
  }
  iree_host_size_t width = iree_hal_buffer_view_shape_dim(view, 2);
  iree_host_size_t start =
      iree_hal_buffer_view_byte_length(view) / sizeof(float) - width;
  iree_host_size_t token = find_argmax(actual + start, width);
  if (token != find_argmax(expected + start, width)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "predicted token mismatch");
  }
  *out_token = (int)token;
  return iree_ok_status();
}

static iree_status_t compare_output(iree_hal_buffer_view_t* actual,
                                    iree_hal_buffer_view_t* expected,
                                    bool is_logits, double* out_max_error,
                                    int* out_token) {
  IREE_RETURN_IF_ERROR(validate_output_views(actual, expected));
  iree_torchax_snapshot_t a = {0}, b = {0};
  iree_status_t status = copy_view_to_host(actual, &a);
  if (iree_status_is_ok(status)) status = copy_view_to_host(expected, &b);
  if (iree_status_is_ok(status)) {
    const float* x = a.data;
    const float* y = b.data;
    for (iree_host_size_t i = 0; i < a.length / sizeof(float); ++i) {
      double error = fabs((double)x[i] - (double)y[i]);
      if (!isfinite(x[i]) || !isfinite(y[i]) ||
          error > 1e-4 + 1e-4 * fabs((double)y[i])) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "element %" PRIhsz ": %.9g vs %.9g", i, x[i],
                                  y[i]);
        break;
      }
      *out_max_error = fmax(*out_max_error, error);
    }
    if (iree_status_is_ok(status) && is_logits) {
      status = compare_logits(actual, x, y, out_token);
    }
  }
  iree_allocator_free(iree_allocator_system(), a.data);
  iree_allocator_free(iree_allocator_system(), b.data);
  return status;
}

static iree_status_t compare_results(iree_vm_list_t* actual,
                                     iree_vm_list_t* expected, int step) {
  if (iree_vm_list_size(actual) != (iree_host_size_t)FLAG_state_count + 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all cache outputs followed by logits");
  }
  double max_cache_error = 0, max_logit_error = 0;
  int token = -1;
  for (int i = 0; i <= FLAG_state_count; ++i) {
    bool is_logits = i == FLAG_state_count;
    iree_status_t status = compare_output(
        iree_vm_list_get_buffer_view_assign(actual, i),
        iree_vm_list_get_buffer_view_assign(expected, i), is_logits,
        is_logits ? &max_logit_error : &max_cache_error, &token);
    if (!iree_status_is_ok(status)) {
      return iree_status_annotate_f(status, "step %d output %d", step, i);
    }
  }
  printf(
      "{\"step\":%d,\"next_token\":%d,\"max_cache_error\":%.9g,"
      "\"max_logit_error\":%.9g,\"retained_state_checked\":%s}\n",
      step, token, max_cache_error, max_logit_error,
      FLAG_retain_state ? "true" : "false");
  fflush(stdout);
  return iree_ok_status();
}

static iree_status_t append_step_inputs(iree_hal_device_t* device, int step,
                                        iree_vm_list_t* state) {
  char name[64];
  snprintf(name, sizeof(name), "step%d-inputs.npy", step);
  iree_vm_list_t* inputs = NULL;
  iree_status_t status = load_arrays(device, name, 2, &inputs);
  if (iree_status_is_ok(status)) status = append_values(inputs, 2, state);
  iree_vm_list_release(inputs);
  return status;
}

// The caller retains the input list until after validation. Each next-state
// reference is retained before releasing the output list that supplied it.
static iree_status_t run_step(iree_hal_device_t* device,
                              iree_vm_context_t* context,
                              iree_vm_function_t function,
                              iree_vm_list_t* state, int step,
                              iree_vm_list_t** out_next) {
  *out_next = NULL;
  iree_vm_list_t *outputs = NULL, *expected = NULL;
  iree_torchax_snapshot_t snapshots[IREE_TORCHAX_MAX_STATE_COUNT] = {0};
  iree_status_t status = append_step_inputs(device, step, state);
  if (iree_status_is_ok(status) && FLAG_retain_state) {
    status = snapshot_caches(state, snapshots);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                 FLAG_state_count + 1, iree_allocator_system(),
                                 &outputs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                            NULL, state, outputs, iree_allocator_system());
  }
  if (iree_status_is_ok(status) && FLAG_retain_state) {
    status = compare_retained_caches(state, snapshots);
  }
  char name[64];
  snprintf(name, sizeof(name), "step%d-expected.npy", step);
  if (iree_status_is_ok(status)) {
    status = load_arrays(device, name, FLAG_state_count + 1, &expected);
  }
  if (iree_status_is_ok(status))
    status = compare_results(outputs, expected, step);
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_list_create(iree_vm_make_undefined_type_def(), FLAG_state_count,
                            iree_allocator_system(), out_next);
  }
  if (iree_status_is_ok(status)) {
    status = append_values(outputs, FLAG_state_count, *out_next);
  }
  iree_vm_list_release(outputs);
  iree_vm_list_release(expected);
  release_snapshots(FLAG_state_count, snapshots);
  return status;
}

static iree_status_t run_sequence(iree_hal_device_t* device,
                                  iree_vm_context_t** contexts,
                                  const iree_vm_function_t* functions) {
  iree_vm_list_t* state = NULL;
  iree_status_t status =
      load_arrays(device, "initial.npy", FLAG_state_count, &state);
  for (int step = 0; step < FLAG_steps && iree_status_is_ok(status); ++step) {
    int program = step == 0 ? 0 : 1;
    iree_vm_list_t* next = NULL;
    status = run_step(device, contexts[program], functions[program], state,
                      step, &next);
    iree_vm_list_release(state);
    state = next;
  }
  iree_vm_list_release(state);
  return status;
}

static iree_status_t create_device(iree_vm_instance_t* instance,
                                   iree_hal_device_t** out_device) {
  *out_device = NULL;
  iree_allocator_t host = iree_allocator_system();
  iree_hal_executable_loader_t* loader = NULL;
#if defined(IREE_TORCHAX_STATIC_MODEL)
  IREE_RETURN_IF_ERROR(iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(iree_torchax_model_libraries),
      iree_torchax_model_libraries, iree_hal_executable_import_provider_null(),
      host, &loader));
#else
  IREE_RETURN_IF_ERROR(
      iree_hal_vmvx_module_loader_create(instance, 0, NULL, host, &loader));
#endif
  iree_hal_allocator_t* allocator = NULL;
  iree_async_proactor_pool_t* pool = NULL;
  iree_status_t status =
      iree_hal_allocator_create_heap(IREE_SV("wasm"), host, host, &allocator);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        1, NULL, iree_async_proactor_pool_options_default(), host, &pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_sync_device_params_t params;
    iree_hal_sync_device_params_initialize(&params);
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = pool;
    status = iree_hal_sync_device_create(IREE_SV("local-sync"), &params,
                                         &create_params, 1, &loader, allocator,
                                         host, out_device);
  }
  iree_async_proactor_pool_release(pool);
  iree_hal_allocator_release(allocator);
  iree_hal_executable_loader_release(loader);
  return status;
}

static iree_status_t create_context(iree_vm_instance_t* instance,
                                    iree_vm_module_t* parameters,
                                    iree_vm_module_t* hal, const char* path,
                                    iree_vm_context_t** out_context,
                                    iree_vm_function_t* out_function) {
  iree_vm_module_t* bytecode = NULL;
  IREE_RETURN_IF_ERROR(load_module(instance, path, &bytecode));
  iree_vm_module_t* modules[] = {parameters, hal, bytecode};
  iree_status_t status = iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      iree_allocator_system(), out_context);
  if (iree_status_is_ok(status)) {
    status = iree_vm_module_lookup_function_by_name(
        bytecode, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("main"),
        out_function);
  }
  iree_vm_module_release(bytecode);
  return status;
}

static iree_status_t create_contexts(iree_vm_instance_t* instance,
                                     iree_hal_device_t* device,
                                     iree_vm_context_t** contexts,
                                     iree_vm_function_t* functions) {
  iree_allocator_t host = iree_allocator_system();
  iree_async_frontier_tracker_t* tracker = NULL;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), host, &tracker));
  iree_hal_device_group_t* group = NULL;
  iree_vm_module_t *parameters = NULL, *hal = NULL;
  iree_status_t status =
      iree_hal_device_group_create_from_device(device, tracker, host, &group);
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_parameters_module_from_flags(
        instance, 0, NULL, host, &parameters);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_module_create(
        instance, iree_hal_module_device_policy_default(), group,
        IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
        iree_hal_module_debug_sink_stdio(stderr), host, &hal);
  }
  const char* paths[] = {FLAG_prefill, FLAG_decode};
  for (int i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    status = create_context(instance, parameters, hal, paths[i], &contexts[i],
                            &functions[i]);
  }
  iree_vm_module_release(parameters);
  iree_vm_module_release(hal);
  iree_hal_device_group_release(group);
  iree_async_frontier_tracker_release(tracker);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_steps < 1 || FLAG_state_count < 1 ||
      FLAG_state_count > IREE_TORCHAX_MAX_STATE_COUNT || !FLAG_prefill[0] ||
      !FLAG_decode[0] || !FLAG_case_directory[0]) {
    fprintf(stderr,
            "require prefill, decode, case_directory, positive steps, "
            "and state_count in [1,64]\n");
    return 1;
  }
  iree_vm_instance_t* instance = NULL;
  iree_hal_device_t* device = NULL;
  iree_vm_context_t* contexts[2] = {0};
  iree_vm_function_t functions[2] = {0};
  iree_status_t status = iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance);
  if (iree_status_is_ok(status))
    status = iree_hal_module_register_all_types(instance);
  if (iree_status_is_ok(status)) status = create_device(instance, &device);
  if (iree_status_is_ok(status)) {
    status = create_contexts(instance, device, contexts, functions);
  }
  if (iree_status_is_ok(status))
    status = run_sequence(device, contexts, functions);
  for (int i = 0; i < 2; ++i) iree_vm_context_release(contexts[i]);
  iree_hal_device_release(device);
  iree_vm_instance_release(instance);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}
