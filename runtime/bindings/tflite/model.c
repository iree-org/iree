// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime/bindings/tflite/model.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/tracing.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/bytecode/module.h"

static iree_status_t _TfLiteModelCalculateFunctionIOCounts(
    const iree_vm_function_signature_t* signature, int32_t* out_input_count,
    int32_t* out_output_count) {
  iree_string_view_t arguments, results;
  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      signature, &arguments, &results));
  // NOTE: today we only pass 1:1 buffer views with what tflite does.
  // That means that both these should be one `r` per buffer view and our counts
  // are just the number of chars in the cconv.
  *out_input_count = (int32_t)arguments.size;
  *out_output_count = (int32_t)results.size;
  return iree_ok_status();
}

static iree_status_t _TfLiteModelInitializeModule(const void* flatbuffer_data,
                                                  size_t flatbuffer_size,
                                                  iree_allocator_t allocator,
                                                  TfLiteModel* model) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_vm_instance_create(allocator, &model->instance));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_module_register_all_types(model->instance));

  iree_const_byte_span_t flatbuffer_span =
      iree_make_const_byte_span(flatbuffer_data, flatbuffer_size);
  iree_allocator_t flatbuffer_allocator = iree_allocator_null();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_vm_bytecode_module_create(model->instance, flatbuffer_span,
                                     flatbuffer_allocator, allocator,
                                     &model->module),
      "error creating bytecode module");

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_vm_module_lookup_function_by_name(
          model->module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_make_cstring_view("_tflite_main"), &model->exports._main),
      "unable to find '_tflite_main' export in module, module must be compiled "
      "with tflite bindings support");

  // Get the input and output counts of the function; this is useful for being
  // able to preallocate storage when creating interpreters.
  iree_vm_function_signature_t main_signature =
      iree_vm_function_signature(&model->exports._main);
  IREE_RETURN_IF_ERROR(_TfLiteModelCalculateFunctionIOCounts(
      &main_signature, &model->input_count, &model->output_count));

  // NOTE: the input shape query is not required as it's possible (though
  // silly) for a model to have no inputs. In testing this can happen a lot
  // but in the wild it's rare ... says someone who previously filed bugs
  // against tflite because they didn't support models with no inputs when I
  // was being silly and needed them ;)
  IREE_IGNORE_ERROR(iree_vm_module_lookup_function_by_name(
      model->module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("_tflite_main_query_input_shape"),
      &model->exports._query_input_shape));

  // NOTE: the input shape resizing function is only required if the model has
  // dynamic shapes.
  IREE_IGNORE_ERROR(iree_vm_module_lookup_function_by_name(
      model->module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("_tflite_main_resize_input_shape"),
      &model->exports._resize_input_shape));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_vm_module_lookup_function_by_name(
          model->module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_make_cstring_view("_tflite_main_query_output_shape"),
          &model->exports._query_output_shape),
      "unable to find '_tflite_main_query_output_shape' export in module");

  // It's OK for this to fail; the model may not have variables.
  IREE_IGNORE_ERROR(iree_vm_module_lookup_function_by_name(
      model->module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("_tflite_main_reset_variables"),
      &model->exports._reset_variables));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreate(const void* model_data,
                                                      size_t model_size) {
  iree_allocator_t allocator = iree_allocator_system();
  IREE_TRACE_ZONE_BEGIN(z0);

  TfLiteModel* model = NULL;
  iree_status_t status =
      iree_allocator_malloc(allocator, sizeof(*model), (void**)&model);
  if (!iree_status_is_ok(iree_status_consume_code(status))) {
    IREE_TRACE_MESSAGE(ERROR, "failed model allocation");
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }
  memset(model, 0, sizeof(*model));
  iree_atomic_ref_count_init(&model->ref_count);
  model->allocator = allocator;

  status =
      _TfLiteModelInitializeModule(model_data, model_size, allocator, model);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    TfLiteModelDelete(model);
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }

  IREE_TRACE_ZONE_END(z0);
  return model;
}

TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreateFromFile(
    const char* model_path) {
  iree_allocator_t allocator = iree_allocator_system();
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(#3909): use file mapping C API.
  FILE* file = fopen(model_path, "r");
  if (!file) {
    IREE_TRACE_MESSAGE(ERROR, "failed to open model file");
    IREE_TRACE_MESSAGE_DYNAMIC(ERROR, model_path, strlen(model_path));
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  size_t file_size = ftell(file);
  fseek(file, 0, SEEK_SET);
  TfLiteModel* model = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, sizeof(TfLiteModel) + file_size, (void**)&model);
  if (!iree_status_is_ok(iree_status_consume_code(status))) {
    IREE_TRACE_MESSAGE(ERROR, "failed model+data allocation");
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }
  memset(model, 0, sizeof(*model));
  iree_atomic_ref_count_init(&model->ref_count);
  model->allocator = allocator;
  model->owned_model_data = (uint8_t*)model + file_size;
  int ret = fread(model->owned_model_data, 1, file_size, file);
  fclose(file);
  if (ret != file_size) {
    TfLiteModelDelete(model);
    IREE_TRACE_MESSAGE(ERROR, "failed model+data read");
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }

  status = _TfLiteModelInitializeModule(model->owned_model_data, file_size,
                                        allocator, model);
  if (!iree_status_is_ok(iree_status_consume_code(status))) {
    TfLiteModelDelete(model);
    IREE_TRACE_ZONE_END(z0);
    return NULL;
  }

  IREE_TRACE_ZONE_END(z0);
  return model;
}

void _TfLiteModelRetain(TfLiteModel* model) {
  if (model) {
    iree_atomic_ref_count_inc(&model->ref_count);
  }
}

void _TfLiteModelRelease(TfLiteModel* model) {
  if (model && iree_atomic_ref_count_dec(&model->ref_count) == 1) {
    IREE_TRACE_ZONE_BEGIN(z0);
    iree_vm_module_release(model->module);
    iree_vm_instance_release(model->instance);
    iree_allocator_free(model->allocator, model);
    IREE_TRACE_ZONE_END(z0);
  }
}

TFL_CAPI_EXPORT extern void TfLiteModelDelete(TfLiteModel* model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  _TfLiteModelRelease(model);
  IREE_TRACE_ZONE_END(z0);
}
