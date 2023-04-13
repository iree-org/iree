// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "runtime/bindings/tflite/interpreter.h"

#include "iree/base/internal/call_once.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/init.h"
#include "iree/modules/hal/module.h"
#include "runtime/bindings/tflite/model.h"
#include "runtime/bindings/tflite/shim.h"
#include "runtime/bindings/tflite/tensor.h"

//===----------------------------------------------------------------------===//
// HAL / driver support
//===----------------------------------------------------------------------===//

static iree_once_flag _TfLiteInterpreterRegisterDriverFlag =
    IREE_ONCE_FLAG_INIT;
static void _TfLiteInterpreterRegisterDrivers(void) {
  IREE_IGNORE_ERROR(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));
}

// TODO(#3977): if already provided a HAL device in the options use that.
static iree_status_t _TfLiteInterpreterPrepareHAL(
    TfLiteInterpreter* interpreter) {
  iree_call_once(&_TfLiteInterpreterRegisterDriverFlag,
                 _TfLiteInterpreterRegisterDrivers);

  iree_hal_driver_registry_t* driver_registry =
      iree_hal_driver_registry_default();

  iree_host_size_t driver_info_count = 0;
  iree_hal_driver_info_t* driver_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
      driver_registry, interpreter->allocator, &driver_info_count,
      &driver_infos));

  // TODO(benvanik): figure out how we want to emulate device selection; may
  // just say "whatever is first" on a query.
  // iree_string_view_t driver_name = driver_infos[0].driver_name;
  // NOTE: currently the sample file is compiled only with vmvx.
  iree_string_view_t driver_name = iree_make_cstring_view("local-task");

  // TODO(benvanik): switch to iree_hal_driver_registry_try_create when
  // implemented.
  iree_status_t status = iree_hal_driver_registry_try_create(
      driver_registry, driver_name, interpreter->allocator,
      &interpreter->driver);
  iree_allocator_free(interpreter->allocator, driver_infos);
  IREE_RETURN_IF_ERROR(status, "failed to create driver '%.*s'",
                       (int)driver_name.size, driver_name.data);

  IREE_RETURN_IF_ERROR(
      iree_hal_driver_create_default_device(
          interpreter->driver, interpreter->allocator, &interpreter->device),
      "failed creating the default device for driver '%.*s'",
      (int)driver_name.size, driver_name.data);

  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      interpreter->instance, interpreter->device, IREE_HAL_MODULE_FLAG_NONE,
      interpreter->allocator, &interpreter->hal_module));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Model shape function query/mutation utilities
//===----------------------------------------------------------------------===//

// On-stack storage for shape function invocations.
// Avoids all allocations and allows for reuse when running down lists of
// inputs and outputs calling shape functions.
typedef struct {
  // Inlined list for the !vm.list in the shape function arguments.
  uint8_t
      shape_list_storage[128 + sizeof(int32_t) * IREE_BINDINGS_TFLITE_MAX_RANK];
  iree_vm_list_t* shape_list;

  // Inlined list for the shape function arguments.
  uint8_t arg_list_storage[128 + sizeof(uintptr_t) * 2];
  iree_vm_list_t* arg_list;
} _TfLiteInterpreterShapeFrame;

// Initializes an on-stack shape frame. Existing contents are discarded.
static iree_status_t _TfLiteInterpreterShapeFrameInitialize(
    _TfLiteInterpreterShapeFrame* frame) {
  // [int32...] storage for the shape dimension inputs/outputs.
  iree_vm_type_def_t dim_type =
      iree_vm_make_value_type_def(IREE_VM_VALUE_TYPE_I32);
  IREE_RETURN_IF_ERROR(iree_vm_list_initialize(
      iree_make_byte_span(frame->shape_list_storage,
                          IREE_ARRAYSIZE(frame->shape_list_storage)),
      &dim_type, IREE_BINDINGS_TFLITE_MAX_RANK, &frame->shape_list));

  // (%index : i32, %shape : !vm.list<i32>)
  IREE_RETURN_IF_ERROR(iree_vm_list_initialize(
      iree_make_byte_span(frame->arg_list_storage,
                          IREE_ARRAYSIZE(frame->arg_list_storage)),
      /*element_type=*/NULL, /*index*/ 1 + /*shape*/ 1, &frame->arg_list));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(frame->arg_list, 2));

  // Arg 1 is always the shape list for all I/O, so do that once here.
  iree_vm_ref_t shape_list_ref = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
      frame->shape_list, iree_vm_list_type(), &shape_list_ref));
  IREE_RETURN_IF_ERROR(
      iree_vm_list_set_ref_retain(frame->arg_list, 1, &shape_list_ref));

  return iree_ok_status();
}

// Deinitializes an on-stack shape frame.
// Though this does not free the frame memory (it's on the stack, afterall) it
// will release any resources that may be retained and is required.
static void _TfLiteInterpreterShapeFrameDeinitialize(
    _TfLiteInterpreterShapeFrame* frame) {
  iree_vm_list_deinitialize(frame->arg_list);
  iree_vm_list_deinitialize(frame->shape_list);
}

// Reads the shape value in the frame storage from the prior application.
static iree_status_t _TfLiteInterpreterShapeFrameReadValue(
    _TfLiteInterpreterShapeFrame* frame, int32_t* out_shape_rank,
    int32_t* out_shape_dims) {
  *out_shape_rank = (int32_t)iree_vm_list_size(frame->shape_list);
  for (int32_t i = 0; i < *out_shape_rank; ++i) {
    iree_vm_value_t dim;
    IREE_RETURN_IF_ERROR(iree_vm_list_get_value_as(
        frame->shape_list, i, IREE_VM_VALUE_TYPE_I32, &dim));
    out_shape_dims[i] = dim.i32;
  }
  return iree_ok_status();
}

// Writes the shape value to the current frame storage for future applications.
static iree_status_t _TfLiteInterpreterShapeFrameWriteValue(
    _TfLiteInterpreterShapeFrame* frame, int32_t shape_rank,
    const int32_t* shape_dims) {
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(frame->shape_list, shape_rank));
  for (int32_t i = 0; i < shape_rank; ++i) {
    iree_vm_value_t dim = iree_vm_value_make_i32(shape_dims[i]);
    IREE_RETURN_IF_ERROR(iree_vm_list_set_value(frame->shape_list, i, &dim));
  }
  return iree_ok_status();
}

// Calls the |apply_fn| with the current shape frame state.
static iree_status_t _TfLiteInterpreterShapeFrameApply(
    _TfLiteInterpreterShapeFrame* frame, TfLiteInterpreter* interpreter,
    iree_vm_function_t apply_fn, int32_t index) {
  // Populate shape_list with the shape dimensions for this particular output.
  iree_vm_value_t index_value = iree_vm_value_make_i32(index);
  IREE_IGNORE_ERROR(iree_vm_list_set_value(frame->arg_list, 0, &index_value));
  return iree_vm_invoke(interpreter->context, apply_fn,
                        IREE_VM_INVOCATION_FLAG_NONE,
                        /*policy=*/NULL, frame->arg_list, /*outputs=*/NULL,
                        interpreter->allocator);
}

//===----------------------------------------------------------------------===//
// Shape I/O queries
//===----------------------------------------------------------------------===//

// Queries all input shapes from the module; some may still be dynamic (-1).
static iree_status_t _TfLiteInterpreterRefreshInputShapes(
    TfLiteInterpreter* interpreter, _TfLiteInterpreterShapeFrame* frame) {
  // NOTE: we could optimize this more by using iree_vm_invoke_within, but that
  // shouldn't be needed (it's just stack pointer manipulation).
  for (int32_t i = 0; i < interpreter->model->input_count; ++i) {
    TfLiteTensor* tensor = &interpreter->input_tensors[i];
    IREE_RETURN_IF_ERROR(_TfLiteInterpreterShapeFrameApply(
        frame, interpreter, interpreter->model->exports._query_input_shape, i));
    IREE_RETURN_IF_ERROR(_TfLiteInterpreterShapeFrameReadValue(
        frame, &tensor->shape_rank, tensor->shape_dims));
  }
  return iree_ok_status();
}

// Queries all output shapes from the module allowing it use the current input
// shapes to compute the possibly dynamic values.
static iree_status_t _TfLiteInterpreterRefreshOutputShapes(
    TfLiteInterpreter* interpreter, _TfLiteInterpreterShapeFrame* frame) {
  // NOTE: we could optimize this more by using iree_vm_invoke_within, but that
  // shouldn't be needed (it's just stack pointer manipulation).
  for (int32_t i = 0; i < interpreter->model->output_count; ++i) {
    TfLiteTensor* tensor = &interpreter->output_tensors[i];
    IREE_RETURN_IF_ERROR(_TfLiteInterpreterShapeFrameApply(
        frame, interpreter, interpreter->model->exports._query_output_shape,
        i));
    IREE_RETURN_IF_ERROR(_TfLiteInterpreterShapeFrameReadValue(
        frame, &tensor->shape_rank, tensor->shape_dims));
  }
  return iree_ok_status();
}

// Refreshes both input and output tensor shapes by querying the module.
// This should be called after each shape change so that we can let the module
// run "shape propagation" and compute the new output shapes.
static iree_status_t _TfLiteInterpreterRefreshIOShapes(
    TfLiteInterpreter* interpreter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  _TfLiteInterpreterShapeFrame frame;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, _TfLiteInterpreterShapeFrameInitialize(&frame));

  // Query all shapes.
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = _TfLiteInterpreterRefreshInputShapes(interpreter, &frame);
  }
  if (iree_status_is_ok(status)) {
    status = _TfLiteInterpreterRefreshOutputShapes(interpreter, &frame);
  }

  _TfLiteInterpreterShapeFrameDeinitialize(&frame);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Creation and static initialization
//===----------------------------------------------------------------------===//

// Computes the storage requirement for the TfLiteInterpreter struct.
static iree_host_size_t _TfLiteInterpreterCalculateSize(
    const TfLiteModel* model) {
  iree_host_size_t total_size =
      iree_host_align(sizeof(TfLiteInterpreter), iree_max_align_t);

  iree_vm_type_def_t buffer_view_type_def =
      iree_vm_make_ref_type_def(iree_hal_buffer_type());
  total_size +=
      iree_vm_list_storage_size(&buffer_view_type_def, model->input_count);
  total_size +=
      iree_vm_list_storage_size(&buffer_view_type_def, model->output_count);
  total_size += sizeof(TfLiteTensor) * model->input_count;
  total_size += sizeof(TfLiteTensor) * model->output_count;

  return total_size;
}

// Allocates the interpreter slab and populates all internal pointers to the
// appropriate offsets.
static iree_status_t _TfLiteInterpreterAllocate(
    const TfLiteModel* model, TfLiteInterpreter** out_interpreter) {
  iree_host_size_t interpreter_size = _TfLiteInterpreterCalculateSize(model);
  TfLiteInterpreter* interpreter = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(model->allocator, interpreter_size,
                                             (void**)&interpreter));
  memset(interpreter, 0, interpreter_size);
  interpreter->allocator = model->allocator;
  _TfLiteInterpreterOptionsSetDefaults(&interpreter->options);
  *out_interpreter = interpreter;

  interpreter->model = (TfLiteModel*)model;
  _TfLiteModelRetain(interpreter->model);

  uint8_t* p = (uint8_t*)interpreter +
               iree_host_align(sizeof(*interpreter), iree_max_align_t);

  iree_vm_type_def_t buffer_view_type_def =
      iree_vm_make_ref_type_def(iree_hal_buffer_type());

  iree_byte_span_t input_list_storage = iree_make_byte_span(
      p, iree_vm_list_storage_size(&buffer_view_type_def, model->input_count));
  IREE_RETURN_IF_ERROR(
      iree_vm_list_initialize(input_list_storage, &buffer_view_type_def,
                              model->input_count, &interpreter->input_list));
  p += input_list_storage.data_length;

  iree_byte_span_t output_list_storage = iree_make_byte_span(
      p, iree_vm_list_storage_size(&buffer_view_type_def, model->output_count));
  IREE_RETURN_IF_ERROR(
      iree_vm_list_initialize(output_list_storage, &buffer_view_type_def,
                              model->output_count, &interpreter->output_list));
  p += output_list_storage.data_length;

  interpreter->input_tensors = (TfLiteTensor*)p;
  p += sizeof(TfLiteTensor) * model->input_count;
  interpreter->output_tensors = (TfLiteTensor*)p;
  // p += sizeof(TfLiteTensor) * model->output_count;

  return iree_ok_status();
}

// Populates the input and output tensor lists with static metadata from the
// model and prepares for allocation/invocation.
static iree_status_t _TfLiteInterpreterPopulateIO(
    TfLiteInterpreter* interpreter) {
  iree_vm_function_t main_fn = interpreter->model->exports._main;
  iree_string_view_t io_names_attr = iree_vm_function_lookup_attr_by_name(
      &main_fn, iree_make_cstring_view("tfl.io.names"));
  iree_string_view_t io_types_attr = iree_vm_function_lookup_attr_by_name(
      &main_fn, iree_make_cstring_view("tfl.io.types"));
  iree_string_view_t io_quant_attr = iree_vm_function_lookup_attr_by_name(
      &main_fn, iree_make_cstring_view("tfl.io.quant"));

  // Setup static tensor metadata.
  for (iree_host_size_t i = 0; i < interpreter->model->input_count; ++i) {
    TfLiteTensor* tensor = &interpreter->input_tensors[i];
    memset(tensor, 0, sizeof(*tensor));
    iree_string_view_t io_name_part = iree_string_view_empty();
    iree_string_view_split(io_names_attr, ';', &io_name_part, &io_names_attr);
    iree_string_view_t io_type_part = iree_string_view_empty();
    iree_string_view_split(io_types_attr, ';', &io_type_part, &io_types_attr);
    iree_string_view_t io_quant_part = iree_string_view_empty();
    iree_string_view_split(io_quant_attr, ';', &io_quant_part, &io_quant_attr);
    IREE_RETURN_IF_ERROR(_TfLiteTensorParseNameAttr(tensor, io_name_part,
                                                    interpreter->allocator));
    IREE_RETURN_IF_ERROR(_TfLiteTensorParseTypeAttr(tensor, io_type_part));
    IREE_RETURN_IF_ERROR(_TfLiteTensorParseQuantAttr(tensor, io_quant_part));
  }
  for (iree_host_size_t i = 0; i < interpreter->model->output_count; ++i) {
    TfLiteTensor* tensor = &interpreter->output_tensors[i];
    memset(tensor, 0, sizeof(*tensor));
    iree_string_view_t io_name_part = iree_string_view_empty();
    iree_string_view_split(io_names_attr, ';', &io_name_part, &io_names_attr);
    iree_string_view_t io_type_part = iree_string_view_empty();
    iree_string_view_split(io_types_attr, ';', &io_type_part, &io_types_attr);
    iree_string_view_t io_quant_part = iree_string_view_empty();
    iree_string_view_split(io_quant_attr, ';', &io_quant_part, &io_quant_attr);
    IREE_RETURN_IF_ERROR(_TfLiteTensorParseNameAttr(tensor, io_name_part,
                                                    interpreter->allocator));
    IREE_RETURN_IF_ERROR(_TfLiteTensorParseTypeAttr(tensor, io_type_part));
    IREE_RETURN_IF_ERROR(_TfLiteTensorParseQuantAttr(tensor, io_quant_part));
  }

  // Prepare the IO lists we use when calling into the model.
  // The actual contents of these cannot be set until
  // TfLiteInterpreterAllocateTensors has been called.
  IREE_RETURN_IF_ERROR(iree_vm_list_reserve(interpreter->input_list,
                                            interpreter->model->input_count));
  IREE_RETURN_IF_ERROR(iree_vm_list_reserve(interpreter->output_list,
                                            interpreter->model->output_count));

  return iree_ok_status();
}

static iree_status_t _TfLiteInterpreterCreate(
    const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options,
    TfLiteInterpreter** out_interpreter) {
  *out_interpreter = NULL;

  // We allocate a large majority of the interpreter structures as a single
  // slab. There's still some allocations that we could prevent (like internal
  // VM stuff) but this at least covers half of it.
  IREE_RETURN_IF_ERROR(_TfLiteInterpreterAllocate(model, out_interpreter));
  TfLiteInterpreter* interpreter = *out_interpreter;

  if (optional_options) {
    memcpy(&interpreter->options, optional_options,
           sizeof(interpreter->options));
  }

  interpreter->instance = model->instance;
  iree_vm_instance_retain(interpreter->instance);
  interpreter->user_module = model->module;
  iree_vm_module_retain(interpreter->user_module);

  // External contexts could possibly used to emulate sharing this, but really
  // if a user is running with multiple models the tflite API is insufficient.
  IREE_RETURN_IF_ERROR(_TfLiteInterpreterPrepareHAL(interpreter));

  // Context will contain both the user-provided bytecode and the HAL module.
  // If we were to support custom ops we would also have a
  // tflite_resolver_module that we would register to resolve tflite ops into
  // IREE functions that will call custom ops through TfLiteRegistrations.
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      interpreter->instance, IREE_VM_CONTEXT_FLAG_NONE,
      IREE_ARRAYSIZE(interpreter->all_modules), interpreter->all_modules,
      interpreter->allocator, &interpreter->context));

  // Setup all I/O tensors and buffer views.
  IREE_RETURN_IF_ERROR(_TfLiteInterpreterPopulateIO(interpreter));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Core API
//===----------------------------------------------------------------------===//

TFL_CAPI_EXPORT extern TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options) {
  IREE_TRACE_ZONE_BEGIN(z0);
  TfLiteInterpreter* interpreter = NULL;
  iree_status_t status =
      _TfLiteInterpreterCreate(model, optional_options, &interpreter);
  if (iree_status_is_ok(iree_status_consume_code(status))) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "num_threads=", strlen("num_threads="));
    IREE_TRACE_ZONE_APPEND_VALUE(z0, interpreter->options.num_threads);
  } else {
    IREE_TRACE_MESSAGE(ERROR, "failed interpreter creation");
    TfLiteInterpreterDelete(interpreter);
    interpreter = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return interpreter;
}

TFL_CAPI_EXPORT extern TfLiteInterpreter*
TfLiteInterpreterCreateWithSelectedOps(
    const TfLiteModel* model, const TfLiteInterpreterOptions* options) {
  // No different from TfLiteInterpreterCreate: we don't have "ops" :)
  return TfLiteInterpreterCreate(model, options);
}

TFL_CAPI_EXPORT extern void TfLiteInterpreterDelete(
    TfLiteInterpreter* interpreter) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < interpreter->model->input_count; ++i) {
    _TfLiteTensorReset(&interpreter->input_tensors[i], interpreter->allocator);
  }
  for (iree_host_size_t i = 0; i < interpreter->model->output_count; ++i) {
    _TfLiteTensorReset(&interpreter->output_tensors[i], interpreter->allocator);
  }
  iree_vm_list_deinitialize(interpreter->input_list);
  iree_vm_list_deinitialize(interpreter->output_list);

  iree_vm_context_release(interpreter->context);
  iree_vm_module_release(interpreter->hal_module);
  iree_vm_module_release(interpreter->user_module);
  iree_hal_driver_release(interpreter->driver);
  iree_vm_instance_release(interpreter->instance);

  _TfLiteModelRelease(interpreter->model);
  iree_allocator_free(interpreter->allocator, interpreter);

  IREE_TRACE_ZONE_END(z0);
}

TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The compiler emits a special function we can use to reset just variables.
  // NOTE: the function is optional if the model had no variables.
  iree_status_t status = iree_ok_status();
  iree_vm_function_t reset_variables_fn =
      interpreter->model->exports._reset_variables;
  if (!iree_vm_function_is_null(reset_variables_fn)) {
    status = iree_vm_invoke(interpreter->context, reset_variables_fn,
                            IREE_VM_INVOCATION_FLAG_NONE,
                            /*policy=*/NULL, /*inputs=*/NULL, /*outputs=*/NULL,
                            interpreter->allocator);
  }

  IREE_TRACE_ZONE_END(z0);
  return _TfLiteStatusFromIREEStatus(status);
}

TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter) {
  return interpreter->model->input_count;
}

TFL_CAPI_EXPORT extern TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index) {
  if (input_index < 0 || input_index >= interpreter->model->input_count) {
    return NULL;
  }
  return &interpreter->input_tensors[input_index];
}

static iree_status_t _TfLiteInterpreterResizeInputTensor(
    TfLiteInterpreter* interpreter, int32_t input_index, const int* input_dims,
    int32_t input_dims_size) {
  if (input_index < 0 || input_index >= interpreter->model->input_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input_index out of range (0 <= %d < %d)",
                            input_index, interpreter->model->input_count);
  }
  if (iree_vm_function_is_null(
          interpreter->model->exports._resize_input_shape)) {
    // TODO(#3975): check if this is a no-op success in tflite.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model has no dynamic shapes");
  }

  _TfLiteInterpreterShapeFrame frame;
  IREE_RETURN_IF_ERROR(_TfLiteInterpreterShapeFrameInitialize(&frame));

  // Poke the model and let it update its internal shape.
  // TODO(#3975): return bool to allow model to say it failed.
  TfLiteTensor* tensor = &interpreter->input_tensors[input_index];
  iree_status_t status = _TfLiteInterpreterShapeFrameWriteValue(
      &frame, tensor->shape_rank, tensor->shape_dims);
  if (iree_status_is_ok(status)) {
    status = _TfLiteInterpreterShapeFrameApply(
        &frame, interpreter, interpreter->model->exports._resize_input_shape,
        input_index);
  }

  // NOTE: the allocation may now not match the requested shape. This is just
  // how the tflite API works unfortunately; until
  // TfLiteInterpreterAllocateTensors it will remain in an indeterminate state.

  _TfLiteInterpreterShapeFrameDeinitialize(&frame);
  return status;
}

TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterResizeInputTensor(
    TfLiteInterpreter* interpreter, int32_t input_index, const int* input_dims,
    int32_t input_dims_size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _TfLiteInterpreterResizeInputTensor(
      interpreter, input_index, input_dims, input_dims_size);
  IREE_TRACE_ZONE_END(z0);
  return _TfLiteStatusFromIREEStatus(status);
}

static iree_status_t _TfLiteInterpreterAllocateTensors(
    TfLiteInterpreter* interpreter) {
  // NOTE: we could slab allocate like tflite does, but then if any single
  // tensor has any single dimension that is resized the whole thing gets
  // reallocated upon resize. That's no good. Instead, we realloc each tensor
  // if their size has changed.

  // Refresh all shapes from the model. It should have all of the
  // non-data-dependent output shapes.
  IREE_RETURN_IF_ERROR(_TfLiteInterpreterRefreshIOShapes(interpreter));

  // Drop all input tensors we hang on to in the input list. This way we aren't
  // double-allocating during the resize.
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(interpreter->input_list, 0));

  // Reallocate input tensors (if needed).
  for (iree_host_size_t i = 0; i < interpreter->model->input_count; ++i) {
    TfLiteTensor* tensor = &interpreter->input_tensors[i];
    IREE_RETURN_IF_ERROR(_TfLiteTensorReallocateIfNeeded(
        tensor, iree_hal_device_allocator(interpreter->device),
        interpreter->allocator));
    iree_vm_ref_t buffer_ref = iree_hal_buffer_retain_ref(tensor->buffer);
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_ref_move(interpreter->input_list, &buffer_ref));
  }

  // TODO(benvanik): preallocate outputs when we support using them.
  // We could stash the buffer views in interpreter->output_list.
  // For now we just drop them all.
  for (iree_host_size_t i = 0; i < interpreter->model->output_count; ++i) {
    _TfLiteTensorDiscardBuffer(&interpreter->output_tensors[i]);
  }

  return iree_ok_status();
}

TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterAllocateTensors(
    TfLiteInterpreter* interpreter) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = _TfLiteInterpreterAllocateTensors(interpreter);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_device_size_t total_input_size = 0;
  for (iree_host_size_t i = 0; i < interpreter->model->input_count; ++i) {
    total_input_size +=
        iree_hal_buffer_byte_length(interpreter->input_tensors[i].buffer);
  }
  IREE_TRACE_ZONE_APPEND_VALUE(z0, total_input_size);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  IREE_TRACE_ZONE_END(z0);
  return _TfLiteStatusFromIREEStatus(status);
}

static iree_status_t _TfLiteInterpreterInvoke(TfLiteInterpreter* interpreter) {
  // tflite models only have a single entry point and the IREE converter
  // emits it as '_main'.
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(interpreter->context, interpreter->model->exports._main,
                     IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/NULL, interpreter->input_list,
                     interpreter->output_list, interpreter->allocator));

  // Refresh output shapes.
  // TODO(#3975): just use buffer view results or at least just refresh outputs.
  IREE_RETURN_IF_ERROR(_TfLiteInterpreterRefreshIOShapes(interpreter));

  // Map the output buffers.
  // NOTE: we could defer the mapping unless requested and ensure state buffers
  // remain where they currently are for the next invocation.
  for (iree_host_size_t i = 0; i < interpreter->model->output_count; ++i) {
    iree_hal_buffer_t* buffer =
        iree_vm_list_get_buffer_assign(interpreter->output_list, i);
    TfLiteTensor* tensor = &interpreter->output_tensors[i];
    IREE_RETURN_IF_ERROR(_TfLiteTensorBind(tensor, buffer));
  }

  return iree_ok_status();
}

TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterInvoke(
    TfLiteInterpreter* interpreter) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = _TfLiteInterpreterInvoke(interpreter);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_device_size_t total_output_size = 0;
  for (iree_host_size_t i = 0; i < interpreter->model->output_count; ++i) {
    total_output_size +=
        iree_hal_buffer_byte_length(interpreter->output_tensors[i].buffer);
  }
  IREE_TRACE_ZONE_APPEND_VALUE(z0, total_output_size);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  IREE_TRACE_ZONE_END(z0);
  return _TfLiteStatusFromIREEStatus(status);
}

TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter) {
  return interpreter->model->output_count;
}

TFL_CAPI_EXPORT extern const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index) {
  if (output_index < 0 || output_index >= interpreter->model->output_count) {
    return NULL;
  }
  return &interpreter->output_tensors[output_index];
}
