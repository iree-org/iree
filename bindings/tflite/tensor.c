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

#include "bindings/tflite/tensor.h"

#include "bindings/tflite/shim.h"
#include "iree/base/tracing.h"

iree_status_t _TfLiteTensorParseNameAttr(TfLiteTensor* tensor,
                                         iree_string_view_t attr,
                                         iree_allocator_t allocator) {
  char* str = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, attr.size + 1, (void**)&str));
  memcpy(str, attr.data, attr.size);
  str[attr.size] = 0;
  tensor->name = iree_make_string_view(str, attr.size);
  return iree_ok_status();
}

iree_status_t _TfLiteTensorParseTypeAttr(TfLiteTensor* tensor,
                                         iree_string_view_t attr) {
  // TODO(#3978): extract tensor type and plumb through iree.reflection.
  tensor->type = kTfLiteFloat32;
  return iree_ok_status();
}

iree_status_t _TfLiteTensorParseQuantAttr(TfLiteTensor* tensor,
                                          iree_string_view_t attr) {
  // TODO(#3972): extract !quant.uniform and plumb through iree.reflection.
  tensor->quantization_params.scale = 0.0f;
  tensor->quantization_params.zero_point = 0;
  return iree_ok_status();
}

// Who the hell uses sizeof(bool) - an **implementation-defined value** -
// as a wire format? https://stackoverflow.com/a/4897859
static_assert(sizeof(bool) == 1, "bool must be 1 byte to match tf/tflite");

// Converts a tflite type to the HAL storage type.
// If the is a composite of multiple primitive types (such as a complex number)
// then |out_storage_scalar| is set to >1.
static iree_status_t _TfLiteTypeToElementType(
    TfLiteType type, iree_hal_element_type_t* out_element_type,
    iree_host_size_t* out_storage_scalar) {
  *out_element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  *out_storage_scalar = 1;
  switch (type) {
    default:
    case kTfLiteNoType:
      // Hopefully only used as a sentinel.
      *out_element_type = IREE_HAL_ELEMENT_TYPE_NONE;
      break;
    case kTfLiteInt8:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_8;
      break;
    case kTfLiteUInt8:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_UINT_8;
      break;
    case kTfLiteInt16:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_16;
      break;
    case kTfLiteInt32:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_32;
      break;
    case kTfLiteInt64:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_SINT_64;
      break;
    case kTfLiteUInt64:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_UINT_64;
      break;
    case kTfLiteFloat16:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
      break;
    case kTfLiteFloat32:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      break;
    case kTfLiteFloat64:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      break;
    case kTfLiteBool:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_UINT_8;
      break;
    case kTfLiteComplex64:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      *out_storage_scalar = 2;  // real + imag
      break;
    case kTfLiteComplex128:
      *out_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      *out_storage_scalar = 2;  // real + imag
      break;
    case kTfLiteString:
      // This isn't a tensor, it's an std::vector<std::string>. Don't use this
      // type and instead use the IREE C API which has such amazing modern
      // programming concepts like ... lists.
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "kTfLiteString is not implemented (and won't "
                              "be); use the IREE C API");
  }
  return iree_ok_status();
}

iree_status_t _TfLiteTensorReallocateIfNeeded(
    TfLiteTensor* tensor, iree_hal_allocator_t* buffer_allocator,
    iree_allocator_t heap_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Format conversion; ensure we can support the type.
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_host_size_t storage_scalar = 1;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      _TfLiteTypeToElementType(tensor->type, &element_type, &storage_scalar));

  // Compute the total allocation size required, possibly with padding.
  iree_hal_dim_t shape_dims[IREE_BINDINGS_TFLITE_MAX_RANK];
  for (int32_t i = 0; i < tensor->shape_rank; ++i) {
    shape_dims[i] = (iree_hal_dim_t)tensor->shape_dims[i];
  }
  iree_device_size_t allocation_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_buffer_compute_view_size(shape_dims, tensor->shape_rank,
                                            element_type, &allocation_size));
  allocation_size *= storage_scalar;

  // If the old buffer is the same size then no need to realloc.
  if (tensor->buffer &&
      iree_hal_buffer_byte_length(tensor->buffer) == allocation_size) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Allocate the underlying buffer for the tensor.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_allocator_allocate_buffer(
          buffer_allocator,
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
          IREE_HAL_BUFFER_USAGE_ALL, allocation_size, &tensor->buffer));

  // Map the buffer memory immediately. The tflite API doesn't let us know if
  // this is a buffer the user will actually touch or some state buffer that is
  // just going to be passed to future invocations. We could move this to an
  // on-demand mapping when the user calls TfLiteTensorData but this at least
  // puts potential errors in the same easy to find place.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_buffer_map_range(tensor->buffer, IREE_HAL_MEMORY_ACCESS_ALL, 0,
                                IREE_WHOLE_BUFFER, &tensor->buffer_mapping));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t _TfLiteTensorBind(TfLiteTensor* tensor,
                                iree_hal_buffer_t* buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  _TfLiteTensorDiscardBuffer(tensor);
  if (!buffer) {
    // Just a discard (invalid output/etc).
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Attempt to map the buffer. The tflite API doesn't let us know if this
  // should be read or read/write - or if we even need to map at all. We could
  // move this to an on-demand mapping when the user calls TfLiteTensorData but
  // this at least puts potential errors in the same easy to find place.
  iree_device_size_t byte_offset = 0;
  iree_device_size_t byte_length = IREE_WHOLE_BUFFER;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_buffer_map_range(
          buffer, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
          byte_offset, byte_length, &tensor->buffer_mapping));

  // Retain the buffer view until discarded/reset.
  tensor->buffer = buffer;
  iree_hal_buffer_retain(tensor->buffer);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void _TfLiteTensorDiscardBuffer(TfLiteTensor* tensor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (tensor->buffer_mapping.contents.data != NULL) {
    iree_hal_buffer_unmap_range(&tensor->buffer_mapping);
  }
  iree_hal_buffer_release(tensor->buffer);
  tensor->buffer = NULL;
  IREE_TRACE_ZONE_END(z0);
}

void _TfLiteTensorReset(TfLiteTensor* tensor, iree_allocator_t allocator) {
  _TfLiteTensorDiscardBuffer(tensor);
  if (tensor->name.data) {
    iree_allocator_free(allocator, (void*)tensor->name.data);
  }
}

TFL_CAPI_EXPORT extern TfLiteType TfLiteTensorType(const TfLiteTensor* tensor) {
  return tensor->type;
}

TFL_CAPI_EXPORT extern int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor) {
  return tensor->shape_rank;
}

TFL_CAPI_EXPORT extern int32_t TfLiteTensorDim(const TfLiteTensor* tensor,
                                               int32_t dim_index) {
  return tensor->shape_dims[dim_index];
}

TFL_CAPI_EXPORT extern size_t TfLiteTensorByteSize(const TfLiteTensor* tensor) {
  return (size_t)iree_hal_buffer_byte_length(tensor->buffer);
}

TFL_CAPI_EXPORT extern void* TfLiteTensorData(const TfLiteTensor* tensor) {
  return tensor->buffer_mapping.contents.data;
}

TFL_CAPI_EXPORT extern const char* TfLiteTensorName(
    const TfLiteTensor* tensor) {
  return tensor->name.data;
}

TFL_CAPI_EXPORT extern TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    const TfLiteTensor* tensor) {
  return tensor->quantization_params;
}

TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyFromBuffer(
    TfLiteTensor* tensor, const void* input_data, size_t input_data_size) {
  if (input_data_size != tensor->buffer_mapping.contents.data_length) {
    return kTfLiteApplicationError;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, tensor->buffer_mapping.contents.data_length);

  // NOTE: we could use a iree_hal_buffer_write_data here but we already have
  // the buffer mapped. If we knew the user would never use TfLiteTensorData and
  // could avoid mapping the buffer it would be more efficient and portable to
  // do the iree_hal_buffer_copy_data.
  memcpy(tensor->buffer_mapping.contents.data, input_data, input_data_size);

  IREE_TRACE_ZONE_END(z0);
  return kTfLiteOk;
}

TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyToBuffer(
    const TfLiteTensor* output_tensor, void* output_data,
    size_t output_data_size) {
  if (output_data_size != output_tensor->buffer_mapping.contents.data_length) {
    return kTfLiteApplicationError;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(
      z0, output_tensor->buffer_mapping.contents.data_length);

  // NOTE: as with above we should use an iree_hal_buffer_read_data here.
  memcpy(output_data, output_tensor->buffer_mapping.contents.data,
         output_data_size);

  IREE_TRACE_ZONE_END(z0);
  return kTfLiteOk;
}
