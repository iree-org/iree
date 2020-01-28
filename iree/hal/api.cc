// Copyright 2019 Google LLC
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

#include "iree/hal/api.h"

#include <cctype>
#include <cstdio>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/shape.h"
#include "iree/base/tracing.h"
#include "iree/hal/api_detail.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/device.h"
#include "iree/hal/driver.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/fence.h"
#include "iree/hal/heap_buffer.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {

// Defines the iree_hal_<type_name>_retain/_release methods.
#define IREE_HAL_API_RETAIN_RELEASE(type_name, cc_type)         \
  IREE_API_EXPORT iree_status_t iree_hal_##type_name##_retain(  \
      iree_hal_##type_name##_t* type_name) {                    \
    auto* handle = reinterpret_cast<cc_type*>(type_name);       \
    if (!handle) return IREE_STATUS_INVALID_ARGUMENT;           \
    handle->AddReference();                                     \
    return IREE_STATUS_OK;                                      \
  }                                                             \
  IREE_API_EXPORT iree_status_t iree_hal_##type_name##_release( \
      iree_hal_##type_name##_t* type_name) {                    \
    auto* handle = reinterpret_cast<cc_type*>(type_name);       \
    if (!handle) return IREE_STATUS_INVALID_ARGUMENT;           \
    handle->ReleaseReference();                                 \
    return IREE_STATUS_OK;                                      \
  }

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_element_type(
    iree_string_view_t value, iree_hal_element_type_t* out_element_type) {
  if (!out_element_type) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_element_type = IREE_HAL_ELEMENT_TYPE_NONE;

  auto str_value = absl::string_view(value.data, value.size);

  iree_hal_numerical_type_t numerical_type = IREE_HAL_NUMERICAL_TYPE_UNKNOWN;
  if (absl::StartsWith(str_value, "i")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED;
    str_value.remove_prefix(1);
  } else if (absl::StartsWith(str_value, "u")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED;
    str_value.remove_prefix(1);
  } else if (absl::StartsWith(str_value, "f")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE;
    str_value.remove_prefix(1);
  } else if (absl::StartsWith(str_value, "x")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_UNKNOWN;
    str_value.remove_prefix(1);
  } else {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  int32_t bit_count = 0;
  if (!absl::SimpleAtoi(str_value, &bit_count)) {
    return IREE_STATUS_OUT_OF_RANGE;
  }

  *out_element_type = iree_hal_make_element_type(numerical_type, bit_count);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_format_element_type(
    iree_hal_element_type_t element_type, size_t capacity, char* buffer,
    size_t* out_length) {
  if (out_length) {
    *out_length = 0;
  }
  const char* prefix;
  switch (iree_hal_element_numerical_type(element_type)) {
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED:
      prefix = "i";
      break;
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED:
      prefix = "u";
      break;
    case IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE:
      prefix = "f";
      break;
    default:
      prefix = "x";
      break;
  }
  int n = std::snprintf(
      buffer, capacity, "%s%d", prefix,
      static_cast<int32_t>(iree_hal_element_bit_count(element_type)));
  if (n < 0) {
    return IREE_STATUS_FAILED_PRECONDITION;
  }
  if (out_length) {
    *out_length = n;
  }
  return n >= capacity ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(allocator, Allocator);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_size(
    const iree_hal_allocator_t* allocator, const int32_t* shape,
    size_t shape_rank, iree_hal_element_type_t element_type,
    iree_device_size_t* out_allocation_size) {
  if (!out_allocation_size) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_allocation_size = 0;
  if (!allocator) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): layout/padding.
  iree_device_size_t byte_length = iree_hal_element_byte_count(element_type);
  for (int i = 0; i < shape_rank; ++i) {
    byte_length *= shape[i];
  }
  *out_allocation_size = byte_length;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_offset(
    const iree_hal_allocator_t* allocator, const int32_t* shape,
    size_t shape_rank, iree_hal_element_type_t element_type,
    const int32_t* indices, size_t indices_count,
    iree_device_size_t* out_offset) {
  if (!out_offset) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_offset = 0;
  if (!allocator) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  if (shape_rank != indices_count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): layout/padding.
  iree_device_size_t offset = 0;
  for (int i = 0; i < indices_count; ++i) {
    if (indices[i] >= shape[i]) {
      return IREE_STATUS_OUT_OF_RANGE;
    }
    iree_device_size_t axis_offset = indices[i];
    for (int j = i + 1; j < shape_rank; ++j) {
      axis_offset *= shape[j];
    }
    offset += axis_offset;
  }
  offset *= iree_hal_element_byte_count(element_type);

  *out_offset = offset;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_range(
    const iree_hal_allocator_t* allocator, const int32_t* shape,
    size_t shape_rank, iree_hal_element_type_t element_type,
    const int32_t* start_indices, size_t indices_count, const int32_t* lengths,
    size_t lengths_count, iree_device_size_t* out_start_offset,
    iree_device_size_t* out_length) {
  if (!out_start_offset || !out_length) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_start_offset = 0;
  *out_length = 0;
  if (!allocator) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  if (indices_count != lengths_count || indices_count != shape_rank) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): layout/padding.
  absl::InlinedVector<int32_t, 6> end_indices(shape_rank);
  iree_device_size_t element_size = iree_hal_element_byte_count(element_type);
  iree_device_size_t subspan_length = element_size;
  for (int i = 0; i < lengths_count; ++i) {
    subspan_length *= lengths[i];
    end_indices[i] = start_indices[i] + lengths[i] - 1;
  }

  iree_device_size_t start_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_offset(
      allocator, shape, shape_rank, element_type, start_indices, indices_count,
      &start_byte_offset));
  iree_device_size_t end_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_offset(
      allocator, shape, shape_rank, element_type, end_indices.data(),
      end_indices.size(), &end_byte_offset));

  // Non-contiguous regions not yet implemented. Will be easier to detect when
  // we have strides.
  auto offset_length = end_byte_offset - start_byte_offset + element_size;
  if (subspan_length != offset_length) {
    return IREE_STATUS_UNIMPLEMENTED;
  }

  *out_start_offset = start_byte_offset;
  *out_length = subspan_length;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t buffer_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_allocator_allocate_buffer");
  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer = nullptr;
  auto* handle = reinterpret_cast<Allocator*>(allocator);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto buffer,
      handle->Allocate(static_cast<MemoryTypeBitfield>(memory_type),
                       static_cast<BufferUsageBitfield>(buffer_usage),
                       allocation_size));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(buffer.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_wrap_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t buffer_usage, iree_byte_span_t data,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_allocator_wrap_buffer");
  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer = nullptr;
  auto* handle = reinterpret_cast<Allocator*>(allocator);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto buffer,
      handle->WrapMutable(static_cast<MemoryTypeBitfield>(memory_type),
                          static_cast<MemoryAccessBitfield>(allowed_access),
                          static_cast<BufferUsageBitfield>(buffer_usage),
                          data.data, data.data_length));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(buffer.release());
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(buffer, Buffer);

IREE_API_EXPORT iree_status_t iree_hal_buffer_subspan(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_subspan");

  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer = nullptr;

  if (!buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  auto handle = add_ref(reinterpret_cast<Buffer*>(buffer));

  IREE_API_ASSIGN_OR_RETURN(auto new_handle,
                            Buffer::Subspan(handle, byte_offset, byte_length));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(new_handle.release());

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_buffer_allocator(const iree_hal_buffer_t* buffer) {
  const auto* handle = reinterpret_cast<const Buffer*>(buffer);
  CHECK(handle) << "NULL buffer handle";
  return reinterpret_cast<iree_hal_allocator_t*>(handle->allocator());
}

IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer) {
  const auto* handle = reinterpret_cast<const Buffer*>(buffer);
  CHECK(handle) << "NULL buffer handle";
  return handle->byte_length();
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_zero(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_zero");
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  if (!buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_API_RETURN_IF_ERROR(handle->Fill8(byte_offset, byte_length, 0));
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_fill(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length, const void* pattern,
                     iree_host_size_t pattern_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_fill");
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  if (!buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_API_RETURN_IF_ERROR(
      handle->Fill(byte_offset, byte_length, pattern, pattern_length));
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_read_data(
    iree_hal_buffer_t* buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_read_data");
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  if (!buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_API_RETURN_IF_ERROR(
      handle->ReadData(source_offset, target_buffer, data_length));
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_write_data(
    iree_hal_buffer_t* buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_write_data");
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  if (!buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  IREE_API_RETURN_IF_ERROR(
      handle->WriteData(target_offset, source_buffer, data_length));
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_map(
    iree_hal_buffer_t* buffer, iree_hal_memory_access_t memory_access,
    iree_device_size_t element_offset, iree_device_size_t element_length,
    iree_hal_mapped_memory_t* out_mapped_memory) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_map");

  if (!out_mapped_memory) {
    LOG(ERROR) << "output mapped memory not set";
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_mapped_memory, 0, sizeof(*out_mapped_memory));

  if (!buffer) {
    LOG(ERROR) << "buffer not set";
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto* buffer_handle = reinterpret_cast<Buffer*>(buffer);
  IREE_API_ASSIGN_OR_RETURN(
      auto mapping, buffer_handle->MapMemory<uint8_t>(
                        static_cast<MemoryAccessBitfield>(memory_access),
                        element_offset, element_length));

  static_assert(sizeof(iree_hal_mapped_memory_t::reserved) >=
                    sizeof(MappedMemory<uint8_t>),
                "C mapped memory struct must have large enough storage for the "
                "matching C++ struct");
  auto* mapping_storage =
      reinterpret_cast<MappedMemory<uint8_t>*>(out_mapped_memory->reserved);
  *mapping_storage = std::move(mapping);

  out_mapped_memory->contents = {mapping_storage->unsafe_data(),
                                 mapping_storage->size()};

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_unmap(
    iree_hal_buffer_t* buffer, iree_hal_mapped_memory_t* mapped_memory) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_map");
  auto* buffer_handle = reinterpret_cast<Buffer*>(buffer);
  if (!buffer_handle || !mapped_memory) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto* mapping =
      reinterpret_cast<MappedMemory<uint8_t>*>(mapped_memory->reserved);
  mapping->reset();

  std::memset(mapped_memory, 0, sizeof(*mapped_memory));
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::HeapBuffer
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_heap_buffer_allocate(
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t usage,
    iree_host_size_t allocation_size, iree_allocator_t contents_allocator,
    iree_allocator_t allocator, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_heap_buffer_allocate");

  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer = nullptr;

  if (!allocation_size) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto handle = HeapBuffer::Allocate(
      static_cast<MemoryTypeBitfield>(memory_type),
      static_cast<BufferUsageBitfield>(usage), allocation_size);

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(
      static_cast<Buffer*>(handle.release()));

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_heap_buffer_allocate_copy(
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t usage,
    iree_hal_memory_access_t allowed_access, iree_byte_span_t contents,
    iree_allocator_t contents_allocator, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_heap_buffer_allocate_copy");

  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer = nullptr;

  if (!contents.data || !contents.data_length) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto handle = HeapBuffer::AllocateCopy(
      static_cast<BufferUsageBitfield>(usage),
      static_cast<MemoryAccessBitfield>(allowed_access), contents.data,
      contents.data_length);

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(handle.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_heap_buffer_wrap(
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t usage, iree_byte_span_t contents,
    iree_allocator_t allocator, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_heap_buffer_wrap");

  if (!out_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer = nullptr;

  if (!contents.data || !contents.data_length) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto handle =
      HeapBuffer::WrapMutable(static_cast<MemoryTypeBitfield>(memory_type),
                              static_cast<MemoryAccessBitfield>(allowed_access),
                              static_cast<BufferUsageBitfield>(usage),
                              contents.data, contents.data_length);

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(handle.release());
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(buffer_view, iree_hal_buffer_view);

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, const int32_t* shape, size_t shape_rank,
    iree_hal_element_type_t element_type, iree_allocator_t allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_create");

  if (!out_buffer_view) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer_view = nullptr;

  if (!buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // Allocate and initialize the iree_hal_buffer_view struct.
  // Note that we have the dynamically-sized shape dimensions on the end.
  iree_hal_buffer_view* buffer_view = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(*buffer_view) + sizeof(int32_t) * shape_rank,
      reinterpret_cast<void**>(&buffer_view)));
  new (buffer_view) iree_hal_buffer_view();
  buffer_view->allocator = allocator;
  buffer_view->buffer = buffer;
  iree_hal_buffer_retain(buffer_view->buffer);
  buffer_view->element_type = element_type;
  buffer_view->byte_length =
      iree_hal_element_byte_count(buffer_view->element_type);
  buffer_view->shape_rank = shape_rank;
  for (int i = 0; i < shape_rank; ++i) {
    buffer_view->shape[i] = shape[i];
    buffer_view->byte_length *= shape[i];
  }

  *out_buffer_view = buffer_view;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_subview(
    const iree_hal_buffer_view_t* buffer_view, const int32_t* start_indices,
    size_t indices_count, const int32_t* lengths, size_t lengths_count,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view) {
  if (!out_buffer_view) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // NOTE: we rely on the compute range call to do parameter validation.
  iree_device_size_t start_offset = 0;
  iree_device_size_t subview_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
      buffer_view, start_indices, indices_count, lengths, lengths_count,
      &start_offset, &subview_length));

  iree_hal_buffer_t* subview_buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(buffer_view->buffer,
                                               start_offset, subview_length,
                                               allocator, &subview_buffer));

  iree_status_t result = iree_hal_buffer_view_create(
      subview_buffer, lengths, lengths_count, buffer_view->element_type,
      allocator, out_buffer_view);
  iree_hal_buffer_release(subview_buffer);
  return result;
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_view_buffer(
    const iree_hal_buffer_view_t* buffer_view) {
  CHECK(buffer_view) << "NULL buffer_view handle";
  return buffer_view->buffer;
}

IREE_API_EXPORT size_t IREE_API_CALL
iree_hal_buffer_view_shape_rank(const iree_hal_buffer_view_t* buffer_view) {
  CHECK(buffer_view) << "NULL buffer_view handle";
  return buffer_view->shape_rank;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, size_t rank_capacity,
    int32_t* out_shape, size_t* out_shape_rank) {
  if (out_shape_rank) {
    *out_shape_rank = 0;
  }
  if (!buffer_view || !out_shape) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  if (out_shape_rank) {
    *out_shape_rank = buffer_view->shape_rank;
  }
  if (rank_capacity < buffer_view->shape_rank) {
    return IREE_STATUS_OUT_OF_RANGE;
  }

  for (int i = 0; i < buffer_view->shape_rank; ++i) {
    out_shape[i] = buffer_view->shape[i];
  }

  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_hal_element_type_t IREE_API_CALL
iree_hal_buffer_view_element_type(const iree_hal_buffer_view_t* buffer_view) {
  CHECK(buffer_view) << "NULL buffer_view handle";
  return buffer_view->element_type;
}

IREE_API_EXPORT size_t
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view) {
  CHECK(buffer_view) << "NULL buffer_view handle";
  return iree_hal_element_byte_count(buffer_view->element_type);
}

IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_view_byte_length(const iree_hal_buffer_view_t* buffer_view) {
  CHECK(buffer_view) << "NULL buffer_view handle";
  return buffer_view->byte_length;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_offset(
    const iree_hal_buffer_view_t* buffer_view, const int32_t* indices,
    size_t indices_count, iree_device_size_t* out_offset) {
  if (!buffer_view) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  return iree_hal_allocator_compute_offset(
      iree_hal_buffer_allocator(buffer_view->buffer), buffer_view->shape,
      buffer_view->shape_rank, buffer_view->element_type, indices,
      indices_count, out_offset);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_range(
    const iree_hal_buffer_view_t* buffer_view, const int32_t* start_indices,
    size_t indices_count, const int32_t* lengths, size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length) {
  if (!buffer_view) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  return iree_hal_allocator_compute_range(
      iree_hal_buffer_allocator(buffer_view->buffer), buffer_view->shape,
      buffer_view->shape_rank, buffer_view->element_type, start_indices,
      indices_count, lengths, lengths_count, out_start_offset, out_length);
}

//===----------------------------------------------------------------------===//
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(command_buffer, CommandBuffer);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_allocator_t allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_create");
  if (!out_command_buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_command_buffer = nullptr;
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto command_buffer,
      handle->CreateCommandBuffer(
          static_cast<CommandBufferModeBitfield>(mode),
          static_cast<CommandCategoryBitfield>(command_categories)));

  *out_command_buffer =
      reinterpret_cast<iree_hal_command_buffer_t*>(command_buffer.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_begin(iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_begin");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  return ToApiStatus(handle->Begin());
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_end(iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_end");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  return ToApiStatus(handle->End());
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_execution_barrier");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  // TODO(benvanik): refactor the C++ types to use the C types for storage so
  // that we can safely map between the two. For now assume size equality
  // is layout equality (as compilers aren't allowed to reorder structs).
  static_assert(sizeof(MemoryBarrier) == sizeof(iree_hal_memory_barrier_t),
                "Expecting identical layout");
  static_assert(sizeof(BufferBarrier) == sizeof(iree_hal_buffer_barrier_t),
                "Expecting identical layout");
  return ToApiStatus(handle->ExecutionBarrier(
      static_cast<ExecutionStageBitfield>(source_stage_mask),
      static_cast<ExecutionStageBitfield>(target_stage_mask),
      absl::MakeConstSpan(
          reinterpret_cast<const MemoryBarrier*>(memory_barriers),
          memory_barrier_count),
      absl::MakeConstSpan(
          reinterpret_cast<const BufferBarrier*>(buffer_barriers),
          buffer_barrier_count)));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_fill_buffer");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  return ToApiStatus(
      handle->FillBuffer(reinterpret_cast<Buffer*>(target_buffer),
                         target_offset, length, pattern, pattern_length));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_update_buffer(iree_hal_command_buffer_t* command_buffer,
                                      const void* source_buffer,
                                      iree_host_size_t source_offset,
                                      iree_hal_buffer_t* target_buffer,
                                      iree_device_size_t target_offset,
                                      iree_device_size_t length) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_update_buffer");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  return ToApiStatus(handle->UpdateBuffer(
      source_buffer, source_offset, reinterpret_cast<Buffer*>(target_buffer),
      target_offset, length));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_copy_buffer");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  return ToApiStatus(handle->CopyBuffer(
      reinterpret_cast<Buffer*>(source_buffer), source_offset,
      reinterpret_cast<Buffer*>(target_buffer), target_offset, length));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, int32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_bind_descriptor_set");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  // TODO(benvanik): implement descriptor sets.
  return IREE_STATUS_UNIMPLEMENTED;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point, int32_t workgroup_x,
    int32_t workgroup_y, int32_t workgroup_z) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_dispatch");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  // TODO(benvanik): implement descriptor sets.
  return IREE_STATUS_UNIMPLEMENTED;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_dispatch_indirect");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  // TODO(benvanik): implement descriptor sets.
  return IREE_STATUS_UNIMPLEMENTED;
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(descriptor_set, DescriptorSet);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_descriptor_set_create(
    iree_hal_device_t* device, iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_allocator_t allocator,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  IREE_TRACE_SCOPE0("iree_hal_descriptor_set_create");
  if (!out_descriptor_set) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_descriptor_set = nullptr;
  if (!set_layout) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  if (binding_count && !bindings) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): refactor the C++ types to use the C types for storage so
  // that we can safely map between the two. For now assume size equality
  // is layout equality (as compilers aren't allowed to reorder structs).
  static_assert(sizeof(DescriptorSet::Binding) ==
                    sizeof(iree_hal_descriptor_set_binding_t),
                "Expecting identical layout");
  IREE_API_ASSIGN_OR_RETURN(
      auto descriptor_set,
      handle->CreateDescriptorSet(
          add_ref(reinterpret_cast<DescriptorSetLayout*>(set_layout)),
          absl::MakeConstSpan(
              reinterpret_cast<const DescriptorSet::Binding*>(bindings),
              binding_count)));

  *out_descriptor_set =
      reinterpret_cast<iree_hal_descriptor_set_t*>(descriptor_set.release());
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSetLayout
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(descriptor_set_layout, DescriptorSetLayout);

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_descriptor_set_layout_create(
    iree_hal_device_t* device, iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_TRACE_SCOPE0("iree_hal_descriptor_set_layout_create");
  if (!out_descriptor_set_layout) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_descriptor_set_layout = nullptr;
  if (binding_count && !bindings) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  // TODO(benvanik): refactor the C++ types to use the C types for storage so
  // that we can safely map between the two. For now assume size equality
  // is layout equality (as compilers aren't allowed to reorder structs).
  static_assert(sizeof(DescriptorSetLayout::Binding) ==
                    sizeof(iree_hal_descriptor_set_layout_binding_t),
                "Expecting identical layout");
  IREE_API_ASSIGN_OR_RETURN(
      auto descriptor_set_layout,
      handle->CreateDescriptorSetLayout(absl::MakeConstSpan(
          reinterpret_cast<const DescriptorSetLayout::Binding*>(bindings),
          binding_count)));

  *out_descriptor_set_layout =
      reinterpret_cast<iree_hal_descriptor_set_layout_t*>(
          descriptor_set_layout.release());
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(device, Device);

IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_device_allocator(iree_hal_device_t* device) {
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) return nullptr;
  return reinterpret_cast<iree_hal_allocator_t*>(handle->allocator());
}

//===----------------------------------------------------------------------===//
// iree::hal::Driver
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(driver, Driver);

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_query_available_devices(
    iree_hal_driver_t* driver, iree_allocator_t allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count) {
  IREE_TRACE_SCOPE0("iree_hal_driver_query_available_devices");
  if (!out_device_info_count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_device_info_count = 0;
  if (!out_device_infos) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  auto* handle = reinterpret_cast<Driver*>(driver);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(auto device_infos,
                            handle->EnumerateAvailableDevices());
  size_t total_string_size = 0;
  for (const auto& device_info : device_infos) {
    total_string_size += device_info.name().size();
  }

  *out_device_info_count = device_infos.size();
  iree_hal_device_info_t* device_info_storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator,
      device_infos.size() * sizeof(*device_info_storage) + total_string_size,
      (void**)&device_info_storage));

  char* p = reinterpret_cast<char*>(device_info_storage) +
            device_infos.size() * sizeof(*device_info_storage);
  for (int i = 0; i < device_infos.size(); ++i) {
    const auto& device_info = device_infos[i];
    device_info_storage[i].device_id = device_info.device_id();

    size_t name_size = device_info.name().size();
    std::memcpy(p, device_info.name().c_str(), name_size);
    device_info_storage[i].name = iree_string_view_t{p, name_size};
    p += name_size;
  }

  *out_device_infos = device_info_storage;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_driver_create_device(
    iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
    iree_allocator_t allocator, iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_driver_create_device");
  if (!out_device) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_device = nullptr;
  auto* handle = reinterpret_cast<Driver*>(driver);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(auto device, handle->CreateDevice(device_id));

  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_create_default_device(iree_hal_driver_t* driver,
                                      iree_allocator_t allocator,
                                      iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_driver_create_default_device");
  if (!out_device) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_device = nullptr;
  auto* handle = reinterpret_cast<Driver*>(driver);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(auto device, handle->CreateDefaultDevice());
  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::DriverRegistry
//===----------------------------------------------------------------------===//

IREE_API_EXPORT bool IREE_API_CALL
iree_hal_driver_registry_has_driver(iree_string_view_t driver_name) {
  return DriverRegistry::shared_registry()->HasDriver(
      absl::string_view{driver_name.data, driver_name.size});
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_registry_query_available_drivers(
    iree_allocator_t allocator, iree_string_view_t** out_driver_names,
    iree_host_size_t* out_driver_count) {
  IREE_TRACE_SCOPE0("iree_hal_driver_registry_query_available_drivers");
  if (!out_driver_count) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_driver_count = 0;
  if (!out_driver_names) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  auto* registry = DriverRegistry::shared_registry();
  auto available_drivers = registry->EnumerateAvailableDrivers();
  size_t total_string_size = 0;
  for (const auto& driver_name : available_drivers) {
    total_string_size += driver_name.size();
  }

  *out_driver_count = available_drivers.size();
  iree_string_view_t* driver_name_storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator,
      available_drivers.size() * sizeof(*driver_name_storage) +
          total_string_size,
      (void**)&driver_name_storage));

  char* p = reinterpret_cast<char*>(driver_name_storage) +
            available_drivers.size() * sizeof(*driver_name_storage);
  for (int i = 0; i < available_drivers.size(); ++i) {
    const auto& driver_name = available_drivers[i];
    size_t name_size = driver_name.size();
    std::memcpy(p, driver_name.c_str(), name_size);
    driver_name_storage[i] = iree_string_view_t{p, name_size};
    p += name_size;
  }

  *out_driver_names = driver_name_storage;
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_registry_create_driver(iree_string_view_t driver_name,
                                       iree_allocator_t allocator,
                                       iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE0("iree_hal_driver_registry_create_driver");
  if (!out_driver) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_driver = nullptr;

  auto* registry = DriverRegistry::shared_registry();
  IREE_API_ASSIGN_OR_RETURN(
      auto driver,
      registry->Create(absl::string_view(driver_name.data, driver_name.size)));

  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::Executable
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(executable, Executable);

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableLayout
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(executable_layout, ExecutableLayout);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_executable_layout_create(
    iree_hal_device_t* device, iree_host_size_t set_layout_count,
    const iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constants, iree_allocator_t allocator,
    iree_hal_executable_layout_t** out_executable_layout) {
  IREE_TRACE_SCOPE0("iree_hal_executable_layout_create");
  if (!out_executable_layout) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_executable_layout = nullptr;
  if (set_layout_count && !set_layouts) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  IREE_API_ASSIGN_OR_RETURN(
      auto executable_layout,
      handle->CreateExecutableLayout(
          absl::MakeConstSpan(
              reinterpret_cast<const ref_ptr<DescriptorSetLayout>*>(
                  set_layouts),
              set_layout_count),
          push_constants));

  *out_executable_layout = reinterpret_cast<iree_hal_executable_layout_t*>(
      executable_layout.release());
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::Fence
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(fence, Fence);

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(semaphore, Semaphore);

}  // namespace hal
}  // namespace iree
