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

#include "hal/api.h"

#include "base/api.h"
#include "base/api_util.h"
#include "base/shape.h"
#include "base/tracing.h"
#include "hal/api_detail.h"
#include "hal/buffer.h"
#include "hal/buffer_view.h"
#include "hal/fence.h"
#include "hal/heap_buffer.h"
#include "hal/semaphore.h"

namespace iree {
namespace hal {

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

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

IREE_API_EXPORT iree_status_t
iree_hal_buffer_retain(iree_hal_buffer_t* buffer) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_retain");
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_release(iree_hal_buffer_t* buffer) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_release");
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_byte_length");
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
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  std::memset(out_mapped_memory, 0, sizeof(*out_mapped_memory));

  auto* buffer_handle = reinterpret_cast<Buffer*>(buffer);
  if (!buffer_handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

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

  out_mapped_memory->contents = {const_cast<uint8_t*>(mapping_storage->data()),
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

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, iree_shape_t shape, int8_t element_size,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_create");

  if (!out_buffer_view) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  *out_buffer_view = nullptr;

  if (!buffer) {
    return IREE_STATUS_INVALID_ARGUMENT;
  } else if (shape.rank > kMaxRank || element_size <= 0) {
    return IREE_STATUS_OUT_OF_RANGE;
  }

  // Allocate and initialize the iree_hal_buffer_view struct.
  iree_hal_buffer_view* handle = nullptr;
  IREE_API_RETURN_IF_API_ERROR(allocator.alloc(
      allocator.self, sizeof(*handle), reinterpret_cast<void**>(&handle)));
  new (handle) iree_hal_buffer_view();
  handle->allocator = allocator;

  handle->impl.buffer = add_ref(reinterpret_cast<Buffer*>(buffer));
  handle->impl.shape = {shape.dims, shape.rank};
  handle->impl.element_size = element_size;

  *out_buffer_view = reinterpret_cast<iree_hal_buffer_view_t*>(handle);
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_view_retain(iree_hal_buffer_view_t* buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_retain");
  auto* handle = reinterpret_cast<iree_hal_buffer_view*>(buffer_view);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_view_release(iree_hal_buffer_view_t* buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_release");
  auto* handle = reinterpret_cast<iree_hal_buffer_view*>(buffer_view);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_assign(
    iree_hal_buffer_view_t* buffer_view, iree_hal_buffer_t* buffer,
    iree_shape_t shape, int8_t element_size) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_assign");
  auto* handle = reinterpret_cast<iree_hal_buffer_view*>(buffer_view);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->impl.buffer.reset();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_view_reset(iree_hal_buffer_view_t* buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_reset");
  auto* handle = reinterpret_cast<iree_hal_buffer_view*>(buffer_view);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->impl.buffer.reset();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_view_buffer(
    const iree_hal_buffer_view_t* buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_buffer");
  const auto* handle =
      reinterpret_cast<const iree_hal_buffer_view*>(buffer_view);
  CHECK(handle) << "NULL buffer_view handle";
  return reinterpret_cast<iree_hal_buffer_t*>(handle->impl.buffer.get());
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_shape_t* out_shape) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_shape");

  if (!out_shape) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  out_shape->rank = 0;

  const auto* handle =
      reinterpret_cast<const iree_hal_buffer_view*>(buffer_view);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }

  const auto& shape = handle->impl.shape;
  return ToApiShape(shape, out_shape);
}

IREE_API_EXPORT int8_t
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_element_size");
  const auto* handle =
      reinterpret_cast<const iree_hal_buffer_view*>(buffer_view);
  CHECK(handle) << "NULL buffer_view handle";
  return handle->impl.element_size;
}

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_retain(iree_hal_semaphore_t* semaphore) {
  IREE_TRACE_SCOPE0("iree_hal_semaphore_retain");
  auto* handle = reinterpret_cast<Semaphore*>(semaphore);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_release(iree_hal_semaphore_t* semaphore) {
  IREE_TRACE_SCOPE0("iree_hal_semaphore_release");
  auto* handle = reinterpret_cast<Semaphore*>(semaphore);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

//===----------------------------------------------------------------------===//
// iree::hal::Fence
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_fence_retain(iree_hal_fence_t* fence) {
  IREE_TRACE_SCOPE0("iree_hal_fence_retain");
  auto* handle = reinterpret_cast<Fence*>(fence);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t iree_hal_fence_release(iree_hal_fence_t* fence) {
  IREE_TRACE_SCOPE0("iree_hal_fence_release");
  auto* handle = reinterpret_cast<Fence*>(fence);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

}  // namespace hal
}  // namespace iree
