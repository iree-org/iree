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

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_hal_allocator_retain(iree_hal_allocator_t* allocator) {
  IREE_TRACE_SCOPE0("iree_hal_allocator_retain");
  auto* handle = reinterpret_cast<Allocator*>(allocator);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_allocator_release(iree_hal_allocator_t* allocator) {
  IREE_TRACE_SCOPE0("iree_hal_allocator_release");
  auto* handle = reinterpret_cast<Allocator*>(allocator);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
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
  IREE_API_RETURN_IF_API_ERROR(iree_allocator_malloc(
      allocator, sizeof(*handle), reinterpret_cast<void**>(&handle)));
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
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

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
iree_hal_command_buffer_retain(iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_retain");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_release(iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_release");
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
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

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_hal_device_retain(iree_hal_device_t* device) {
  IREE_TRACE_SCOPE0("iree_hal_device_retain");
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_device_release(iree_hal_device_t* device) {
  IREE_TRACE_SCOPE0("iree_hal_device_release");
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_device_allocator(iree_hal_device_t* device) {
  auto* handle = reinterpret_cast<Device*>(device);
  if (!handle) {
    return nullptr;
  }
  return reinterpret_cast<iree_hal_allocator_t*>(handle->allocator());
}

//===----------------------------------------------------------------------===//
// iree::hal::Driver
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_retain(iree_hal_driver_t* driver) {
  IREE_TRACE_SCOPE0("iree_hal_driver_retain");
  auto* handle = reinterpret_cast<Driver*>(driver);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_release(iree_hal_driver_t* driver) {
  IREE_TRACE_SCOPE0("iree_hal_driver_release");
  auto* handle = reinterpret_cast<Driver*>(driver);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->ReleaseReference();
  return IREE_STATUS_OK;
}

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
  IREE_API_RETURN_IF_API_ERROR(iree_allocator_malloc(
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
  IREE_API_RETURN_IF_API_ERROR(iree_allocator_malloc(
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

IREE_API_EXPORT iree_status_t
iree_hal_executable_retain(iree_hal_executable_t* executable) {
  IREE_TRACE_SCOPE0("iree_hal_executable_retain");
  auto* handle = reinterpret_cast<Executable*>(executable);
  if (!handle) {
    return IREE_STATUS_INVALID_ARGUMENT;
  }
  handle->AddReference();
  return IREE_STATUS_OK;
}

IREE_API_EXPORT iree_status_t
iree_hal_executable_release(iree_hal_executable_t* executable) {
  IREE_TRACE_SCOPE0("iree_hal_executable_release");
  auto* handle = reinterpret_cast<Executable*>(executable);
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

}  // namespace hal
}  // namespace iree
