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

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_API_H_
#define IREE_HAL_API_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

typedef struct iree_hal_buffer iree_hal_buffer_t;
typedef struct iree_hal_buffer_view iree_hal_buffer_view_t;
typedef struct iree_hal_semaphore iree_hal_semaphore_t;
typedef struct iree_hal_fence iree_hal_fence_t;

// Reference to a buffer's mapped memory.
typedef struct {
  // Contents of the buffer. Behavior is undefined if an access is performed
  // whose type was not specified during mapping.
  iree_byte_span_t contents;

  // Used internally - do not modify.
  uint64_t reserved[8];
} iree_hal_mapped_memory_t;

// A bitfield specifying how memory will be accessed in a mapped memory region.
typedef enum {
  // Memory is not mapped.
  IREE_HAL_MEMORY_ACCESS_NONE = 0,
  // Memory will be read.
  // If a buffer is only mapped for reading it may still be possible to write to
  // it but the results will be undefined (as it may present coherency issues).
  IREE_HAL_MEMORY_ACCESS_READ = 1 << 0,
  // Memory will be written.
  // If a buffer is only mapped for writing it may still be possible to read
  // from it but the results will be undefined or incredibly slow (as it may
  // be mapped by the driver as uncached).
  IREE_HAL_MEMORY_ACCESS_WRITE = 1 << 1,
  // Memory will be discarded prior to mapping.
  // The existing contents will be undefined after mapping and must be written
  // to ensure validity.
  IREE_HAL_MEMORY_ACCESS_DISCARD = 1 << 2,
  // Memory will be discarded and completely overwritten in a single operation.
  IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE =
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD,
  // Memory may have any operation performed on it.
  IREE_HAL_MEMORY_ACCESS_ALL = IREE_HAL_MEMORY_ACCESS_READ |
                               IREE_HAL_MEMORY_ACCESS_WRITE |
                               IREE_HAL_MEMORY_ACCESS_DISCARD,
} iree_hal_memory_access_t;

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Returns a reference to a subspan of the |buffer|.
// If |byte_length| is IREE_WHOLE_BUFFER the remaining bytes in the buffer after
// |byte_offset| (possibly 0) will be selected.
//
// The parent buffer will remain alive for the lifetime of the subspan
// returned. If the subspan is a small portion this may cause additional
// memory to remain allocated longer than required.
//
// Returns the given |buffer| if the requested span covers the entire range.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_subspan(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer);

// Retains the given |buffer| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_buffer_retain(iree_hal_buffer_t* buffer);

// Releases the given |buffer| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_buffer_release(iree_hal_buffer_t* buffer);

// Returns the size in bytes of the buffer.
IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer);

// Sets a range of the buffer to binary zero.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_buffer_zero(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length);

// Sets a range of the buffer to the given value.
// Only |pattern_length| values with 1, 2, or 4 bytes are supported.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_buffer_fill(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length, const void* pattern,
                     iree_host_size_t pattern_length);

// Reads a block of data from the buffer at the given offset.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_read_data(
    iree_hal_buffer_t* buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length);

// Writes a block of byte data into the buffer at the given offset.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_write_data(
    iree_hal_buffer_t* buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length);

// Maps the buffer to be accessed as a host pointer into |out_mapped_memory|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_map(
    iree_hal_buffer_t* buffer, iree_hal_memory_access_t memory_access,
    iree_device_size_t element_offset, iree_device_size_t element_length,
    iree_hal_mapped_memory_t* out_mapped_memory);

// Unmaps the buffer as was previously mapped to |mapped_memory|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_unmap(
    iree_hal_buffer_t* buffer, iree_hal_mapped_memory_t* mapped_memory);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Creates a buffer view with the given |buffer|, which may be nullptr.
// |out_buffer_view| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, iree_shape_t shape, int8_t element_size,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view);

// Retains the given |buffer_view| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_buffer_view_retain(iree_hal_buffer_view_t* buffer_view);

// Releases the given |buffer_view| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_buffer_view_release(iree_hal_buffer_view_t* buffer_view);

// Sets the buffer view to point at the new |buffer| with the given metadata.
// To clear a buffer_view to empty use iree_hal_buffer_view_reset.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_assign(
    iree_hal_buffer_view_t* buffer_view, iree_hal_buffer_t* buffer,
    iree_shape_t shape, int8_t element_size);

// Resets the buffer view to have an empty buffer and shape.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_buffer_view_reset(iree_hal_buffer_view_t* buffer_view);

// Returns the buffer underlying the buffer view.
// The caller must retain the returned buffer if they want to continue using it.
IREE_API_EXPORT iree_hal_buffer_t* IREE_API_CALL
iree_hal_buffer_view_buffer(const iree_hal_buffer_view_t* buffer_view);

// Returns the shape of the buffer view in |out_shape|.
// If there is not enough space in |out_shape| to store all dimensions then
// IREE_STATUS_OUT_OF_RANGE is returned and |out_shape|.rank is set to the rank.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_shape_t* out_shape);

// Returns the size of each element in the buffer view in bytes.
IREE_API_EXPORT int8_t IREE_API_CALL
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Retains the given |semaphore| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_retain(iree_hal_semaphore_t* semaphore);

// Releases the given |semaphore| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_release(iree_hal_semaphore_t* semaphore);

#endif  // IREE_API_NO_PROTOTYPES

//===----------------------------------------------------------------------===//
// iree::hal::Fence
//===----------------------------------------------------------------------===//

#ifndef IREE_API_NO_PROTOTYPES

// Retains the given |fence| for the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_fence_retain(iree_hal_fence_t* fence);

// Releases the given |fence| from the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_fence_release(iree_hal_fence_t* fence);

#endif  // IREE_API_NO_PROTOTYPES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_API_H_
