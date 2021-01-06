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

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

typedef struct iree_hal_allocator iree_hal_allocator_t;
typedef struct iree_hal_buffer iree_hal_buffer_t;
typedef struct iree_hal_buffer_view iree_hal_buffer_view_t;
typedef struct iree_hal_command_buffer iree_hal_command_buffer_t;
typedef struct iree_hal_descriptor_set iree_hal_descriptor_set_t;
typedef struct iree_hal_descriptor_set_layout iree_hal_descriptor_set_layout_t;
typedef struct iree_hal_device iree_hal_device_t;
typedef struct iree_hal_driver iree_hal_driver_t;
typedef struct iree_hal_driver_registry_s iree_hal_driver_registry_t;
typedef struct iree_hal_executable iree_hal_executable_t;
typedef struct iree_hal_executable_cache iree_hal_executable_cache_t;
typedef struct iree_hal_executable_layout iree_hal_executable_layout_t;
typedef struct iree_hal_semaphore iree_hal_semaphore_t;

// Reference to a buffer's mapped memory.
typedef struct {
  // Contents of the buffer. Behavior is undefined if an access is performed
  // whose type was not specified during mapping.
  iree_byte_span_t contents;

  // Used internally - do not modify.
  uint64_t reserved[8];
} iree_hal_mapped_memory_t;

// A bitfield specifying properties for a memory type.
enum iree_hal_memory_type_e {
  IREE_HAL_MEMORY_TYPE_NONE = 0u,

  // Memory is lazily allocated by the device and only exists transiently.
  // This is the optimal mode for memory used only within a single command
  // buffer. Transient buffers, even if they have
  // IREE_HAL_MEMORY_TYPE_HOST_VISIBLE set, should be treated as device-local
  // and opaque as they may have no memory attached to them outside of the time
  // they are being evaluated on devices.
  //
  // This flag can be treated as a hint in most cases; allocating a buffer with
  // it set _may_ return the same as if it had not be set. Certain allocation
  // routines may use the hint to more tightly control reuse or defer wiring the
  // memory.
  IREE_HAL_MEMORY_TYPE_TRANSIENT = 1u << 0,

  // Memory allocated with this type can be mapped for host access using
  // iree_hal_buffer_map.
  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE = 1u << 1,

  // The host cache management commands MappedMemory::Flush and
  // MappedMemory::Invalidate are not needed to flush host writes
  // to the device or make device writes visible to the host, respectively.
  IREE_HAL_MEMORY_TYPE_HOST_COHERENT = 1u << 2,

  // Memory allocated with this type is cached on the host. Host memory
  // accesses to uncached memory are slower than to cached memory, however
  // uncached memory is always host coherent. MappedMemory::Flush must be used
  // to ensure the device has visibility into any changes made on the host and
  // Invalidate must be used to ensure the host has visibility into any changes
  // made on the device.
  IREE_HAL_MEMORY_TYPE_HOST_CACHED = 1u << 3,

  // Memory is accessible as normal host allocated memory.
  IREE_HAL_MEMORY_TYPE_HOST_LOCAL =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT,

  // Memory allocated with this type is visible to the device for execution.
  // Being device visible does not mean the same thing as
  // IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL. Though an allocation may be visible to
  // the device and therefore useable for execution it may require expensive
  // mapping or implicit transfers.
  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE = 1u << 4,

  // Memory allocated with this type is the most efficient for device access.
  // Devices may support using memory that is not device local via
  // IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE but doing so can incur non-trivial
  // performance penalties. Device local memory, on the other hand, is
  // guaranteed to be fast for all operations.
  IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL =
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE | (1u << 5),
};
typedef uint32_t iree_hal_memory_type_t;

// A bitfield specifying how memory will be accessed in a mapped memory region.
enum iree_hal_memory_access_e {
  // Memory is not mapped.
  IREE_HAL_MEMORY_ACCESS_NONE = 0u,
  // Memory will be read.
  // If a buffer is only mapped for reading it may still be possible to write to
  // it but the results will be undefined (as it may present coherency issues).
  IREE_HAL_MEMORY_ACCESS_READ = 1u << 0,
  // Memory will be written.
  // If a buffer is only mapped for writing it may still be possible to read
  // from it but the results will be undefined or incredibly slow (as it may
  // be mapped by the driver as uncached).
  IREE_HAL_MEMORY_ACCESS_WRITE = 1u << 1,
  // Memory will be discarded prior to mapping.
  // The existing contents will be undefined after mapping and must be written
  // to ensure validity.
  IREE_HAL_MEMORY_ACCESS_DISCARD = 1u << 2,
  // Memory will be discarded and completely overwritten in a single operation.
  IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE =
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD,
  // A flag that can be applied to any access type to indicate that the buffer
  // storage being accessed may alias with other accesses occurring concurrently
  // within or across operations. The lack of the flag indicates that the access
  // is guaranteed not to alias (ala C's `restrict` keyword).
  IREE_HAL_MEMORY_ACCESS_MAY_ALIAS = 1u << 3,
  // Memory may have any operation performed on it.
  IREE_HAL_MEMORY_ACCESS_ALL = IREE_HAL_MEMORY_ACCESS_READ |
                               IREE_HAL_MEMORY_ACCESS_WRITE |
                               IREE_HAL_MEMORY_ACCESS_DISCARD,
};
typedef uint32_t iree_hal_memory_access_t;

// Bitfield that defines how a buffer is intended to be used.
// Usage allows the driver to appropriately place the buffer for more
// efficient operations of the specified types.
enum iree_hal_buffer_usage_e {
  IREE_HAL_BUFFER_USAGE_NONE = 0u,

  // The buffer, once defined, will not be mapped or updated again.
  // This should be used for uniform parameter values such as runtime
  // constants for executables. Doing so may allow drivers to inline values or
  // represent them in command buffers more efficiently (avoiding memory reads
  // or swapping, etc).
  IREE_HAL_BUFFER_USAGE_CONSTANT = 1u << 0,

  // The buffer can be used as the source or target of a transfer command
  // (CopyBuffer, UpdateBuffer, etc).
  //
  // If |IREE_HAL_BUFFER_USAGE_MAPPING| is not specified drivers may safely
  // assume that the host may never need visibility of this buffer as all
  // accesses will happen via command buffers.
  IREE_HAL_BUFFER_USAGE_TRANSFER = 1u << 1,

  // The buffer can be mapped by the host application for reading and writing.
  //
  // As mapping may require placement in special address ranges or system
  // calls to enable visibility the driver can use the presence (or lack of)
  // this flag to perform allocation-type setup and avoid initial mapping
  // overhead.
  IREE_HAL_BUFFER_USAGE_MAPPING = 1u << 2,

  // The buffer can be provided as an input or output to an executable.
  // Buffers of this type may be directly used by drivers during dispatch.
  IREE_HAL_BUFFER_USAGE_DISPATCH = 1u << 3,

  // Buffer may be used for any operation.
  IREE_HAL_BUFFER_USAGE_ALL = IREE_HAL_BUFFER_USAGE_TRANSFER |
                              IREE_HAL_BUFFER_USAGE_MAPPING |
                              IREE_HAL_BUFFER_USAGE_DISPATCH,
};
typedef uint32_t iree_hal_buffer_usage_t;

// An opaque driver-specific handle to identify different devices.
typedef uintptr_t iree_hal_device_id_t;

// Describes an enumerated HAL device.
typedef struct {
  // Opaque handle used by drivers. Not valid across driver instances.
  iree_hal_device_id_t device_id;
  // Name of the device as returned by the API.
  iree_string_view_t name;
} iree_hal_device_info_t;

// An opaque factory-specific handle to identify different drivers.
typedef uint64_t iree_hal_driver_id_t;

#define IREE_HAL_DRIVER_ID_INVALID 0ull

// Describes a driver providing device enumeration and creation.
// The lifetime of memory referenced by this structure (such as strings) is
// dependent on where it originated.
//
// * When using iree_hal_driver_registry_enumerate the driver info is copied
//   into memory owned by the caller.
// * When queried from a live driver with iree_hal_driver_info the memory is
//   only guaranteed to live for as long as the driver is.
// * When enumerating via factories the information may be valid only while the
//   driver registry lock is held.
typedef struct {
  IREE_API_UNSTABLE

  // Opaque handle used by factories. Unique across all factories.
  iree_hal_driver_id_t driver_id;

  // Canonical name of the driver as used in command lines, documentation, etc.
  // Examples: 'metal', 'vulkan'
  iree_string_view_t driver_name;

  // Full human-readable name of the driver for display.
  // Examples: 'Vulkan 1.2 (NVIDIA)'.
  iree_string_view_t full_name;

  // TODO(benvanik): version information; useful if wanting to expose multiple
  // versions that may have completely different implementations (like vulkan
  // 1.0, 1.1, and 1.2) but allow a nice sort/selection process.
  // TODO(benvanik): triple, feature flags, etc.
} iree_hal_driver_info_t;

// A bitfield specifying the mode of operation for a command buffer.
enum iree_hal_command_buffer_mode_e {
  // Command buffer will be submitted once and never used again.
  // This may enable in-place patching of command buffers that reduce overhead
  // when it's known that command buffers will not be reused.
  IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT = 1u << 0,
};
typedef uint32_t iree_hal_command_buffer_mode_t;

// A bitfield specifying the category of commands in a command queue.
enum iree_hal_command_category_e {
  // Command is considered a transfer operation (memcpy, etc).
  IREE_HAL_COMMAND_CATEGORY_TRANSFER = 1u << 0,
  // Command is considered a dispatch operation (dispatch/execute).
  IREE_HAL_COMMAND_CATEGORY_DISPATCH = 1u << 1,
  // Commands may be of any type.
  // Using this value may prevent optimizations and if possible callers should
  // always specify the strictest set possible (for example, only transfer
  // commands to ensure they get placed on a DMA queue).
  IREE_HAL_COMMAND_CATEGORY_ANY =
      IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH,
};
typedef uint32_t iree_hal_command_category_t;

// Specifies the type of a descriptor in a descriptor set.
enum iree_hal_descriptor_type_e {
  IREE_HAL_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6u,
  IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7u,
  IREE_HAL_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = 8u,
  IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC = 9u,
};
typedef uint32_t iree_hal_descriptor_type_t;

// Specifies a descriptor set binding.
// The range specified by [offset, length) will be made available to executables
// on the given binding. If the descriptor type is dynamic then the range will
// be [offset + dynamic_offset, length).
//
// The IREE HAL buffer type may internally be offset; such offset is applied
// here as if it were the base address of the buffer. Note that the offset will
// be applied at the time the binding is recording into the command buffer.
//
// Maps to VkDescriptorSetBinding.
typedef struct {
  // The binding number of this entry and corresponds to a resource of the
  // same binding number in the executable interface.
  int32_t binding;
  // Buffer bound to the binding number.
  // May be nullptr if the binding is not used by the executable.
  iree_hal_buffer_t* buffer;
  // Offset, in bytes, into the buffer that the binding starts at.
  // If the descriptor type is dynamic this will be added to the dynamic
  // offset provided during binding.
  iree_device_size_t offset;
  // Length, in bytes, of the buffer that is available to the executable.
  // This can be IREE_WHOLE_BUFFER, however note that if the entire buffer
  // contents are larger than supported by the device (~128MiB, usually) this
  // will fail. If the descriptor type is dynamic this will be used for all
  // ranges regardless of offset.
  iree_device_size_t length;
} iree_hal_descriptor_set_binding_t;

// Specifies the usage type of the descriptor set.
enum iree_hal_descriptor_set_layout_usage_type_e {
  // Descriptor set will be initialized once and never changed.
  IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE = 0u,
  // Descriptor set is never created and instead used with push descriptors.
  IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_PUSH_ONLY = 1u,
};
typedef uint32_t iree_hal_descriptor_set_layout_usage_type_t;

// Specifies a descriptor set layout binding.
//
// Maps to VkDescriptorSetLayoutBinding.
typedef struct {
  // The binding number of this entry and corresponds to a resource of the
  // same binding number in the executable interface.
  int32_t binding;
  // Specifies which type of resource descriptors are used for this binding.
  iree_hal_descriptor_type_t type;
  // Specifies the memory access performed by the executables.
  iree_hal_memory_access_t access;
} iree_hal_descriptor_set_layout_binding_t;

// Describes features supported by a device.
// These flags indicate the availability of features that may be enabled at the
// request of the calling application. Note that certain features may disable
// runtime optimizations or require compilation flags to ensure the required
// metadata is present in executables.
enum iree_hal_device_feature_e {
  IREE_HAL_DEVICE_FEATURE_NONE = 0,

  // Device supports executable debugging.
  // When present executables *may* be compiled with
  // IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_DEBUGGING and will have usable
  // debugging related methods. Note that if the input executables do not have
  // embedded debugging information they still may not be able to perform
  // disassembly or fine-grained breakpoint insertion.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_DEBUGGING = 1 << 0,

  // Device supports executable coverage information.
  // When present executables *may* be compiled with
  // IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_COVERAGE and will produce
  // coverage buffers during dispatch. Note that input executables must have
  // partial embedded debug information to allow mapping back to source offsets.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_COVERAGE = 1 << 1,

  // Device supports executable and command queue profiling.
  // When present executables *may* be compiled with
  // IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_PROFILING and will produce
  // profiling buffers during dispatch. Note that input executables must have
  // partial embedded debug information to allow mapping back to source offsets.
  IREE_HAL_DEVICE_FEATURE_SUPPORTS_PROFILING = 1 << 2,
};
typedef uint32_t iree_hal_device_feature_t;

// An identifier for executable formats used to query support.
typedef uint32_t iree_hal_executable_format_t;

// Defines how the executable cache performs preparation.
enum iree_hal_executable_caching_mode_e {
  // Allows the cache to reference the provided executable_data after it has
  // prepared the executable. Callers must ensure the data remains valid for the
  // lifetime of the cache. If memory mapping constant executable data from
  // disk this can be used to avoid copies.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA = 1u << 0,
  // Allows the prepared executable to be cached persistently (on disk/etc).
  // Enable for any executable that is likely to be used in future runs.
  // Note that not all caches support persistent serialization and this is just
  // a hint.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING = 1u << 1,
  // Allows the cache to optimize the executable as much as it can.
  // This may cause preparation to take significantly longer while (hopefully)
  // improving runtime performance. Avoid for one-shot executables.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION = 1u << 2,
  // Enables Executable debugging methods if supported by the device and
  // executable. This may disable certain optimizations or retain additional
  // data to allow disassembly, stepping, etc.
  //
  // Device must support the IREE_HAL_DEVICE_FEATURE_SUPPORTS_DEBUGGING feature
  // and executables must support the ExecutableFeature::kDebugging feature.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_DEBUGGING = 1u << 3,
  // Enables Executable coverage if supported by the device and executable.
  // Depending on the optimization mode this may produce partial coverage
  // results (for example, when certain source operations were optimized away).
  //
  // Device must support the IREE_HAL_DEVICE_FEATURE_SUPPORTS_COVERAGE feature
  // and executables must support the ExecutableFeature::kCoverage feature.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_COVERAGE = 1u << 4,
  // Enables Executable profiling if supported by the device and executable.
  // Depending on the optimization mode this may produce partial profiling
  // results. Profiling attribution (whether to the entire executable or
  // specific operations) depends on the implementation.
  //
  // Device must support the IREE_HAL_DEVICE_FEATURE_SUPPORTS_PROFILING feature
  // and executables must support the ExecutableFeature::kProfiling feature.
  IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_PROFILING = 1u << 5,
  // Default caching mode.
  IREE_HAL_EXECUTABLE_CACHING_MODE_DEFAULT =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION,
};
typedef uint32_t iree_hal_executable_caching_mode_t;

// Bitfield specifying which execution stage a barrier should start/end at.
//
// Maps to VkPipelineStageFlagBits.
enum iree_hal_execution_stage_e {
  // Top of the pipeline when commands are initially issued by the device.
  IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE = 1u << 0,
  // Stage of the pipeline when dispatch parameter data is consumed.
  IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS = 1u << 1,
  // Stage where dispatch commands execute.
  IREE_HAL_EXECUTION_STAGE_DISPATCH = 1u << 2,
  // Stage where transfer (copy/clear/fill/etc) commands execute.
  IREE_HAL_EXECUTION_STAGE_TRANSFER = 1u << 3,
  // Final stage in the pipeline when commands are retired on the device.
  IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE = 1u << 4,
  // Pseudo-stage for read/writes by the host. Not executed on device.
  IREE_HAL_EXECUTION_STAGE_HOST = 1u << 5,
};
typedef uint32_t iree_hal_execution_stage_t;

// Bitfield specifying which scopes will access memory and how.
//
// Maps to VkAccessFlagBits.
enum iree_hal_access_scope_e {
  // Read access to indirect command data as part of an indirect dispatch.
  IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ = 1u << 0,
  // Constant uniform buffer reads by the device.
  IREE_HAL_ACCESS_SCOPE_CONSTANT_READ = 1u << 1,
  // Storage buffer reads by dispatch commands.
  IREE_HAL_ACCESS_SCOPE_DISPATCH_READ = 1u << 2,
  // Storage buffer writes by dispatch commands.
  IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE = 1u << 3,
  // Source of a transfer operation.
  IREE_HAL_ACCESS_SCOPE_TRANSFER_READ = 1u << 4,
  // Target of a transfer operation.
  IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE = 1u << 5,
  // Read operation by the host through mapped memory.
  IREE_HAL_ACCESS_SCOPE_HOST_READ = 1u << 6,
  // Write operation by the host through mapped memory.
  IREE_HAL_ACCESS_SCOPE_HOST_WRITE = 1u << 7,
  // External/non-specific read.
  IREE_HAL_ACCESS_SCOPE_MEMORY_READ = 1u << 8,
  // External/non-specific write.
  IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE = 1u << 9,
};
typedef uint32_t iree_hal_access_scope_t;

// Defines a global memory barrier.
// These are cheaper to encode than buffer-specific barriers but may cause
// stalls and bubbles in device pipelines if applied too broadly. Prefer them
// over equivalently large sets of buffer-specific barriers (such as when
// completely changing execution contexts).
//
// Maps to VkMemoryBarrier.
typedef struct {
  // All access scopes prior-to the barrier (inclusive).
  iree_hal_access_scope_t source_scope;
  // All access scopes following the barrier (inclusive).
  iree_hal_access_scope_t target_scope;
} iree_hal_memory_barrier_t;

// Defines a memory barrier that applies to a range of a specific buffer.
// Use of these (vs. global memory barriers) provides fine-grained execution
// ordering to device command processors and allows for more aggressive
// reordering.
//
// Maps to VkBufferMemoryBarrier.
typedef struct {
  // All access scopes prior-to the barrier (inclusive).
  iree_hal_access_scope_t source_scope;
  // All access scopes following the barrier (inclusive).
  iree_hal_access_scope_t target_scope;
  // Buffer the barrier is restricted to.
  // The barrier will apply to the entire physical device allocation.
  iree_hal_buffer_t* buffer;
  // Relative offset/length within |buffer| (which may itself be mapped into the
  // device allocation at an offset).
  iree_device_size_t offset;
  iree_device_size_t length;
} iree_hal_buffer_barrier_t;

// A list of semaphores and their corresponding payloads.
// When signaling each semaphore will be set to the new payload value provided.
// When waiting each semaphore must reach or exceed the payload value.
typedef struct {
  iree_host_size_t count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* payload_values;
} iree_hal_semaphore_list_t;

// A single batch of command buffers submitted to a device queue.
// All of the wait semaphores must reach or exceed the given payload value prior
// to the batch beginning execution. Each command buffer begins execution in the
// order it is present in the list, though note that the command buffers
// execute concurrently and require internal synchronization via events if there
// are any dependencies between them. Only after all command buffers have
// completed will the signal semaphores be updated to the provided payload
// values.
//
// Matches Vulkan's VkSubmitInfo:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSubmitInfo.html
// Note that as the HAL only models timeline semaphores we take the payload
// values directly in this struct; see:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkTimelineSemaphoreSubmitInfo.html
typedef struct {
  // Semaphores to wait on prior to executing any command buffer.
  iree_hal_semaphore_list_t wait_semaphores;

  // Command buffers to execute, in order.
  iree_host_size_t command_buffer_count;
  iree_hal_command_buffer_t** command_buffers;

  // Semaphores to signal once all command buffers have completed execution.
  iree_hal_semaphore_list_t signal_semaphores;
} iree_hal_submission_batch_t;

// Defines how a multi-wait operation treats the results of multiple semaphores.
enum iree_hal_wait_mode_e {
  // Waits for all semaphores to reach or exceed their specified values.
  IREE_HAL_WAIT_MODE_ALL = 0,
  // Waits for one or more semaphores to reach or exceed their specified values.
  IREE_HAL_WAIT_MODE_ANY = 1,
};
typedef uint8_t iree_hal_wait_mode_t;

// Keep these in sync with iree/compiler/Dialect/HAL/IR/HALTypes.cpp

enum iree_hal_numerical_type_e {
  IREE_HAL_NUMERICAL_TYPE_UNKNOWN = 0x00u,
  IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED = 0x01u,
  IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED = 0x02u,
  // TODO(benvanik): specialize with semantics from APFloat.
  IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE = 0x03u,
};
typedef uint8_t iree_hal_numerical_type_t;

#define IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, bit_count) \
  (((uint32_t)(numerical_type) << 24) | (uint32_t)(bit_count))

#define iree_hal_make_element_type(numerical_type, bit_count) \
  (iree_hal_element_type_t)(                                  \
      IREE_HAL_ELEMENT_TYPE_VALUE(numerical_type, bit_count))
#define iree_hal_element_numerical_type(element_type) \
  (iree_hal_numerical_type_t)((uint32_t)(element_type) >> 24)
#define iree_hal_element_bit_count(element_type) (size_t)((element_type)&0xFF)
#define iree_hal_element_byte_count(element_type) \
  ((iree_hal_element_bit_count(element_type) + 8 - 1) / 8)

// Defines the element type of a buffer in a standard format.
//
// Composed as a 32-bit bitfield to allow for opaque data types. Use
// iree_hal_make_element_type to make a bitfield with the appropriate ordering.
//
//   MSB ----------------------------------------------- LSB
//   [numerical type] [reserved] [reserved] [number of bits]
//
// clang-format off
enum iree_hal_element_type_e {
  IREE_HAL_ELEMENT_TYPE_NONE             = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,             0),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_8         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,             8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_16        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_32        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_OPAQUE_64        = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_UNKNOWN,            64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_8           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,      8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_8           = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,    8),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_16          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_16          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_32          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_32          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_SINT_64          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED,     64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_UINT_64          = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED,   64),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_16         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         16),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_32         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         32),  // NOLINT
  IREE_HAL_ELEMENT_TYPE_FLOAT_64         = IREE_HAL_ELEMENT_TYPE_VALUE(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE,         64),  // NOLINT
};
typedef uint32_t iree_hal_element_type_t;
// clang-format on

// A dimension within a shape.
typedef int32_t iree_hal_dim_t;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Parses a serialized set of shape dimensions using the canonical shape format
// (the same as produced by iree_hal_format_shape).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_shape(
    iree_string_view_t value, iree_host_size_t shape_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank);

// Converts shape dimensions into a `4x5x6` format.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_format_shape(const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
                      iree_host_size_t buffer_capacity, char* buffer,
                      iree_host_size_t* out_buffer_length);

// Parses a serialized iree_hal_element_type_t and sets |out_element_type| if
// it is valid. The format is the same as produced by
// iree_hal_format_element_type.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_element_type(
    iree_string_view_t value, iree_hal_element_type_t* out_element_type);

// Converts an iree_hal_element_type_t enum value to a canonical string
// representation, like `IREE_HAL_ELEMENT_TYPE_FLOAT_16` to `f16`.
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_format_element_type(
    iree_hal_element_type_t element_type, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

// Parses a serialized element of |element_type| to its in-memory form.
// |data_ptr| must be at least large enough to contain the bytes of the element.
// For example, "1.2" of type IREE_HAL_ELEMENT_TYPE_FLOAT32 will write the 4
// byte float value of 1.2 to |data_ptr|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_element(
    iree_string_view_t data_str, iree_hal_element_type_t element_type,
    iree_byte_span_t data_ptr);

// Converts a single element of |element_type| to a string.
//
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters. Returns
// IREE_STATUS_OUT_OF_RANGE if the buffer capacity is insufficient to hold the
// formatted elements and |out_buffer_length| will contain the required size.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_format_element(
    iree_const_byte_span_t data, iree_hal_element_type_t element_type,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length);

// Parses a serialized set of elements of the given |element_type|.
// The resulting parsed data is written to |data_ptr|, which must be at least
// large enough to contain the parsed elements. The format is the same as
// produced by iree_hal_format_buffer_elements. Supports additional inputs of
// empty to denote a 0 fill and a single element to denote a splat.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_buffer_elements(
    iree_string_view_t data_str, iree_hal_element_type_t element_type,
    iree_byte_span_t data_ptr);

// Converts a shaped buffer of |element_type| elements to a string.
// This will include []'s to denote each dimension, for example for a shape of
// 2x3 the elements will be formatted as `[1 2 3][4 5 6]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
//
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters. Returns
// IREE_STATUS_OUT_OF_RANGE if the buffer capacity is insufficient to hold the
// formatted elements and |out_buffer_length| will contain the required size.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_format_buffer_elements(
    iree_const_byte_span_t data, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

//===----------------------------------------------------------------------===//
// iree_hal_allocator_t
//===----------------------------------------------------------------------===//

// Creates a host-local heap allocator that can be used when buffers are
// required that will not interact with a real hardware device (such as those
// used in file IO or tests). Buffers allocated with this will not be compatible
// with real device allocators and will likely incur a copy if used.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_allocator_create_host_local(iree_allocator_t allocator,
                                     iree_hal_allocator_t** out_allocator);

// Retains the given |allocator| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_allocator_retain(iree_hal_allocator_t* allocator);

// Releases the given |allocator| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_allocator_release(iree_hal_allocator_t* allocator);

// Calculates the allocation size of a buffer.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_size(
    const iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_device_size_t* out_allocation_size);

// Calculates a byte offset into a buffer at the given indices.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_offset(
    const iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    const iree_hal_dim_t* indices, size_t indices_count,
    iree_device_size_t* out_offset);

// Calculates a byte range into a buffer of the given contiguous range.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_range(
    const iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length);

// Allocates a buffer from the allocator.
// Fails if the memory type requested for the given usage cannot be serviced.
// Callers can use iree_hal_allocator_can_allocate to decide their memory use
// strategy.
//
// The memory type of the buffer returned may differ from the requested value
// if the device can provide more functionality; for example, if requesting
// IREE_HAL_MEMORY_TYPE_HOST_VISIBLE but the memory is really host cached you
// may get a buffer back with IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
// IREE_HAL_MEMORY_TYPE_HOST_CACHED. The only requirement is that the buffer
// satisfy the required bits.
//
// Fails if it is not possible to allocate and satisfy all placements for the
// requested |buffer_usage|.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t buffer_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer);

// Wraps an existing host allocation in a buffer.
// Ownership of the allocation remains with the caller and the memory must
// remain valid for so long as the buffer may be in use.
//
// Fails if the allocator cannot access host memory in this way.
// |out_buffer| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_wrap_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t buffer_usage, iree_byte_span_t data,
    iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t
//===----------------------------------------------------------------------===//

// Allocated memory buffer wrapper type and utilities.
//
// Buffers are the basic unit of memory used by the inference system. They may
// be allocated such that they are accessible from the host (normal C++ code
// running on the main CPU), a particular device (such as an accelerator) or
// family of devices, or from some mix of all of those.
//
// The type of memory a buffer is allocated within has implications on it's
// performance and lifetime. For example if an application attempts to use a
// host-allocated buffer (IREE_HAL_MEMORY_TYPE_HOST_LOCAL) on an accelerator
// with discrete memory the accelerator may either be unable to access the
// memory or take a non-trivial performance hit when attempting to do so
// (involving setting up kernel mappings, doing DMA transfers, etc). Likewise,
// trying to access a device-allocated buffer
// (IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) may incur similar overhead or not be
// possible at all. This may be due to restrictions in the memory visibility,
// address spaces, mixed endianness or pointer widths, and other weirdness.
//
// The memory types (defined by a bitfield of iree_hal_memory_type_t values)
// that a particular context (host or device) may use vary from device to device
// and must be queried by the application when allocating buffers. It's strongly
// recommended that the most specific memory type be set as possible. For
// example allocating a buffer with IREE_HAL_MEMORY_TYPE_HOST_COHERENT even when
// it will never be used in a way that requires coherency may occupy address
// space reservations or memory mapping that would otherwise not be needed.
//
// As buffers may sometimes not be accessible from the host the base Buffer type
// does not allow for direct void* access and instead buffers must be either
// manipulated using utility functions (such as ReadData or WriteData) or by
// mapping them into a host-accessible address space via MapMemory. Buffer must
// be unmapped before any command may use it.
//
// Buffers may map (roughly) 1:1 with an allocation either from the host heap or
// a device. iree_hal_buffer_Subspan can be used to reference subspans of
// buffers like absl::Span - though unlike absl::Span the returned Buffer holds
// a reference to the parent buffer.

// Buffer overlap testing results.
enum iree_hal_buffer_overlap_e {
  // No overlap between the two buffers.
  IREE_HAL_BUFFER_OVERLAP_DISJOINT = 0,
  // Partial overlap between the two buffers.
  IREE_HAL_BUFFER_OVERLAP_PARTIAL,
  // Complete overlap between the two buffers (they are the same).
  IREE_HAL_BUFFER_OVERLAP_COMPLETE,
};
typedef uint8_t iree_hal_buffer_overlap_t;

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
IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_retain(iree_hal_buffer_t* buffer);

// Releases the given |buffer| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_release(iree_hal_buffer_t* buffer);

// Returns the allocator this buffer was allocated from.
IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_buffer_allocator(const iree_hal_buffer_t* buffer);

// Returns a pointer to the buffer containing the actual allocation.
// The buffer represents a span of the allocated bytes defined by byte_offset
// and byte_length. If the provided buffer *is* the allocated buffer then the
// returned value will be the provided buffer pointer.
IREE_API_EXPORT iree_hal_buffer_t* IREE_API_CALL
iree_hal_buffer_allocated_buffer(const iree_hal_buffer_t* buffer);

// Returns the size of the resource memory allocation in bytes.
// This may be rounded up from the originally requested size or the ideal
// size for the resource based on device restrictions.
IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_allocation_size(const iree_hal_buffer_t* buffer);

// Returns the offset in bytes of the buffer within its allocated_buffer.
IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_byte_offset(const iree_hal_buffer_t* buffer);

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
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length);

// Writes a block of byte data into the buffer at the given offset.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_write_data(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length);

// Copies data from the provided |source_buffer| into the |target_buffer|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_copy_data(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t data_length);

// Maps the buffer to be accessed as a host pointer into |out_mapped_memory|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_map(
    iree_hal_buffer_t* buffer, iree_hal_memory_access_t memory_access,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_mapped_memory_t* out_mapped_memory);

// Unmaps the buffer as was previously mapped to |mapped_memory|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_unmap(
    iree_hal_buffer_t* buffer, iree_hal_mapped_memory_t* mapped_memory);

//===----------------------------------------------------------------------===//
// iree_hal_buffer_view_t
//===----------------------------------------------------------------------===//

// Creates a buffer view with the given |buffer|.
// |out_buffer_view| must be released by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view);

// Creates a buffer view referencing a subview of the given |buffer_view|.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_subview(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view);

// Retains the given |buffer_view| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_retain(iree_hal_buffer_view_t* buffer_view);

// Releases the given |buffer_view| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_buffer_view_release(iree_hal_buffer_view_t* buffer_view);

// Returns the buffer underlying the buffer view.
// The caller must retain the returned buffer if they want to continue using it.
IREE_API_EXPORT iree_hal_buffer_t* IREE_API_CALL
iree_hal_buffer_view_buffer(const iree_hal_buffer_view_t* buffer_view);

// Returns the rank of the shape associated with the buffer view.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_hal_buffer_view_shape_rank(const iree_hal_buffer_view_t* buffer_view);

// Returns the value of the given dimension.
IREE_API_EXPORT iree_hal_dim_t IREE_API_CALL iree_hal_buffer_view_shape_dim(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t index);

// Returns the dimensions of the shape in |out_shape| and its rank in
// |out_shape_rank|. |rank_capacity| indicates the number of dimensions
// available in the |out_shape| buffer. If there is not enough capacity to store
// all of the dimensions IREE_STATUS_OUT_OF_RANGE is returned.
// |out_shape_rank| can be omitted if the rank is already known.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t rank_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank);

// Returns the total number of elements stored in the view.
IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_count(const iree_hal_buffer_view_t* buffer_view);

// Returns the element type of the buffer.
IREE_API_EXPORT iree_hal_element_type_t IREE_API_CALL
iree_hal_buffer_view_element_type(const iree_hal_buffer_view_t* buffer_view);

// Returns the size of each element in the buffer view in bytes.
// Note that not all buffers are contiguous or densely packed.
IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view);

// Returns the total size of the specified view in bytes.
// Note that not all buffers are contiguous or densely packed.
IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_view_byte_length(const iree_hal_buffer_view_t* buffer_view);

// Calculates a byte offset into the |buffer_view| at the given indices.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_offset(
    const iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* indices,
    iree_host_size_t indices_count, iree_device_size_t* out_offset);

// Calculates a byte range into the |buffer_view| of the given contiguous range.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_range(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length);

// Parses a serialized set of buffer elements in the canonical tensor format
// (the same as produced by iree_hal_buffer_view_format). The underlying buffer
// will be allocated with |buffer_allocator| as a host-local/device-visible
// buffer.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_parse(
    iree_string_view_t value, iree_hal_allocator_t* buffer_allocator,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view);

// Converts buffer view elements into a fully-specified string-form format like
// `2x4xi16=[[1 2][3 4]]`.
//
// |max_element_count| can be used to limit the total number of elements printed
// when the count may be large. Elided elements will be replaced with `...`.
//
// |buffer_capacity| defines the size of |buffer| in bytes and
// |out_buffer_length| will return the string length in characters. Returns
// IREE_STATUS_OUT_OF_RANGE if the buffer capacity is insufficient to hold the
// formatted elements and |out_buffer_length| will contain the required size.
//
// Follows the standard API string formatting rules. See iree/base/api.h.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_format(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length);

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

// Asynchronous command buffer recording interface.
// Commands are recorded by the implementation for later submission to command
// queues.
//
// Buffers and synchronization objects referenced must remain valid and not be
// modified or read while there are commands in-flight. The usual flow is to
// populate input buffers, Dispatch using those buffers, wait on a Semaphore
// until the buffers are guaranteed to no longer be in use, and then reuse or
// release the buffers.
//
// Errors that can be recognized when operations are enqueued will be returned
// immediately, such as invalid argument errors. Errors that can only be
// determined at execution time will be returned on semaphores. Once a failure
// occurs the device queue will enter an error state that invalidates all
// operations on the device queue (as ordering is not strict and any may still
// be in-flight). In this case the user of the device queue should treat all
// in-flight operations as cancelled and fully reset themselves. Other device
// queues that may be waiting on events from the device queue will also enter
// error states. Only once a user has acknowledged and cleared the error state
// with a Reset the queue will become usable, and otherwise all operations will
// return errors.
//
// Command buffers are thread-compatible. Use multiple command buffers if trying
// to record commands from multiple threads. Command buffers must not be mutated
// between when they have are submitted for execution on a queue and when the
// semaphore fires indicating the completion of their execution.

// Creates a command buffer ready to begin recording, possibly reusing an
// existing one from the |device| pool.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_allocator_t allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Retains the given |command_buffer| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_command_buffer_retain(iree_hal_command_buffer_t* command_buffer);

// Releases the given |command_buffer| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_command_buffer_release(iree_hal_command_buffer_t* command_buffer);

// Resets and begins recording into the command buffer, clearing all
// previously recorded contents.
// The command buffer must not be in-flight.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_begin(iree_hal_command_buffer_t* command_buffer);

// Ends recording into the command buffer.
// This must be called prior to submitting the command buffer for execution.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_end(iree_hal_command_buffer_t* command_buffer);

// Defines a memory dependency between commands recorded before and after the
// barrier. One or more memory or buffer barriers can be specified to indicate
// between which stages or buffers the dependencies exist.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

// Fills the target buffer with the given repeating value.
// Expects that |pattern_length| is one of 1, 2, or 4 and that the offset and
// length are aligned to the natural alignment of the value.
// The target buffer must be compatible with the devices owned by this
// device queue and be allocated with IREE_HAL_BUFFER_USAGE_TRANSFER.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length);

// Updates a range of the given target buffer from the source host memory.
// The source host memory is copied immediately into the command buffer and
// occupies command buffer space. It is strongly recommended that large buffer
// updates are performed via iree_hal_command_buffer_copy_buffer where there is
// the possibility of a zero-copy path.
// The |source_buffer| may be releaed by the caller immediately after this
// call returns.
// The |target_buffer| must be compatible with the devices owned by this
// device queue and be allocated with IREE_HAL_BUFFER_USAGE_TRANSFER.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_update_buffer(iree_hal_command_buffer_t* command_buffer,
                                      const void* source_buffer,
                                      iree_host_size_t source_offset,
                                      iree_hal_buffer_t* target_buffer,
                                      iree_device_size_t target_offset,
                                      iree_device_size_t length);

// Copies a range of one buffer to another.
// Both buffers must be compatible with the devices owned by this device
// queue and be allocated with IREE_HAL_BUFFER_USAGE_TRANSFER. Though the source
// and target buffer may be the same the ranges must not overlap (as with
// memcpy).
//
// This can be used to perform device->host, host->device, and device->device
// copies.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length);

// Pushes an inline set of constants that can be accessed by subsequent
// dispatches using a compatible executable layout.
//
// Push constants are always 4-byte values and treated as opaque, meaning that
// they may be bit-casted floats, bit-packed booleans, etc.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_push_constants(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length);

// Pushes a descriptor set and associates it with |set|.
// This uses an internal ringbuffer inside of the command buffer to avoid the
// need for creating and binding descriptor sets and managing their lifetime.
//
// The descriptor set will remain bound and valid so long as the executable
// layouts used by dispatches are compatible (same descriptor layouts and push
// constant sizes).
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, int32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings);

// Binds a descriptor set to the given |set| matching that used in the
// executable layout interface.
//
// The descriptor set will remain bound and valid so long as the executable
// layouts used by dispatches are compatible (same descriptor layouts and push
// constant sizes).
//
// If any dynamic descriptor types are defined in the descriptor set layout then
// the dynamic offsets must be provided. These offsets will be added to the base
// offset of the descriptor layout binding.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, int32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets);

// Dispatches an execution request.
// The request may execute overlapped with any other transfer operation or
// dispatch made within the same barrier-defined sequence.
//
// The executable specified must be registered for use with the device driver
// owning this queue. It must not be unregistered until all requests that use
// it have completed.
//
// Fails if the queue does not support dispatch operations (as indicated by
// can_dispatch).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z);

// Dispatches an execution request with deferred workgroup counts.
// This is the same as iree_hal_command_buffer_dispatch but the workgroup counts
// are read from the given |workgroups_buffer| at offset |workgroups_offset| as
// 3 uint32_t XYZ values before performing the dispatch. This allows prior
// dispatches within the command sequence to populate the workgroup counts.
//
// The buffer must have been allocated with IREE_HAL_BUFFER_USAGE_DISPATCH and
// be of IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer, iree_device_size_t workgroups_offset);

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_t
//===----------------------------------------------------------------------===//

// Opaque handle to a descriptor set object.
//
// Maps to VkDescriptorSet:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDescriptorSet.html

// Creates a descriptor set of the given layout and bindings.
// Descriptor sets are immutable and retain their bindings.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_descriptor_set_create(
    iree_hal_device_t* device, iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_allocator_t allocator, iree_hal_descriptor_set_t** out_descriptor_set);

// Retains the given |set| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_descriptor_set_retain(iree_hal_descriptor_set_t* descriptor_set);

// Releases the given |set| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_descriptor_set_release(iree_hal_descriptor_set_t* descriptor_set);

//===----------------------------------------------------------------------===//
// iree_hal_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

// Opaque handle to a descriptor set layout object.
//
// Maps to VkDescriptorSetLayout:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkDescriptorSetLayout.html

// Creates a descriptor set layout with the given bindings.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_descriptor_set_layout_create(
    iree_hal_device_t* device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Retains the given |descriptor_set_layout| for the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_descriptor_set_layout_retain(
    iree_hal_descriptor_set_layout_t* descriptor_set_layout);

// Releases the given |descriptor_set_layout| from the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_descriptor_set_layout_release(
    iree_hal_descriptor_set_layout_t* descriptor_set_layout);

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

// Retains the given |device| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_device_retain(iree_hal_device_t* device);

// Releases the given |device| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_device_release(iree_hal_device_t* device);

// Returns the device identifier.
// This identifier may vary based on the runtime device type; for example, a
// Vulkan device may return `vulkan-v1.1` or `vulkan-v1.2-spec1`.
IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_hal_device_id(iree_hal_device_t* device);

// Returns a reference to the allocator of the device that can be used for
// allocating buffers.
IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_device_allocator(iree_hal_device_t* device);

// Submits one or more batches of work to a device queue.
//
// The queue is selected based on the flags set in |command_categories| and the
// |queue_affinity|. As the number of available queues can vary the
// |queue_affinity| is used to hash into the available queues for the required
// categories. For example if 2 queues support transfer commands and the
// affinity is 5 the resulting queue could be index hash(5)=1. The affinity can
// thus be treated as just a way to indicate whether two submissions must be
// placed on to the same queue. Note that the exact hashing function is
// implementation dependent.
//
// The submission behavior matches Vulkan's vkQueueSubmit, with each batch
// executing its command buffers in the order they are defined but allowing the
// command buffers to complete out-of-order. See:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit.html
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_device_queue_submit(
    iree_hal_device_t* device, iree_hal_command_category_t command_categories,
    uint64_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches);

// Blocks the caller until the semaphores reach or exceed the specified payload
// values or the |deadline_ns| elapses. All semaphores in |semaphore_list| must
// be created from this device (or be imported into it).
//
// |wait_mode| can be used to decide when the wait will proceed; whether *all*
// semaphores in |semaphore_list| must be signaled or whether *any* (one or
// more) can be signaled before an early return.
//
// Returns success if the wait is successful and semaphores have been signaled
// satisfying the |wait_mode|.
//
// Returns DEADLINE_EXCEEDED if the |deadline_ns| elapses without the
// |wait_mode| being satisfied. Note that even on success only a subset of the
// semaphores may have been signaled and each can be queried to see which ones.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_device_wait_semaphores_with_deadline(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns);

// Blocks the caller until the semaphores reach or exceed the specified payload
// values or the |timeout_ns| elapses.
// A relative-time version of iree_hal_device_wait_semaphores_with_deadline
// using the relative nanoseconds from the time the call is made.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_device_wait_semaphores_with_timeout(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list,
    iree_duration_t timeout_ns);

//===----------------------------------------------------------------------===//
// iree_hal_driver_t
//===----------------------------------------------------------------------===//

// Retains the given |driver| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_driver_retain(iree_hal_driver_t* driver);

// Releases the given |driver| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_driver_release(iree_hal_driver_t* driver);

// Queries available devices and returns them as a list.
// The provided |allocator| will be used to allocate the returned list and after
// the caller is done with it |out_device_infos| must be freed with that same
// allocator by the caller.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_query_available_devices(
    iree_hal_driver_t* driver, iree_allocator_t allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count);

// Creates a device as queried with iree_hal_driver_query_available_devices.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_driver_create_device(
    iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
    iree_allocator_t allocator, iree_hal_device_t** out_device);

// Creates the driver-defined "default" device. This may simply be the first
// device enumerated.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_create_default_device(iree_hal_driver_t* driver,
                                      iree_allocator_t allocator,
                                      iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_driver_registry_t
//===----------------------------------------------------------------------===//

// Factory interface used for driver enumeration and creation.
// The factory is designed to in many cases live in rodata by not requiring any
// real code or processing when the driver is statically known to be available.
// When drivers may be dynamically available based on system configuration a
// factory can discover them and provide them during enumeration.
//
// Delay-loaded drivers that may require non-trivial setup time (such as those
// implemented in dynamic libraries or over RPC) can be speculatively enumerated
// by a factory and then rely on the try_create to actually perform the slow
// work once the user has explicitly signaled that they are willing to pay the
// cost (and deal with the consequences).
//
// WARNING: this API is unstable until the HAL is fully ported. Do not use.
typedef struct {
  // TODO(benvanik): version field.
  IREE_API_UNSTABLE

  // User-defined pointer passed to all functions.
  void* self;

  // Queries the list of available drivers provided by the factory, if any.
  // |out_driver_infos| will be populated with a *reference* to factory data
  // structures (such as the driver name) that callers may choose to clone if
  // needed.
  //
  // Implementers must make their factory enumeration results immutable for the
  // duration they are registered, though the behavior of try_create is allowed
  // to change call-to-call. If a factory needs to mutate its set of enumerated
  // devices then it must do so by first unregistering itself and re-registering
  // only after the changes have been made.
  //
  // Called with the driver registry lock held; may be called from any thread.
  iree_status_t(IREE_API_PTR* enumerate)(
      void* self, const iree_hal_driver_info_t** out_driver_infos,
      iree_host_size_t* out_driver_info_count);

  // Tries to create a driver as previously queried with enumerate.
  // |driver_id| is the opaque ID returned from enumeration; note that there may
  // be a significant amount of time between enumeration and creation and the
  // driver registry lock may have been release between then.
  //
  // Delay-loaded drivers may still fail here if - for example - required system
  // resources are unavailable or permission is denied.
  //
  // Called with the driver registry lock held; may be called from any thread.
  iree_status_t(IREE_API_PTR* try_create)(void* self,
                                          iree_hal_driver_id_t driver_id,
                                          iree_allocator_t allocator,
                                          iree_hal_driver_t** out_driver);
} iree_hal_driver_factory_t;

// Returns the default per-process driver registry.
// In simple applications this is usually where you want to go to register and
// create drivers. More sophisticated applications that want tighter control
// over the visibility of drivers to certain callers such as when dealing with
// requests from multiple users may choose to allocate their own registries and
// manage their lifetime as desired.
//
// TODO(benvanik): remove global registry and make callers manage always. We can
// provide helpers to make that easier to do, but there's really no benefit to
// having this be global like it is. Alternatively, this can be opt-in thanks to
// LTO: if a user doesn't call this then the default registry is never
// allocated.
IREE_API_EXPORT iree_hal_driver_registry_t* IREE_API_CALL
iree_hal_driver_registry_default();

// Registers a driver factory to serve future queries/requests for drivers.
// See iree_hal_driver_registry_t for more information.
//
// Thread-safe. The factory is not retained and must be kept alive by the caller
// until it is unregistered (or the application terminates).
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_registry_register_factory(
    iree_hal_driver_registry_t* registry,
    const iree_hal_driver_factory_t* factory);

// Unregisters a driver factory.
// Unregistering a factory only prevents new drivers from being created;
// existing drivers may remain live even after unregistering. Factories can
// expect that no new drivers will be created via the factory after the call
// returns.
//
// Thread-safe. As the factory is not retained by the registry the caller must
// release its memory (if needed) after this call returns.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_registry_unregister_factory(
    iree_hal_driver_registry_t* registry,
    const iree_hal_driver_factory_t* factory);

// Enumerates all drivers from registered factories and returns them as a list.
// The provided |allocator| will be used to allocate the returned list and after
// the caller is done with it |out_driver_infos| must be freed with that same
// allocator by the caller.
//
// The set of drivers returned should be considered the superset of those that
// may be available for successful creation as it's possible that delay-loaded
// drivers may fail even if they appear in this list.
//
// Thread-safe. Note that the factory may be unregistered between the query
// completing and any attempt to instantiate the driver.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_driver_registry_enumerate(
    iree_hal_driver_registry_t* registry, iree_allocator_t allocator,
    iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count);

// Attempts to create a driver registered with the driver registry by a specific
// ID as returned during enumeration in iree_hal_driver_info_t::driver_id.
// This can be used to specify the exact driver to create in cases where there
// may be multiple factories providing drivers with the same name.
//
// Thread-safe. May block the caller if the driver is delay-loaded and needs to
// perform additional loading/verification/etc before returning.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_driver_registry_try_create(
    iree_hal_driver_registry_t* registry, iree_hal_driver_id_t driver_id,
    iree_allocator_t allocator, iree_hal_driver_t** out_driver);

// Attempts to create a driver registered with the given canonical driver name.
// Effectively enumerate + find by name + try_create if found. Factories are
// searched in most-recently-added order such that it's possible to override
// drivers with newer registrations when multiple factories provide the same
// driver name.
//
// Thread-safe. May block the caller if the driver is delay-loaded and needs to
// perform additional loading/verification/etc before returning.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_registry_try_create_by_name(
    iree_hal_driver_registry_t* registry, iree_string_view_t driver_name,
    iree_allocator_t allocator, iree_hal_driver_t** out_driver);

//===----------------------------------------------------------------------===//
// iree_hal_event_t
//===----------------------------------------------------------------------===//

// Events are used for defining synchronization scopes within command buffers.
// An event only exists within a single CommandBuffer and must not be used
// across command buffers from the same device or others.
//
// See iree_hal_command_buffer_signal_event and
// iree_hal_command_buffer_wait_events for more info.
//
// Maps to VkEvent:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkEvent.html

//===----------------------------------------------------------------------===//
// iree_hal_executable_t
//===----------------------------------------------------------------------===//

// Retains the given |executable| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_executable_retain(iree_hal_executable_t* executable);

// Releases the given |executable| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_executable_release(iree_hal_executable_t* executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_cache_t
//===----------------------------------------------------------------------===//

// A cache of prepared executables for a particular device.
// Caches may be shared across multiple devices from the same driver or specific
// to individual devices. Caches may persist prepared executables across process
// launches or re-prepare them each run. Callers should assume that the cache is
// a no-op and the returned Executables only live for as long as the cache does.
//
// The term 'cache' here is rather optimistic - it's perfectly acceptable for
// implementations to not cache at all and return new Executables for each
// iree_hal_executable_cache_prepare_executable called (even for the same
// executable). Callers should expect such behavior and try to retain the
// results of the iree_hal_executable_cache_prepare_executable calls to reduce
// overhead in re-preparing executables.
//
// Thread-safe - multiple threads may prepare executables (including the *same*
// executable) simultaneously.

// Creates an executable cache using the given identifier.
// The identifier is provided to the backing cache API as way to partition
// caches between different groups of executables (from different modules, etc).
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_executable_cache_create(
    iree_hal_device_t* device, iree_string_view_t identifier,
    iree_allocator_t allocator,
    iree_hal_executable_cache_t** out_executable_cache);

// Retains the given |executable_cache| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_executable_cache_retain(iree_hal_executable_cache_t* executable_cache);

// Releases the given |executable_cache| from the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_cache_release(
    iree_hal_executable_cache_t* executable_cache);

// Returns true if the executable cache can prepare the given executable input
// format. Preparation may still fail if the particular version or features
// required by the executable are not supported.
IREE_API_EXPORT bool IREE_API_CALL iree_hal_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* executable_cache,
    iree_hal_executable_format_t format);

// Prepares an executable for use.
// The provided |executable_data| will be used to either lookup a previously
// prepared executable in the cache or prepare a new one.
//
// Depending on the driver preparation may take a non-trivial amount of time
// (such as when JITing/etc). As the cache is internally synchronized callers
// can issue preparation requests from multiple threads - even for the same
// executables - and calls will block until preparation completes.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* executable_cache,
    iree_hal_executable_layout_t* executable_layout,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data, iree_allocator_t allocator,
    iree_hal_executable_t** out_executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_layout_t
//===----------------------------------------------------------------------===//

// Defines the resource binding layout used by an executable.
//
// Executables can share the same layout even if they do not use all of the
// resources referenced by descriptor sets referenced by the layout. Doing so
// allows for more efficient binding as bound descriptor sets can be reused when
// command buffer executable bindings change.
//
// Maps to VkPipelineLayout:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPipelineLayout.html

// Creates an executable layout composed of the given descriptor set layouts.
// The returned executable layout can be used by multiple executables with the
// same compatible resource binding layouts.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_executable_layout_create(
    iree_hal_device_t* device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constants, iree_allocator_t allocator,
    iree_hal_executable_layout_t** out_executable_layout);

// Retains the given |executable_layout| for the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_layout_retain(
    iree_hal_executable_layout_t* executable_layout);

// Releases the given |executable_layout| from the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_layout_release(
    iree_hal_executable_layout_t* executable_layout);

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

// Synchronization mechanism for host->device, device->host, host->host,
// and device->device notification. Semaphores behave like Vulkan timeline
// semaphores (or D3D12 fences) and contain a monotonically increasing
// uint64_t payload. They may be waited on any number of times even if they
// have already been signaled for a particular value. They may also be waited
// on for a particular value prior to the signal for that value.
//
// A semaphore is updated to its new value after all prior commands have
// completed but the delay between completion and the host being woken varies.
// Some implementations may coalesce semaphores to avoid spurious waking while
// others will immediately synchronize with the host.
//
// One use of semaphores is for resource lifetime management: all resources used
// by a set of submission batches must be considered live until the semaphore
// attached to the submission has signaled.
//
// Another use of semaphores is device->device synchronization for setting up
// the DAG of command buffers across queue submissions. This allows devices to
// perform non-trivial scheduling behavior without the need to wake the host.
//
// Semaphores may be set to a permanently failed state by implementations when
// errors occur during asynchronous execution. Users are expected to propagate
// the failures and possibly reset the entire device that produced the error.
//
// For more information on semaphores see the following docs describing how
// timelines are generally used (specifically in the device->host case):
// https://www.youtube.com/watch?v=SpE--Rf516Y
// https://www.khronos.org/assets/uploads/developers/library/2018-xdc/Vulkan-Timeline-Semaphores-Part-1_Sep18.pdf
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization

// Creates a semaphore that can be used with command queues owned by this
// device. To use the semaphores with other devices or instances they must
// first be exported.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_semaphore_create(
    iree_hal_device_t* device, uint64_t initial_value,
    iree_allocator_t allocator, iree_hal_semaphore_t** out_semaphore);

// Retains the given |semaphore| for the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_retain(iree_hal_semaphore_t* semaphore);

// Releases the given |semaphore| from the caller.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_release(iree_hal_semaphore_t* semaphore);

// Queries the current payload of the semaphore and stores the result in
// |out_value|. As the payload is monotonically increasing it is guaranteed that
// the value is at least equal to the previous result of a
// iree_hal_semaphore_query call and coherent with any waits for a
// specified value via iree_device_wait_all_semaphores.
//
// Returns the status at the time the method is called without blocking and as
// such is only valid after a semaphore has been signaled. The same failure
// status will be returned regardless of when in the timeline the error
// occurred.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_query(iree_hal_semaphore_t* semaphore, uint64_t* out_value);

// Signals the |semaphore| to the given payload value.
// The call is ignored if the current payload value exceeds |new_value|.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_signal(iree_hal_semaphore_t* semaphore, uint64_t new_value);

// Signals the |semaphore| with a failure. The |status| will be returned from
// iree_hal_semaphore_query and iree_hal_semaphore_signal for the lifetime
// of the semaphore.
IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_fail(iree_hal_semaphore_t* semaphore, iree_status_t status);

// Blocks the caller until the semaphore reaches or exceedes the specified
// payload value or the |deadline_ns| elapses.
//
// Returns success if the wait is successful and the semaphore has met or
// exceeded the required payload value.
//
// Returns DEADLINE_EXCEEDED if the |deadline_ns| elapses without the semaphore
// reaching the required value. If an asynchronous failure occured this will
// return the failure status that was set immediately.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_wait_with_deadline(iree_hal_semaphore_t* semaphore,
                                      uint64_t value, iree_time_t deadline_ns);

// Blocks the caller until the semaphore reaches or exceedes the specified
// payload value or the |timeout_ns| elapses.
// A relative-time version of iree_hal_semaphore_wait_with_deadline using the
// relative nanoseconds from the time the call is made.
IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_wait_with_timeout(iree_hal_semaphore_t* semaphore,
                                     uint64_t value,
                                     iree_duration_t timeout_ns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_API_H_
