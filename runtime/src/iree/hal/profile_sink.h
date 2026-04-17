// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_PROFILE_SINK_H_
#define IREE_HAL_PROFILE_SINK_H_

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_profile_sink_t
//===----------------------------------------------------------------------===//

// Retained interface receiving profiling chunks from HAL implementations.
//
// Profile sinks are the programmatic transport for HAL-native profiling data.
// They may write chunks to files, forward them to live tools, copy them into
// application-owned telemetry buffers, or discard selected streams. Drivers
// should never call sinks from queue submission hot paths or while holding
// queue locks; sinks are allowed to block and allocate unless a specific sink
// implementation documents a stricter contract.
typedef struct iree_hal_profile_sink_t iree_hal_profile_sink_t;

// Bitfield specifying properties of a profiling chunk.
typedef uint64_t iree_hal_profile_chunk_flags_t;
enum iree_hal_profile_chunk_flag_bits_t {
  IREE_HAL_PROFILE_CHUNK_FLAG_NONE = 0u,

  // Chunk contents are partial because the producer dropped or truncated data.
  // The chunk is still structurally valid, but consumers must not treat it as
  // a complete representation of the selected range.
  IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED = 1ull << 0,
};

// Content type for a profiling session boundary metadata chunk.
#define IREE_HAL_PROFILE_CONTENT_TYPE_SESSION \
  IREE_SV("application/vnd.iree.hal.profile.session")

// Content type for an array of iree_hal_profile_device_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES \
  IREE_SV("application/vnd.iree.hal.profile.devices")

// Content type for an array of iree_hal_profile_queue_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES \
  IREE_SV("application/vnd.iree.hal.profile.queues")

// Content type for an array of iree_hal_profile_executable_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES \
  IREE_SV("application/vnd.iree.hal.profile.executables")

// Content type for packed iree_hal_profile_executable_export_record_t records.
#define IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS \
  IREE_SV("application/vnd.iree.hal.profile.executable-exports")

// Content type for an array of iree_hal_profile_command_buffer_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS \
  IREE_SV("application/vnd.iree.hal.profile.command-buffers")

// Content type for an array of iree_hal_profile_command_operation_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS \
  IREE_SV("application/vnd.iree.hal.profile.command-operations")

// Content type for an array of iree_hal_profile_clock_correlation_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS \
  IREE_SV("application/vnd.iree.hal.profile.clock-correlations")

// Content type for an array of iree_hal_profile_dispatch_event_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS \
  IREE_SV("application/vnd.iree.hal.profile.dispatch-events")

// Content type for an array of iree_hal_profile_queue_event_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS \
  IREE_SV("application/vnd.iree.hal.profile.queue-events")

// Content type for an array of iree_hal_profile_memory_event_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS \
  IREE_SV("application/vnd.iree.hal.profile.memory-events")

// Bitfield specifying which optional device record fields are populated.
typedef uint32_t iree_hal_profile_device_flags_t;
enum iree_hal_profile_device_flag_bits_t {
  IREE_HAL_PROFILE_DEVICE_FLAG_NONE = 0u,

  // |physical_device_uuid| contains a stable physical device identifier.
  IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID = 1u << 0,
};

// Session-level physical device description.
//
// Producers should emit device records before event records that reference the
// corresponding |physical_device_ordinal|. The ordinal is a compact
// session-local key; consumers that need hermetic identity should use
// |physical_device_uuid| when present.
typedef struct iree_hal_profile_device_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags specifying which optional fields are populated.
  iree_hal_profile_device_flags_t flags;
  // Session-local physical device ordinal used by compact event records.
  uint32_t physical_device_ordinal;
  // Number of queues described for this physical device.
  uint32_t queue_count;
  // Stable physical device UUID when
  // IREE_HAL_PROFILE_DEVICE_FLAG_PHYSICAL_DEVICE_UUID is set.
  uint8_t physical_device_uuid[16];
} iree_hal_profile_device_record_t;

// Returns a default physical device record.
static inline iree_hal_profile_device_record_t
iree_hal_profile_device_record_default(void) {
  iree_hal_profile_device_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  return record;
}

// Session-level queue description.
//
// Producers should emit queue records before event records that reference the
// corresponding |stream_id| or |queue_ordinal|. Queue ordinals are only scoped
// to their physical device within a profiling session.
typedef struct iree_hal_profile_queue_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Session-local physical device ordinal owning this queue.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal within |physical_device_ordinal|.
  uint32_t queue_ordinal;
  // Reserved for future queue record flags; must be zero.
  uint32_t reserved0;
  // Producer-defined stream identifier used in chunk/event metadata.
  uint64_t stream_id;
} iree_hal_profile_queue_record_t;

// Returns a default queue record.
static inline iree_hal_profile_queue_record_t
iree_hal_profile_queue_record_default(void) {
  iree_hal_profile_queue_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying which executable record fields are populated.
typedef uint32_t iree_hal_profile_executable_flags_t;
enum iree_hal_profile_executable_flag_bits_t {
  IREE_HAL_PROFILE_EXECUTABLE_FLAG_NONE = 0u,
};

// Session-level executable description.
//
// Producers should emit executable records before dispatch event records that
// reference |executable_id|. The id is a compact producer-local key; consumers
// should use code-object hashes when present for cross-process correlation.
typedef struct iree_hal_profile_executable_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags specifying which optional executable fields are populated.
  iree_hal_profile_executable_flags_t flags;
  // Producer-local executable identifier referenced by dispatch events.
  uint64_t executable_id;
  // Number of export records associated with this executable.
  uint32_t export_count;
  // Reserved for future executable record fields; must be zero.
  uint32_t reserved0;
  // Strong code-object hash words when a future flag marks them populated.
  uint64_t code_object_hash[2];
} iree_hal_profile_executable_record_t;

// Returns a default executable record.
static inline iree_hal_profile_executable_record_t
iree_hal_profile_executable_record_default(void) {
  iree_hal_profile_executable_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  return record;
}

// Bitfield specifying which executable export fields are populated.
typedef uint32_t iree_hal_profile_executable_export_flags_t;
enum iree_hal_profile_executable_export_flag_bits_t {
  IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_NONE = 0u,
};

// Session-level executable export description followed by |name_length| bytes.
//
// Producers should emit export records before dispatch event records that
// reference the pair of |executable_id| and |export_ordinal|. The trailing name
// is not NUL-terminated.
typedef struct iree_hal_profile_executable_export_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags specifying which optional export fields are populated.
  iree_hal_profile_executable_export_flags_t flags;
  // Producer-local executable identifier owning this export.
  uint64_t executable_id;
  // Export ordinal used by dispatch events.
  uint32_t export_ordinal;
  // Number of constant words expected by the HAL ABI export.
  uint32_t constant_count;
  // Number of binding pointer slots expected by the HAL ABI export.
  uint32_t binding_count;
  // Number of reflected export parameters.
  uint32_t parameter_count;
  // Static workgroup size for each dimension, or the minimum dynamic size.
  uint32_t workgroup_size[3];
  // Byte length of the trailing export name.
  uint32_t name_length;
} iree_hal_profile_executable_export_record_t;

// Returns a default executable export record.
static inline iree_hal_profile_executable_export_record_t
iree_hal_profile_executable_export_record_default(void) {
  iree_hal_profile_executable_export_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.export_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying which command-buffer record fields are populated.
typedef uint32_t iree_hal_profile_command_buffer_flags_t;
enum iree_hal_profile_command_buffer_flag_bits_t {
  IREE_HAL_PROFILE_COMMAND_BUFFER_FLAG_NONE = 0u,
};

// Session-level reusable command-buffer description.
//
// Producers should emit command-buffer records before dispatch event records
// that reference |command_buffer_id|.
typedef struct iree_hal_profile_command_buffer_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags specifying which optional command-buffer fields are populated.
  iree_hal_profile_command_buffer_flags_t flags;
  // Producer-local command-buffer identifier referenced by dispatch events.
  uint64_t command_buffer_id;
  // HAL command-buffer mode bits used to create the command buffer.
  uint64_t mode;
  // HAL command categories supported by the command buffer.
  uint64_t command_categories;
  // Queue affinity normalized at command-buffer creation.
  uint64_t queue_affinity;
  // Physical device ordinal selected for recorded device-specific packets.
  uint32_t physical_device_ordinal;
  // Reserved for future command-buffer record fields; must be zero.
  uint32_t reserved0;
} iree_hal_profile_command_buffer_record_t;

// Returns a default command-buffer record.
static inline iree_hal_profile_command_buffer_record_t
iree_hal_profile_command_buffer_record_default(void) {
  iree_hal_profile_command_buffer_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  return record;
}

// Type of command-buffer operation recorded by a HAL producer.
typedef uint32_t iree_hal_profile_command_operation_type_t;
enum iree_hal_profile_command_operation_type_e {
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_NONE = 0u,

  // Execution or memory visibility barrier.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER = 1u,

  // Kernel dispatch operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH = 2u,

  // Device buffer fill operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL = 3u,

  // Device buffer copy operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY = 4u,

  // Host-to-device buffer update operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE = 5u,

  // Profiling or debug marker operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_PROFILE_MARKER = 6u,

  // Unconditional command-buffer branch operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH = 7u,

  // Conditional command-buffer branch operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH = 8u,

  // Command-buffer return operation.
  IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN = 9u,
};

// Bitfield specifying properties of one command-buffer operation record.
typedef uint32_t iree_hal_profile_command_operation_flags_t;
enum iree_hal_profile_command_operation_flag_bits_t {
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_NONE = 0u,

  // Operation represents an execution or memory visibility barrier.
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER = 1u << 0,

  // Operation reads dynamic dispatch parameters from device memory.
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_INDIRECT_PARAMETERS = 1u << 1,

  // Operation uses at least one dynamic binding-table slot.
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS = 1u << 2,

  // Operation uses at least one statically recorded buffer reference.
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS = 1u << 3,

  // Operation has immutable prepublished dispatch arguments.
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_PREPUBLISHED_ARGUMENTS = 1u << 4,

  // Operation changes command-buffer control flow.
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_CONTROL_FLOW = 1u << 5,
};

// Session-level reusable command-buffer operation description.
//
// Producers should emit command-operation records after the command-buffer
// record defining |command_buffer_id| and before dispatch event records that
// reference |command_buffer_id| and |command_index|. The fields are intended to
// be compact enough for one record per recorded command and generic enough for
// non-AMDGPU producers to populate from their own command encodings.
typedef struct iree_hal_profile_command_operation_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of command-buffer operation represented by this record.
  iree_hal_profile_command_operation_type_t type;
  // Flags describing operation properties.
  iree_hal_profile_command_operation_flags_t flags;
  // Command ordinal within |command_buffer_id|.
  uint32_t command_index;
  // Process-local command-buffer identifier.
  uint64_t command_buffer_id;
  // Producer-local command block ordinal containing this operation.
  uint32_t block_ordinal;
  // Command ordinal within |block_ordinal|.
  uint32_t block_command_ordinal;
  // Process-local executable identifier, or 0 when not applicable.
  uint64_t executable_id;
  // Executable export ordinal, or UINT32_MAX when not applicable.
  uint32_t export_ordinal;
  // Number of binding slots used by the operation, or 0 when not applicable.
  uint32_t binding_count;
  // Static workgroup counts for dispatch operations, or zero when dynamic.
  uint32_t workgroup_count[3];
  // Workgroup sizes for dispatch operations, or zero when not applicable.
  uint32_t workgroup_size[3];
  // Source byte offset for transfer operations, or 0 when not applicable.
  uint64_t source_offset;
  // Target byte offset for transfer operations, or 0 when not applicable.
  uint64_t target_offset;
  // Byte length for transfer operations, or 0 when not applicable.
  uint64_t length;
  // Producer-defined source binding ordinal, or UINT32_MAX when absent.
  uint32_t source_ordinal;
  // Producer-defined target binding ordinal, or UINT32_MAX when absent.
  uint32_t target_ordinal;
  // Primary branch target block ordinal, or UINT32_MAX when not applicable.
  uint32_t target_block_ordinal;
  // Alternate branch target block ordinal, or UINT32_MAX when absent.
  uint32_t alternate_block_ordinal;
  // Reserved for future command-operation record fields; must be zero.
  uint32_t reserved0;
} iree_hal_profile_command_operation_record_t;

// Returns a default command-buffer operation record.
static inline iree_hal_profile_command_operation_record_t
iree_hal_profile_command_operation_record_default(void) {
  iree_hal_profile_command_operation_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.command_index = UINT32_MAX;
  record.block_ordinal = UINT32_MAX;
  record.block_command_ordinal = UINT32_MAX;
  record.export_ordinal = UINT32_MAX;
  record.source_ordinal = UINT32_MAX;
  record.target_ordinal = UINT32_MAX;
  record.target_block_ordinal = UINT32_MAX;
  record.alternate_block_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying which clock correlation fields are populated.
typedef uint32_t iree_hal_profile_clock_correlation_flags_t;
enum iree_hal_profile_clock_correlation_flag_bits_t {
  IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_NONE = 0u,

  // |device_tick| contains a timestamp in the physical device's clock domain.
  IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK = 1u << 0,

  // |host_cpu_timestamp_ns| contains a host CPU counter sampled by the driver.
  IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP = 1u << 1,

  // |host_system_timestamp| and |host_system_frequency_hz| are populated.
  IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_SYSTEM_TIMESTAMP = 1u << 2,

  // |host_time_begin_ns| and |host_time_end_ns| bracket the driver sample.
  IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET = 1u << 3,
};

// Correlates a physical device clock-domain tick with host clock domains.
//
// Producers should emit at least two records per physical device in a profiling
// session when device-timestamped event records are present. Consumers should
// treat the host bracket as uncertainty around the driver-provided sample and
// should not assume one sample is sufficient to determine clock drift.
typedef struct iree_hal_profile_clock_correlation_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags specifying which optional fields are populated.
  iree_hal_profile_clock_correlation_flags_t flags;
  // Session-local physical device ordinal owning |device_tick|.
  uint32_t physical_device_ordinal;
  // Reserved for future clock correlation record fields; must be zero.
  uint32_t reserved0;
  // Producer-defined sample identifier unique within the profiling session.
  uint64_t sample_id;
  // Device timestamp in the same domain as dispatch event start/end ticks.
  uint64_t device_tick;
  // Driver-sampled host CPU timestamp in nanoseconds, when available. This may
  // be a different clock domain than the IREE monotonic host time bracket.
  uint64_t host_cpu_timestamp_ns;
  // Driver-sampled host system timestamp in |host_system_frequency_hz| units.
  uint64_t host_system_timestamp;
  // Frequency in Hz for |host_system_timestamp|.
  uint64_t host_system_frequency_hz;
  // IREE monotonic host timestamp immediately before the driver sample.
  int64_t host_time_begin_ns;
  // IREE monotonic host timestamp immediately after the driver sample.
  int64_t host_time_end_ns;
} iree_hal_profile_clock_correlation_record_t;

// Returns a default clock correlation record.
static inline iree_hal_profile_clock_correlation_record_t
iree_hal_profile_clock_correlation_record_default(void) {
  iree_hal_profile_clock_correlation_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying properties of one dispatch event record.
typedef uint32_t iree_hal_profile_dispatch_event_flags_t;
enum iree_hal_profile_dispatch_event_flag_bits_t {
  IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_NONE = 0u,

  // Dispatch was enqueued through a reusable command buffer.
  IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER = 1u << 0,

  // Workgroup counts were loaded from device memory before dispatch.
  IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS = 1u << 1,
};

// Device-timestamped dispatch execution event.
//
// Producers emit dispatch events after the device and queue metadata chunks
// that define the chunk's physical_device_ordinal, queue_ordinal, and
// stream_id. Times are raw device ticks in the producer's clock domain.
typedef struct iree_hal_profile_dispatch_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags describing how the dispatch was produced.
  iree_hal_profile_dispatch_event_flags_t flags;
  // Producer-defined event identifier unique within the chunk stream.
  uint64_t event_id;
  // Queue submission epoch containing this dispatch.
  uint64_t submission_id;
  // Process-local command-buffer identifier, or 0 for direct queue dispatch.
  uint64_t command_buffer_id;
  // Process-local executable identifier, or 0 when unavailable.
  uint64_t executable_id;
  // Command ordinal within a command buffer, or UINT32_MAX for direct dispatch.
  uint32_t command_index;
  // Executable export ordinal dispatched.
  uint32_t export_ordinal;
  // Workgroup counts submitted for each dimension.
  uint32_t workgroup_count[3];
  // Workgroup sizes submitted for each dimension.
  uint32_t workgroup_size[3];
  // Device timestamp captured when dispatch execution started.
  uint64_t start_tick;
  // Device timestamp captured when dispatch execution completed.
  uint64_t end_tick;
} iree_hal_profile_dispatch_event_t;

// Returns a default dispatch event record.
static inline iree_hal_profile_dispatch_event_t
iree_hal_profile_dispatch_event_default(void) {
  iree_hal_profile_dispatch_event_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.command_index = UINT32_MAX;
  record.export_ordinal = UINT32_MAX;
  return record;
}

// Type of queue operation recorded by a HAL producer.
typedef uint32_t iree_hal_profile_queue_event_type_t;
enum iree_hal_profile_queue_event_type_e {
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE = 0u,

  // Queue submission containing no payload beyond dependency/order effects.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER = 1u,

  // Direct dispatch submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH = 2u,

  // Reusable command buffer execution submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE = 3u,

  // Device buffer copy submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY = 4u,

  // Device buffer fill submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL = 5u,

  // Host-to-device buffer update submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE = 6u,

  // HAL file read submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ = 7u,

  // HAL file write submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE = 8u,

  // Async allocation submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA = 9u,

  // Async deallocation submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA = 10u,

  // User-visible host callback submitted through a queue operation.
  IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL = 11u,
};

// Bitfield specifying properties of one queue event record.
typedef uint32_t iree_hal_profile_queue_event_flags_t;
enum iree_hal_profile_queue_event_flag_bits_t {
  IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_NONE = 0u,

  // Queue operation was issued from software-deferred pending state.
  IREE_HAL_PROFILE_QUEUE_EVENT_FLAG_SOFTWARE_DEFERRED = 1u << 0,
};

// Strategy used to satisfy queue operation wait dependencies.
typedef uint32_t iree_hal_profile_queue_dependency_strategy_t;
enum iree_hal_profile_queue_dependency_strategy_e {
  IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE = 0u,

  // Wait dependencies were already satisfied or represented by the payload's
  // own packet ordering, without a dedicated wait packet or host deferral.
  IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE = 1u,

  // Wait dependencies required one or more device-side barrier/wait packets.
  IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER = 2u,

  // Wait dependencies or temporary resources required host-side deferral.
  IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER = 3u,
};

// Host-timestamped queue operation event.
//
// Queue events describe submission-time behavior: which queue accepted an
// operation, how dependency edges were represented, what user-visible
// submission id was assigned, and how much payload the operation covered. The
// timestamp is in IREE host monotonic time, not a device clock domain.
typedef struct iree_hal_profile_queue_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of queue operation represented by this record.
  iree_hal_profile_queue_event_type_t type;
  // Flags describing queue operation properties.
  iree_hal_profile_queue_event_flags_t flags;
  // Strategy used for wait dependencies on this operation.
  iree_hal_profile_queue_dependency_strategy_t dependency_strategy;
  // Producer-defined event identifier unique within the queue event stream.
  uint64_t event_id;
  // IREE monotonic host timestamp in nanoseconds.
  int64_t host_time_ns;
  // Queue submission epoch associated with this operation, or 0 when absent.
  uint64_t submission_id;
  // Process-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;
  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;
  // Producer-defined stream identifier matching the queue metadata record.
  uint64_t stream_id;
  // Session-local physical device ordinal associated with this operation.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this operation.
  uint32_t queue_ordinal;
  // Number of wait semaphores supplied to the queue operation.
  uint32_t wait_count;
  // Number of signal semaphores supplied to the queue operation.
  uint32_t signal_count;
  // Number of dedicated dependency barrier packets emitted for this operation.
  uint32_t barrier_count;
  // Number of encoded payload operations represented by this queue operation.
  uint32_t operation_count;
  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;
  // Reserved for future queue event fields; must be zero.
  uint64_t reserved0;
} iree_hal_profile_queue_event_t;

// Returns a default queue operation event record.
static inline iree_hal_profile_queue_event_t
iree_hal_profile_queue_event_default(void) {
  iree_hal_profile_queue_event_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  return record;
}

// Type of memory lifecycle event recorded by a HAL producer.
typedef uint32_t iree_hal_profile_memory_event_type_t;
enum iree_hal_profile_memory_event_type_e {
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_NONE = 0u,

  // Slab-provider backing allocation acquired from the platform allocator.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE = 1u,

  // Slab-provider backing allocation released to the platform allocator.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE = 2u,

  // Pool reservation attempt for a logical allocation.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE = 3u,

  // Pool reservation materialized into a HAL buffer.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE = 4u,

  // Pool reservation released for reuse after its death frontier.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE = 5u,

  // Pool reservation could not proceed until a memory-readiness condition.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT = 6u,

  // Queue alloca operation published to a HAL queue.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA = 7u,

  // Queue dealloca operation published to a HAL queue.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA = 8u,
};

// Bitfield specifying properties of one memory lifecycle event.
typedef uint32_t iree_hal_profile_memory_event_flags_t;
enum iree_hal_profile_memory_event_flag_bits_t {
  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_NONE = 0u,

  // Event is associated with a HAL queue operation.
  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION = 1u << 0,

  // Event waited on a frontier supplied by the allocation pool.
  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER = 1u << 1,

  // Event waited on a pool notification instead of a device frontier.
  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION = 1u << 2,

  // Event references an actual pool reservation.
  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION = 1u << 3,
};

// Host-timestamped memory lifecycle event.
//
// Memory events describe allocations, reservations, and queue-visible
// allocation operations. The timestamp is in IREE host monotonic time, not a
// device clock domain. Producer-defined ids are stable only within one profile
// session and are intended for joining records in the same bundle.
typedef struct iree_hal_profile_memory_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of memory lifecycle transition represented by this record.
  iree_hal_profile_memory_event_type_t type;
  // Flags describing queue/wait properties of this memory event.
  iree_hal_profile_memory_event_flags_t flags;
  // Type-specific result code, or UINT32_MAX when not applicable.
  uint32_t result;
  // Producer-defined event identifier unique within the memory event stream.
  uint64_t event_id;
  // IREE monotonic host timestamp in nanoseconds.
  int64_t host_time_ns;
  // Producer-defined allocation identifier for related memory events.
  uint64_t allocation_id;
  // Producer-defined pool or provider identifier for related memory events.
  uint64_t pool_id;
  // Producer-defined backing allocation or slab identifier.
  uint64_t backing_id;
  // Queue submission epoch associated with this event, or 0 when not queued.
  uint64_t submission_id;
  // Session-local physical device ordinal associated with this event.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this event, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Number of frontier entries associated with a wait event.
  uint32_t frontier_entry_count;
  // Reserved for future memory event fields; must be zero.
  uint32_t reserved0;
  // HAL memory type bits associated with the allocation.
  uint64_t memory_type;
  // HAL buffer usage bits associated with the allocation.
  uint64_t buffer_usage;
  // Byte offset within |backing_id|, when known.
  uint64_t offset;
  // Byte length of the allocation, reservation, or slab.
  uint64_t length;
  // Requested or guaranteed byte alignment for the allocation.
  uint64_t alignment;
} iree_hal_profile_memory_event_t;

// Returns a default memory lifecycle event record.
static inline iree_hal_profile_memory_event_t
iree_hal_profile_memory_event_default(void) {
  iree_hal_profile_memory_event_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.result = UINT32_MAX;
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  return record;
}

// Metadata describing one profiling chunk.
typedef struct iree_hal_profile_chunk_metadata_t {
  // MIME-like content type of the chunk payload.
  iree_string_view_t content_type;
  // Human-readable stream or artifact name.
  iree_string_view_t name;
  // Process-local profiling session identifier.
  uint64_t session_id;
  // Producer-defined stream identifier within |session_id|.
  uint64_t stream_id;
  // Producer-defined event identifier associated with this chunk, or 0.
  uint64_t event_id;
  // Process-local executable identifier associated with this chunk, or 0.
  uint64_t executable_id;
  // Process-local command-buffer identifier associated with this chunk, or 0.
  uint64_t command_buffer_id;
  // Physical device ordinal associated with this chunk, or UINT32_MAX.
  uint32_t physical_device_ordinal;
  // Queue ordinal associated with this chunk, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Flags describing chunk completeness and producer behavior.
  iree_hal_profile_chunk_flags_t flags;
} iree_hal_profile_chunk_metadata_t;

// Returns default profiling chunk metadata.
static inline iree_hal_profile_chunk_metadata_t
iree_hal_profile_chunk_metadata_default(void) {
  iree_hal_profile_chunk_metadata_t metadata;
  memset(&metadata, 0, sizeof(metadata));
  metadata.physical_device_ordinal = UINT32_MAX;
  metadata.queue_ordinal = UINT32_MAX;
  return metadata;
}

// Retains the given |sink| for the caller.
IREE_API_EXPORT void iree_hal_profile_sink_retain(
    iree_hal_profile_sink_t* sink);

// Releases the given |sink| from the caller.
IREE_API_EXPORT void iree_hal_profile_sink_release(
    iree_hal_profile_sink_t* sink);

// Begins a profiling session on |sink|.
//
// The |metadata| describes the session-level stream. Implementations may use
// this to allocate per-session state, write headers, or validate that the sink
// can accept the requested content type. All strings and spans passed to this
// function are borrowed and need only remain valid for the duration of the
// call.
IREE_API_EXPORT iree_status_t iree_hal_profile_sink_begin_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata);

// Writes one profiling chunk to |sink|.
//
// The sink must consume or copy the provided iovecs before returning. The
// producer may reuse or release the storage immediately after this call
// returns.
IREE_API_EXPORT iree_status_t iree_hal_profile_sink_write(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs);

// Ends a profiling session on |sink|.
//
// |session_status_code| describes the producer-side session outcome without
// transferring ownership of an iree_status_t. Sinks that need richer error
// details should receive them as a chunk before end_session.
IREE_API_EXPORT iree_status_t iree_hal_profile_sink_end_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code);

//===----------------------------------------------------------------------===//
// iree_hal_profile_sink_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_profile_sink_vtable_t {
  // Destroys the sink when its final reference is released.
  void(IREE_API_PTR* destroy)(iree_hal_profile_sink_t* sink);

  // Notifies the sink that a new profiling session is beginning.
  iree_status_t(IREE_API_PTR* begin_session)(
      iree_hal_profile_sink_t* sink,
      const iree_hal_profile_chunk_metadata_t* metadata);

  // Writes one complete profiling chunk to the sink.
  iree_status_t(IREE_API_PTR* write)(
      iree_hal_profile_sink_t* sink,
      const iree_hal_profile_chunk_metadata_t* metadata,
      iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs);

  // Notifies the sink that the profiling session has ended.
  iree_status_t(IREE_API_PTR* end_session)(
      iree_hal_profile_sink_t* sink,
      const iree_hal_profile_chunk_metadata_t* metadata,
      iree_status_code_t session_status_code);
} iree_hal_profile_sink_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_profile_sink_vtable_t);

IREE_API_EXPORT void iree_hal_profile_sink_destroy(
    iree_hal_profile_sink_t* sink);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_PROFILE_SINK_H_
