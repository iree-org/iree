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
  // a complete representation of the selected range. Producers that know the
  // number of omitted typed records should report it in
  // iree_hal_profile_chunk_metadata_t::dropped_record_count.
  //
  // Producers define pressure behavior by data family. Aggregate observability
  // streams such as queue and memory events may use this flag to keep hot paths
  // bounded. Precise execution timelines such as dispatch and device queue
  // events should fail the profiled operation/session instead of silently
  // omitting records needed for timing attribution.
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

// Content type for packed iree_hal_profile_executable_code_object_record_t
// records followed by exact code-object bytes.
#define IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS \
  IREE_SV("application/vnd.iree.hal.profile.executable-code-objects")

// Content type for an array of
// iree_hal_profile_executable_code_object_load_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS \
  IREE_SV("application/vnd.iree.hal.profile.executable-code-object-loads")

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

// Content type for an array of iree_hal_profile_queue_device_event_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS \
  IREE_SV("application/vnd.iree.hal.profile.queue-device-events")

// Content type for an array of iree_hal_profile_host_execution_event_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS \
  IREE_SV("application/vnd.iree.hal.profile.host-execution-events")

// Content type for an array of iree_hal_profile_memory_event_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS \
  IREE_SV("application/vnd.iree.hal.profile.memory-events")

// Content type for an array of iree_hal_profile_event_relationship_record_t.
#define IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS \
  IREE_SV("application/vnd.iree.hal.profile.event-relationships")

// Content type for packed iree_hal_profile_counter_set_record_t records.
#define IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS \
  IREE_SV("application/vnd.iree.hal.profile.counter-sets")

// Content type for packed iree_hal_profile_counter_record_t records.
#define IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS \
  IREE_SV("application/vnd.iree.hal.profile.counters")

// Content type for packed iree_hal_profile_counter_sample_record_t records.
#define IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES \
  IREE_SV("application/vnd.iree.hal.profile.counter-samples")

// Content type for one iree_hal_profile_executable_trace_record_t followed by
// raw trace bytes.
#define IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES \
  IREE_SV("application/vnd.iree.hal.profile.executable-traces")

//===----------------------------------------------------------------------===//
// Profile time domains
//===----------------------------------------------------------------------===//

// HAL profile records intentionally preserve raw timestamps in their native
// clock domains. Consumers must not compare absolute timestamps from different
// domains unless a clock-correlation record explicitly connects those domains.
// Producers must not populate fields from a different clock domain as a
// fallback: a host-timestamped span belongs in a host-time record, and a
// device-timestamped span requires a real device tick.
//
// Canonical domain names used by JSONL tooling exports:
//
// - iree_host_time_ns:
//   Nanoseconds returned by iree_time_now(). On POSIX this is IREE's
//   monotonic host clock domain, not necessarily CLOCK_MONOTONIC_RAW.
//
// - device_tick:
//   Raw physical-device timestamp ticks. The scale is producer/device-specific
//   and must be fitted using clock-correlation records before conversion to
//   nanoseconds or a host timeline.
//
// - driver_host_cpu_timestamp_ns:
//   Driver-provided CPU timestamp sampled with a device tick. The unit is
//   nanoseconds, but the clock domain is chosen by the driver. For AMDGPU KFD
//   this is the kernel raw monotonic counter, which is not identical to
//   iree_host_time_ns.
//
// - driver_host_system_timestamp:
//   Driver-provided system timestamp sampled with a device tick. The unit is
//   defined by the paired host_system_frequency_hz field.

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
  // The |code_object_hash| field contains a producer-defined deterministic
  // content hash.
  IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH = 1u << 0,
};

// Session-level executable description.
//
// Producers should emit executable records before dispatch event records that
// reference |executable_id|. The id is a compact producer-defined key that is
// unique within the profiling session; consumers should use code-object hashes
// when present for cross-session or cross-process correlation.
typedef struct iree_hal_profile_executable_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags specifying which optional executable fields are populated.
  iree_hal_profile_executable_flags_t flags;
  // Session-local executable identifier referenced by dispatch events.
  uint64_t executable_id;
  // Number of export records associated with this executable.
  uint32_t export_count;
  // Reserved for future executable record fields; must be zero.
  uint32_t reserved0;
  // Producer-defined deterministic code-object content hash words when present
  // in |flags|. Consumers should treat the hash as an opaque equality key
  // unless the producer documents its algorithm and inputs.
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

// Bitfield specifying which code-object record fields are populated.
typedef uint32_t iree_hal_profile_executable_code_object_flags_t;
enum iree_hal_profile_executable_code_object_flag_bits_t {
  IREE_HAL_PROFILE_EXECUTABLE_CODE_OBJECT_FLAG_NONE = 0u,
  // The |code_object_hash| field contains a producer-defined deterministic
  // content hash.
  IREE_HAL_PROFILE_EXECUTABLE_CODE_OBJECT_FLAG_CODE_OBJECT_HASH = 1u << 0,
};

// Session-level executable code-object image followed by |data_length| bytes.
//
// Producers should emit code-object records before executable trace records
// that reference the corresponding |executable_id| and |code_object_id|. The
// trailing bytes are the exact code-object image loaded by the runtime so
// offline consumers can disassemble and decode hardware trace PCs without
// consulting process-local executable state.
typedef struct iree_hal_profile_executable_code_object_record_t {
  // Size of this record in bytes including trailing code-object image bytes.
  uint32_t record_length;
  // Flags specifying which optional code-object fields are populated.
  iree_hal_profile_executable_code_object_flags_t flags;
  // Session-local executable identifier owning this code object.
  uint64_t executable_id;
  // Session-local code-object marker identifier used by hardware traces.
  uint64_t code_object_id;
  // Byte length of the trailing code-object image.
  uint64_t data_length;
  // Producer-defined deterministic code-object content hash words when present
  // in |flags|. Consumers should treat the hash as an opaque equality key
  // unless the producer documents its algorithm and inputs.
  uint64_t code_object_hash[2];
} iree_hal_profile_executable_code_object_record_t;

// Returns a default executable code-object record.
static inline iree_hal_profile_executable_code_object_record_t
iree_hal_profile_executable_code_object_record_default(void) {
  iree_hal_profile_executable_code_object_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  return record;
}

// Session-level executable code-object load range for one physical device.
//
// Hardware trace decoders use |load_delta|, not the process virtual load base,
// to translate runtime PCs into code-object virtual addresses. Producers should
// emit one load record for each physical device where |code_object_id| was
// loaded.
typedef struct iree_hal_profile_executable_code_object_load_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Session-local physical device ordinal owning this loaded code object.
  uint32_t physical_device_ordinal;
  // Session-local executable identifier owning this code object.
  uint64_t executable_id;
  // Session-local code-object marker identifier used by hardware traces.
  uint64_t code_object_id;
  // Loader-provided code-object load delta used for PC translation.
  int64_t load_delta;
  // Byte length of the loaded code-object range on the device.
  uint64_t load_size;
} iree_hal_profile_executable_code_object_load_record_t;

// Returns a default executable code-object load record.
static inline iree_hal_profile_executable_code_object_load_record_t
iree_hal_profile_executable_code_object_load_record_default(void) {
  iree_hal_profile_executable_code_object_load_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying which executable export fields are populated.
typedef uint32_t iree_hal_profile_executable_export_flags_t;
enum iree_hal_profile_executable_export_flag_bits_t {
  IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_NONE = 0u,
  // The |pipeline_hash| field contains a producer-defined deterministic export
  // identity hash.
  IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH = 1u << 0,
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
  // Session-local executable identifier owning this export.
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
  // Producer-defined deterministic executable-export identity hash words when
  // present in |flags|. Consumers should treat the hash as an opaque equality
  // key unless the producer documents its algorithm and inputs.
  uint64_t pipeline_hash[2];
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
  // Session-local command-buffer identifier referenced by dispatch events.
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

  // Producer-local block coordinates are populated. Producers with linear or
  // opaque command encodings must clear this flag and leave all block ordinal
  // fields as UINT32_MAX.
  IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_BLOCK_STRUCTURE = 1u << 6,
};

// Session-level reusable command-buffer operation description.
//
// Producers should emit command-operation records after the command-buffer
// record defining |command_buffer_id| and before dispatch event records that
// reference |command_buffer_id| and |command_index|. |command_index| is the
// portable operation identity within |command_buffer_id| and the join key used
// by dispatch events. The optional block fields are producer-local detail for
// command encodings with explicit blocks or control-flow regions, such as the
// AMDGPU AQL command-buffer program and local-task block ISA. Producers with a
// linear or opaque command encoding must leave the block fields absent instead
// of fabricating block structure.
typedef struct iree_hal_profile_command_operation_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of command-buffer operation represented by this record.
  iree_hal_profile_command_operation_type_t type;
  // Flags describing operation properties.
  iree_hal_profile_command_operation_flags_t flags;
  // Command ordinal within |command_buffer_id|.
  uint32_t command_index;
  // Session-local command-buffer identifier.
  uint64_t command_buffer_id;
  // Producer-local command block ordinal, or UINT32_MAX when absent.
  uint32_t block_ordinal;
  // Command ordinal within |block_ordinal|, or UINT32_MAX when absent.
  uint32_t block_command_ordinal;
  // Session-local executable identifier, or 0 when not applicable.
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
  // Primary branch target block ordinal, or UINT32_MAX when absent.
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

// Returns true if a command-operation record carries producer-local block
// coordinates.
static inline bool iree_hal_profile_command_operation_has_block_structure(
    const iree_hal_profile_command_operation_record_t* record) {
  return iree_all_bits_set(
      record->flags, IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_BLOCK_STRUCTURE);
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
// Producers must only set flags for clock samples that were actually obtained
// and validated; failed, unsupported, or implausible samples must be omitted or
// reported as errors instead of encoded as zero-valued correlations.
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
  // Producer-defined event identifier unique within the dispatch event stream.
  uint64_t event_id;
  // Queue submission epoch containing this dispatch.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 for direct queue dispatch.
  uint64_t command_buffer_id;
  // Session-local executable identifier, or 0 when unavailable.
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
// submission id was assigned, and how much payload the operation covered.
// Timestamps are in IREE host monotonic time, not a device clock domain.
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
  // IREE monotonic host timestamp when the operation was submitted.
  int64_t host_time_ns;
  // Queue submission epoch associated with this operation, or 0 when absent.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 when not applicable.
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
  // IREE monotonic host timestamp when the operation became ready to execute.
  // May be 0 when readiness is not producer-observable or the operation failed
  // before it became ready.
  int64_t ready_host_time_ns;
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

// Device-timestamped queue operation event.
//
// Queue device events describe when queue work became visible to the physical
// device, in the device's raw timestamp domain. They complement, but do not
// replace, host-timestamped queue events: a producer may emit them without
// QUEUE_EVENTS being requested. Consumers should join records by explicit
// relationship records or by queue submission id when relationships are absent.
typedef struct iree_hal_profile_queue_device_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of queue operation represented by this device event.
  iree_hal_profile_queue_event_type_t type;
  // Flags describing queue operation properties.
  iree_hal_profile_queue_event_flags_t flags;
  // Reserved for future queue device event fields; must be zero.
  uint32_t reserved0;
  // Producer-defined event identifier unique within the queue device event
  // stream.
  uint64_t event_id;
  // Queue submission epoch containing this device event.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;
  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;
  // Producer-defined stream identifier matching the queue metadata record.
  uint64_t stream_id;
  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;
  // Session-local physical device ordinal associated with this operation.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this operation.
  uint32_t queue_ordinal;
  // Number of encoded payload operations represented by this queue operation.
  uint32_t operation_count;
  // Reserved for future queue device event fields; must be zero.
  uint32_t reserved1;
  // Device timestamp captured when queue-visible work started.
  uint64_t start_tick;
  // Device timestamp captured when queue-visible work completed.
  uint64_t end_tick;
} iree_hal_profile_queue_device_event_t;

// Returns a default device-timestamped queue operation event record.
static inline iree_hal_profile_queue_device_event_t
iree_hal_profile_queue_device_event_default(void) {
  iree_hal_profile_queue_device_event_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying properties of one host execution span.
typedef uint32_t iree_hal_profile_host_execution_event_flags_t;
enum iree_hal_profile_host_execution_event_flag_bits_t {
  IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_NONE = 0u,

  // Execution was produced by a reusable command buffer.
  IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_COMMAND_BUFFER = 1u << 0,

  // Workgroup counts were loaded indirectly before execution.
  IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_INDIRECT_PARAMETERS = 1u << 1,

  // Execution completed from a deferred callback or worker rather than inline
  // in the queue submission call.
  IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_DEFERRED = 1u << 2,
};

// Host-timestamped queue operation execution span.
//
// Host execution events describe complete spans observed in IREE's host clock
// domain. CPU/local producers use these for dispatch, command-buffer execute,
// transfer, allocation, and host-call work that is not device-timestamped.
// Producers must not emit partial in-flight spans as complete records and must
// only emit spans whose end time is greater than or equal to their start time.
typedef struct iree_hal_profile_host_execution_event_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of queue operation represented by this span.
  iree_hal_profile_queue_event_type_t type;
  // Flags describing host execution properties.
  iree_hal_profile_host_execution_event_flags_t flags;
  // IREE status code for terminal execution result, or UINT32_MAX if unknown.
  uint32_t status_code;
  // Producer-defined event identifier unique within the host execution event
  // stream.
  uint64_t event_id;
  // Queue submission epoch containing this span, or 0 when absent.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;
  // Session-local executable identifier, or 0 when not applicable.
  uint64_t executable_id;
  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;
  // Producer-defined stream identifier matching the queue metadata record.
  uint64_t stream_id;
  // Session-local physical device ordinal associated with this span.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this span, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Command ordinal within a command buffer, or UINT32_MAX when absent.
  uint32_t command_index;
  // Executable export ordinal, or UINT32_MAX when absent.
  uint32_t export_ordinal;
  // Workgroup counts submitted for dispatch-like spans.
  uint32_t workgroup_count[3];
  // Workgroup sizes submitted for dispatch-like spans.
  uint32_t workgroup_size[3];
  // IREE monotonic host timestamp when execution started.
  int64_t start_host_time_ns;
  // IREE monotonic host timestamp when execution completed.
  int64_t end_host_time_ns;
  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;
  // Number of execution tiles represented by this span, or 0 when the
  // producer did not provide tile attribution.
  uint64_t tile_count;
  // Sum of per-tile execution durations in nanoseconds. This may exceed the
  // span duration when tiles execute concurrently.
  int64_t tile_duration_sum_ns;
  // Number of encoded payload operations represented by this span.
  uint32_t operation_count;
  // Reserved for future host execution fields; must be zero.
  uint32_t reserved0;
} iree_hal_profile_host_execution_event_t;

// Returns a default host-timestamped execution span record.
static inline iree_hal_profile_host_execution_event_t
iree_hal_profile_host_execution_event_default(void) {
  iree_hal_profile_host_execution_event_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.status_code = UINT32_MAX;
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  record.command_index = UINT32_MAX;
  record.export_ordinal = UINT32_MAX;
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

  // Synchronous HAL buffer allocation.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE = 9u,

  // Synchronous HAL buffer free.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE = 10u,

  // Device-visible externally-owned buffer imported into the HAL.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT = 11u,

  // Device-visible externally-owned buffer released from the HAL.
  IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT = 12u,
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

  // Event carries an atomic pool-stat snapshot.
  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS = 1u << 4,

  // Event describes memory whose backing allocation is externally owned.
  IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED = 1u << 5,
};

// Host-timestamped memory lifecycle event.
//
// Memory events describe allocations, reservations, and queue-visible
// allocation operations. The timestamp is in IREE host monotonic time, not a
// device clock domain. |allocation_id| is a producer-defined session-local
// lifecycle identifier and is the primary join key for memory events. |pool_id|
// and |backing_id| are producer-defined implementation identifiers used for
// drilldown; they may be raw addresses, provider handles, or driver objects and
// must not be treated as allocation lifecycle identities. Pool-stat snapshots,
// when present, describe the pool state after the event's producer-visible
// mutation.
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
  // Producer-defined session-local allocation lifecycle identifier.
  uint64_t allocation_id;
  // Producer-defined pool or provider implementation identifier.
  uint64_t pool_id;
  // Producer-defined backing allocation, address, or slab identifier.
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
  // Pool bytes currently occupied by live reservations, when present.
  uint64_t pool_bytes_reserved;
  // Pool bytes currently free or available for reservation, when present.
  uint64_t pool_bytes_free;
  // Pool physical memory committed in slabs or pages, when present.
  uint64_t pool_bytes_committed;
  // Pool budget limit in bytes, or 0 for unlimited, when present.
  uint64_t pool_budget_limit;
  // Pool live reservation count, when present.
  uint32_t pool_reservation_count;
  // Pool committed slab count, when present.
  uint32_t pool_slab_count;
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

// Type of relationship between two profile entities.
typedef uint32_t iree_hal_profile_event_relationship_type_t;
enum iree_hal_profile_event_relationship_type_e {
  IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_NONE = 0u,

  // A queue submission contains a device-side dispatch event.
  IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_DISPATCH = 1u,

  // A queue submission contains a device-side queue operation event.
  IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_QUEUE_DEVICE_EVENT =
      2u,

  // A host queue event corresponds to a host execution span.
  IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_EVENT_HOST_EXECUTION_EVENT =
      3u,

  // A queue submission contains a host execution span.
  IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_HOST_EXECUTION_EVENT =
      4u,
};

// Type of profile entity referenced by a relationship endpoint.
//
// Endpoint identifiers use the table below:
// - QUEUE_SUBMISSION: primary id is the queue submission id.
// - QUEUE_EVENT: primary id is iree_hal_profile_queue_event_t.event_id.
// - DISPATCH_EVENT: primary id is iree_hal_profile_dispatch_event_t.event_id.
// - COMMAND_OPERATION: primary id is command_buffer_id and secondary id is
//   command_index.
// - MEMORY_EVENT: primary id is iree_hal_profile_memory_event_t.event_id.
// - ARTIFACT: primary id is producer-defined artifact id.
// - QUEUE_DEVICE_EVENT: primary id is
//   iree_hal_profile_queue_device_event_t.event_id.
// - HOST_EXECUTION_EVENT: primary id is
//   iree_hal_profile_host_execution_event_t.event_id.
typedef uint32_t iree_hal_profile_event_endpoint_type_t;
enum iree_hal_profile_event_endpoint_type_e {
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_NONE = 0u,

  // Queue submission identified by submission id.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION = 1u,

  // Host queue event identified by event id.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_EVENT = 2u,

  // Device dispatch event identified by event id.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_DISPATCH_EVENT = 3u,

  // Recorded command-buffer operation identified by command-buffer id and
  // command index.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_COMMAND_OPERATION = 4u,

  // Memory lifecycle event identified by event id.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_MEMORY_EVENT = 5u,

  // External or embedded artifact identified by producer-defined artifact id.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_ARTIFACT = 6u,

  // Device queue event identified by event id.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_DEVICE_EVENT = 7u,

  // Host execution event identified by event id.
  IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_HOST_EXECUTION_EVENT = 8u,
};

// Relationship between two profile entities.
//
// Relationship records make cross-record joins explicit in the raw profile
// bundle. Viewers should use these records for arrows, flows, and drilldown
// links instead of reconstructing relationships from coincidental matching ids.
//
// Relationship ids are producer-defined and unique within the tuple of
// |physical_device_ordinal|, |queue_ordinal|, and |stream_id| for the profiling
// session. Endpoint ids follow the endpoint-type table above and are
// interpreted within the same device/queue/stream tuple unless the endpoint
// type explicitly names a wider namespace, such as executable ids or
// command-buffer ids stored in the referenced record.
typedef struct iree_hal_profile_event_relationship_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Kind of relationship represented by this record.
  iree_hal_profile_event_relationship_type_t type;
  // Producer-defined relationship id unique within this stream scope.
  uint64_t relationship_id;
  // Source endpoint kind.
  iree_hal_profile_event_endpoint_type_t source_type;
  // Target endpoint kind.
  iree_hal_profile_event_endpoint_type_t target_type;
  // Session-local physical device ordinal shared by both endpoints.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal shared by both endpoints.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this relationship.
  uint64_t stream_id;
  // Primary source endpoint id.
  uint64_t source_id;
  // Secondary source endpoint id, or 0 when absent.
  uint64_t source_secondary_id;
  // Primary target endpoint id.
  uint64_t target_id;
  // Secondary target endpoint id, or 0 when absent.
  uint64_t target_secondary_id;
} iree_hal_profile_event_relationship_record_t;

// Returns a default event relationship record.
static inline iree_hal_profile_event_relationship_record_t
iree_hal_profile_event_relationship_record_default(void) {
  iree_hal_profile_event_relationship_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying profile capture filter predicates.
typedef uint32_t iree_hal_profile_capture_filter_flags_t;
enum iree_hal_profile_capture_filter_flag_bits_t {
  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_NONE = 0u,

  // Match only executable exports whose names match
  // |executable_export_pattern|.
  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN = 1u << 0,

  // Match only operations associated with |command_buffer_id|.
  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID = 1u << 1,

  // Match only command-buffer operations whose index is |command_index|. Direct
  // queue operations have no command index and never match this predicate.
  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX = 1u << 2,

  // Match only operations on |physical_device_ordinal|.
  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL = 1u << 3,

  // Match only operations on |queue_ordinal|.
  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL = 1u << 4,
};

// Selects which operations should produce heavy profile artifacts.
//
// Profile producers should always preserve cheap session/metadata records
// needed to interpret the capture, but may use this filter to decide whether to
// emit expensive per-operation artifacts such as dispatch timestamp packets,
// hardware counter ranges, or trace markers. Fields are active only when their
// matching flag is set; a zero-initialized filter matches all operations.
typedef struct iree_hal_profile_capture_filter_t {
  // Flags selecting which fields below participate in matching.
  iree_hal_profile_capture_filter_flags_t flags;

  // Borrowed glob pattern matched with iree_string_view_match_pattern against
  // an executable export name. Profile sessions retaining this filter after
  // begin must copy the pattern into session-owned storage.
  iree_string_view_t executable_export_pattern;

  // Session-local command-buffer identifier to match.
  uint64_t command_buffer_id;

  // Zero-based command-buffer operation index to match.
  uint32_t command_index;

  // Session-local physical device ordinal to match.
  uint32_t physical_device_ordinal;

  // Session-local queue ordinal to match.
  uint32_t queue_ordinal;

  // Reserved for future filter fields; must be zero.
  uint32_t reserved0;
} iree_hal_profile_capture_filter_t;

// Returns a capture filter matching all operations.
static inline iree_hal_profile_capture_filter_t
iree_hal_profile_capture_filter_default(void) {
  iree_hal_profile_capture_filter_t filter;
  memset(&filter, 0, sizeof(filter));
  return filter;
}

// Returns true when |filter| has no active predicates.
static inline bool iree_hal_profile_capture_filter_is_default(
    const iree_hal_profile_capture_filter_t* filter) {
  return filter->flags == IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_NONE;
}

// Returns true if |filter| matches the given profile location fields.
static inline bool iree_hal_profile_capture_filter_matches_location(
    const iree_hal_profile_capture_filter_t* filter, uint64_t command_buffer_id,
    uint32_t command_index, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal) {
  if (iree_any_bit_set(
          filter->flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID) &&
      filter->command_buffer_id != command_buffer_id) {
    return false;
  }
  if (iree_any_bit_set(filter->flags,
                       IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX)) {
    if (command_buffer_id == 0 || filter->command_index != command_index) {
      return false;
    }
  }
  if (iree_any_bit_set(
          filter->flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL) &&
      filter->physical_device_ordinal != physical_device_ordinal) {
    return false;
  }
  if (iree_any_bit_set(filter->flags,
                       IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL) &&
      filter->queue_ordinal != queue_ordinal) {
    return false;
  }
  return true;
}

// Bitfield specifying properties of a requested hardware counter set.
typedef uint32_t iree_hal_profile_counter_set_selection_flags_t;
enum iree_hal_profile_counter_set_selection_flag_bits_t {
  IREE_HAL_PROFILE_COUNTER_SET_SELECTION_FLAG_NONE = 0u,
};

// Caller-provided hardware counter set selection.
//
// The selection describes one named group of hardware counters requested for a
// profiling session. All pointers are borrowed and must remain valid for the
// duration of iree_hal_device_profiling_begin. A producer that supports the
// selected counters emits one counter-set metadata record, one counter metadata
// record per resolved counter, and counter-sample records using the same
// |counter_set_id|.
typedef struct iree_hal_profile_counter_set_selection_t {
  // Flags controlling counter set selection behavior.
  iree_hal_profile_counter_set_selection_flags_t flags;
  // Human-readable counter set name used in emitted metadata.
  iree_string_view_t name;
  // Number of requested counter names in |counter_names|.
  iree_host_size_t counter_name_count;
  // Borrowed array of requested implementation-specific counter names.
  const iree_string_view_t* counter_names;
} iree_hal_profile_counter_set_selection_t;

// Bitfield specifying properties of one emitted hardware counter set.
typedef uint32_t iree_hal_profile_counter_set_flags_t;
enum iree_hal_profile_counter_set_flag_bits_t {
  IREE_HAL_PROFILE_COUNTER_SET_FLAG_NONE = 0u,
};

// Session-level hardware counter set description followed by |name_length|
// bytes. The trailing name is not NUL-terminated.
//
// A counter set defines the value-vector layout used by all sample records that
// reference |counter_set_id|. Consumers join sample values to counter metadata
// by applying each counter record's |sample_value_offset| and
// |sample_value_count| within the sample record's trailing uint64_t value
// vector.
typedef struct iree_hal_profile_counter_set_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags describing counter set properties.
  iree_hal_profile_counter_set_flags_t flags;
  // Producer-local counter set identifier referenced by counters and samples.
  uint64_t counter_set_id;
  // Session-local physical device ordinal associated with this counter set.
  uint32_t physical_device_ordinal;
  // Number of counter records associated with this counter set.
  uint32_t counter_count;
  // Number of uint64_t values in each sample record for this counter set.
  uint32_t sample_value_count;
  // Byte length of the trailing counter set name.
  uint32_t name_length;
} iree_hal_profile_counter_set_record_t;

// Returns a default hardware counter set record.
static inline iree_hal_profile_counter_set_record_t
iree_hal_profile_counter_set_record_default(void) {
  iree_hal_profile_counter_set_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying properties of one emitted hardware counter.
typedef uint32_t iree_hal_profile_counter_flags_t;
enum iree_hal_profile_counter_flag_bits_t {
  IREE_HAL_PROFILE_COUNTER_FLAG_NONE = 0u,

  // Counter is a raw hardware value without producer-side derivation.
  IREE_HAL_PROFILE_COUNTER_FLAG_RAW = 1u << 0,
};

// Unit describing how a counter value should be displayed.
typedef uint32_t iree_hal_profile_counter_unit_t;
enum iree_hal_profile_counter_unit_e {
  IREE_HAL_PROFILE_COUNTER_UNIT_NONE = 0u,

  // Counter values represent an unscaled event count.
  IREE_HAL_PROFILE_COUNTER_UNIT_COUNT = 1u,

  // Counter values represent clock cycles.
  IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES = 2u,

  // Counter values represent bytes.
  IREE_HAL_PROFILE_COUNTER_UNIT_BYTES = 3u,
};

// Session-level hardware counter description followed by |block_name_length|,
// |name_length|, and |description_length| bytes. Trailing strings are stored in
// that order and are not NUL-terminated.
//
// Some hardware counter requests expand into multiple raw values due to device
// topology dimensions such as XCC/SE/block instances. The
// |sample_value_offset| and |sample_value_count| fields describe where those
// raw values are stored in each sample record for the owning counter set.
typedef struct iree_hal_profile_counter_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags describing counter properties.
  iree_hal_profile_counter_flags_t flags;
  // Display unit for raw sample values.
  iree_hal_profile_counter_unit_t unit;
  // Session-local physical device ordinal associated with this counter.
  uint32_t physical_device_ordinal;
  // Producer-local counter set identifier owning this counter.
  uint64_t counter_set_id;
  // Counter ordinal within |counter_set_id|.
  uint32_t counter_ordinal;
  // First uint64_t value slot occupied by this counter in each sample record.
  uint32_t sample_value_offset;
  // Number of uint64_t value slots occupied by this counter in each sample.
  uint32_t sample_value_count;
  // Byte length of the trailing hardware block name.
  uint32_t block_name_length;
  // Byte length of the trailing counter name.
  uint32_t name_length;
  // Byte length of the trailing counter description.
  uint32_t description_length;
} iree_hal_profile_counter_record_t;

// Returns a default hardware counter record.
static inline iree_hal_profile_counter_record_t
iree_hal_profile_counter_record_default(void) {
  iree_hal_profile_counter_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.physical_device_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying properties of one emitted hardware counter sample.
typedef uint32_t iree_hal_profile_counter_sample_flags_t;
enum iree_hal_profile_counter_sample_flag_bits_t {
  IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_NONE = 0u,

  // |dispatch_event_id| references a dispatch event in the same profile
  // session.
  IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT = 1u << 0,

  // |command_buffer_id| and |command_index| reference a command-buffer
  // operation in the same profile session.
  IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION = 1u << 1,
};

// Hardware counter sample followed by |sample_value_count| uint64_t values.
//
// Samples are intentionally vector-shaped: one sample record represents one
// measured dispatch/range for one counter set, and counter metadata defines how
// to split the trailing value vector into named counters. This keeps dense
// dispatch-counter streams compact while preserving raw per-instance hardware
// values for later tooling-side aggregation.
typedef struct iree_hal_profile_counter_sample_record_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // Flags describing which correlation fields are valid.
  iree_hal_profile_counter_sample_flags_t flags;
  // Producer-local sample identifier unique within the counter sample stream.
  uint64_t sample_id;
  // Producer-local counter set identifier defining the trailing value layout.
  uint64_t counter_set_id;
  // Dispatch event identifier associated with this sample, or 0 when absent.
  uint64_t dispatch_event_id;
  // Queue submission epoch associated with this sample, or 0 when absent.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 when absent.
  uint64_t command_buffer_id;
  // Session-local executable identifier, or 0 when absent.
  uint64_t executable_id;
  // Producer-defined stream identifier matching queue metadata, or 0.
  uint64_t stream_id;
  // Command ordinal within a command buffer, or UINT32_MAX when absent.
  uint32_t command_index;
  // Executable export ordinal, or UINT32_MAX when absent.
  uint32_t export_ordinal;
  // Session-local physical device ordinal associated with this sample.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this sample, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Number of trailing uint64_t sample values.
  uint32_t sample_value_count;
  // Reserved for future counter sample fields; must be zero.
  uint32_t reserved0;
} iree_hal_profile_counter_sample_record_t;

// Returns a default hardware counter sample record.
static inline iree_hal_profile_counter_sample_record_t
iree_hal_profile_counter_sample_record_default(void) {
  iree_hal_profile_counter_sample_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.command_index = UINT32_MAX;
  record.export_ordinal = UINT32_MAX;
  record.physical_device_ordinal = UINT32_MAX;
  record.queue_ordinal = UINT32_MAX;
  return record;
}

// Bitfield specifying properties of one executable trace record.
typedef uint32_t iree_hal_profile_executable_trace_flags_t;
enum iree_hal_profile_executable_trace_flag_bits_t {
  IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_NONE = 0u,

  // |dispatch_event_id| references a dispatch event in the same profile
  // session.
  IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_DISPATCH_EVENT = 1u << 0,

  // |command_buffer_id| and |command_index| reference a command-buffer
  // operation in the same profile session.
  IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_COMMAND_OPERATION = 1u << 1,
};

// Raw executable trace format.
typedef uint32_t iree_hal_profile_executable_trace_format_t;
enum iree_hal_profile_executable_trace_format_e {
  IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_NONE = 0u,

  // AMD ATT/SQTT bytes as returned by aqlprofile_att_iterate_data.
  IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT = 1u,
};

// Executable trace artifact followed by |data_length| raw bytes.
//
// Trace chunks are intentionally one artifact per chunk because instruction
// traces may be large. Consumers join the artifact back to dispatch events,
// command-buffer operations, executable metadata, and queue metadata using the
// ids embedded here and in the chunk metadata.
typedef struct iree_hal_profile_executable_trace_record_t {
  // Size of this record in bytes, excluding trailing trace bytes.
  uint32_t record_length;
  // Raw trace byte format.
  iree_hal_profile_executable_trace_format_t format;
  // Flags describing which correlation fields are valid.
  iree_hal_profile_executable_trace_flags_t flags;
  // Shader engine or producer-defined trace partition ordinal.
  uint32_t shader_engine;
  // Producer-defined trace identifier unique within the trace stream.
  uint64_t trace_id;
  // Dispatch event identifier associated with this trace, or 0 when absent.
  uint64_t dispatch_event_id;
  // Queue submission epoch associated with this trace, or 0 when absent.
  uint64_t submission_id;
  // Session-local command-buffer identifier, or 0 when absent.
  uint64_t command_buffer_id;
  // Session-local executable identifier, or 0 when absent.
  uint64_t executable_id;
  // Producer-defined stream identifier matching queue metadata, or 0.
  uint64_t stream_id;
  // Command ordinal within a command buffer, or UINT32_MAX when absent.
  uint32_t command_index;
  // Executable export ordinal, or UINT32_MAX when absent.
  uint32_t export_ordinal;
  // Session-local physical device ordinal associated with this trace.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this trace, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Byte length of the trailing raw trace data.
  uint64_t data_length;
} iree_hal_profile_executable_trace_record_t;

// Returns a default executable trace record.
static inline iree_hal_profile_executable_trace_record_t
iree_hal_profile_executable_trace_record_default(void) {
  iree_hal_profile_executable_trace_record_t record;
  memset(&record, 0, sizeof(record));
  record.record_length = sizeof(record);
  record.command_index = UINT32_MAX;
  record.export_ordinal = UINT32_MAX;
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
  // Session-local executable identifier associated with this chunk, or 0.
  uint64_t executable_id;
  // Session-local command-buffer identifier associated with this chunk, or 0.
  uint64_t command_buffer_id;
  // Physical device ordinal associated with this chunk, or UINT32_MAX.
  uint32_t physical_device_ordinal;
  // Queue ordinal associated with this chunk, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Flags describing chunk completeness and producer behavior.
  iree_hal_profile_chunk_flags_t flags;
  // Number of typed records omitted from this chunk stream, or 0 when unknown
  // or not truncated. Producers should set
  // IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED when this is nonzero. A truncated
  // chunk may have zero payload records; in that case the metadata still
  // reports that records were lost from the producer-side stream.
  uint64_t dropped_record_count;
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
// returns. |iovec_count| may be zero for metadata-only chunks, including
// TRUNCATED chunks that report producer-side dropped records even when no typed
// records survived.
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
