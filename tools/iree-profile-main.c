// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, format, "text",
          "Output format for profile commands: one of `text` or `jsonl`.");
IREE_FLAG(string, filter, "*",
          "Name/key wildcard filter for profile projection output.");
IREE_FLAG(int64_t, id, -1,
          "Optional id filter for projection commands: dispatch event id, "
          "executable id, command-buffer id, memory event/allocation id, or "
          "queue submission id.");
IREE_FLAG(bool, dispatch_events, false,
          "Emits individual dispatch event rows for projection commands with "
          "`--format=jsonl`.");
IREE_FLAG(bool, agent_md, false,
          "Prints an agent-oriented Markdown guide for iree-profile JSONL "
          "workflows and exits.");

typedef struct iree_profile_device_summary_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Number of device metadata records seen for this ordinal.
  uint32_t device_record_count;
  // Number of queues reported in device metadata.
  uint32_t queue_count;
  // Number of queue metadata records seen for this physical device.
  uint32_t queue_record_count;
  // Number of clock-correlation samples seen for this physical device.
  uint64_t clock_sample_count;
  // First clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  // Last clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
  // Minimum host bracket length observed around KFD clock-counter samples.
  int64_t minimum_clock_uncertainty_ns;
  // Maximum host bracket length observed around KFD clock-counter samples.
  int64_t maximum_clock_uncertainty_ns;
  // Number of dispatch event records seen for this physical device.
  uint64_t dispatch_event_count;
  // Number of dispatch records with unusable or reversed timestamps.
  uint64_t invalid_dispatch_event_count;
  // Sum of valid dispatch durations in raw device ticks.
  double total_dispatch_ticks;
  // Earliest valid dispatch start tick seen for this physical device.
  uint64_t earliest_dispatch_start_tick;
  // Latest valid dispatch end tick seen for this physical device.
  uint64_t latest_dispatch_end_tick;
  // Minimum valid dispatch duration in raw device ticks.
  uint64_t minimum_dispatch_ticks;
  // Maximum valid dispatch duration in raw device ticks.
  uint64_t maximum_dispatch_ticks;
} iree_profile_device_summary_t;

typedef struct iree_profile_summary_t {
  // Host allocator used for dynamic summary arrays.
  iree_allocator_t host_allocator;
  // Dynamic array of per-device summaries.
  iree_profile_device_summary_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Total file records parsed.
  uint64_t file_record_count;
  // Session-begin records parsed.
  uint64_t session_begin_count;
  // Session-end records parsed.
  uint64_t session_end_count;
  // Chunk records parsed.
  uint64_t chunk_count;
  // Records with unknown file record types.
  uint64_t unknown_record_count;
  // Chunks with the truncated flag set.
  uint64_t truncated_chunk_count;
  // Device metadata chunks parsed.
  uint64_t device_chunk_count;
  // Queue metadata chunks parsed.
  uint64_t queue_chunk_count;
  // Executable metadata chunks parsed.
  uint64_t executable_chunk_count;
  // Executable records parsed.
  uint64_t executable_record_count;
  // Executable export metadata chunks parsed.
  uint64_t executable_export_chunk_count;
  // Executable export records parsed.
  uint64_t executable_export_record_count;
  // Command-buffer metadata chunks parsed.
  uint64_t command_buffer_chunk_count;
  // Command-buffer records parsed.
  uint64_t command_buffer_record_count;
  // Command-operation metadata chunks parsed.
  uint64_t command_operation_chunk_count;
  // Command-operation records parsed.
  uint64_t command_operation_record_count;
  // Clock-correlation chunks parsed.
  uint64_t clock_correlation_chunk_count;
  // Dispatch event chunks parsed.
  uint64_t dispatch_event_chunk_count;
  // Queue event chunks parsed.
  uint64_t queue_event_chunk_count;
  // Queue event records parsed.
  uint64_t queue_event_record_count;
  // Memory event chunks parsed.
  uint64_t memory_event_chunk_count;
  // Memory event records parsed.
  uint64_t memory_event_record_count;
  // Chunk records with unknown content types.
  uint64_t unknown_chunk_count;
} iree_profile_summary_t;

typedef enum iree_profile_projection_mode_e {
  // Dispatch projection; --id matches dispatch event ids.
  IREE_PROFILE_PROJECTION_MODE_DISPATCH = 0,
  // Executable projection; --id matches executable ids.
  IREE_PROFILE_PROJECTION_MODE_EXECUTABLE = 1,
  // Command-buffer projection; --id matches command-buffer ids.
  IREE_PROFILE_PROJECTION_MODE_COMMAND = 2,
  // Queue projection; --id matches queue submission ids.
  IREE_PROFILE_PROJECTION_MODE_QUEUE = 3,
} iree_profile_projection_mode_t;

typedef struct iree_profile_memory_device_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Total memory events matched for this device.
  uint64_t event_count;
  // Slab acquire events matched for this device.
  uint64_t slab_acquire_count;
  // Slab release events matched for this device.
  uint64_t slab_release_count;
  // Current slab allocations after applying matched slab events.
  uint64_t current_slab_allocation_count;
  // Maximum current slab allocations observed while applying matched events.
  uint64_t high_water_slab_allocation_count;
  // Pool reservation events matched for this device.
  uint64_t pool_reserve_count;
  // Pool materialization events matched for this device.
  uint64_t pool_materialize_count;
  // Pool release events matched for this device.
  uint64_t pool_release_count;
  // Pool wait events matched for this device.
  uint64_t pool_wait_count;
  // Synchronous HAL buffer allocation events matched for this device.
  uint64_t buffer_allocate_count;
  // Synchronous HAL buffer free events matched for this device.
  uint64_t buffer_free_count;
  // Current live synchronous buffer allocations after matched events.
  uint64_t current_buffer_allocation_count;
  // Maximum live synchronous buffer allocations observed.
  uint64_t high_water_buffer_allocation_count;
  // Current live pool reservations after applying matched pool events.
  uint64_t current_pool_reservation_count;
  // Maximum live pool reservations observed while applying matched events.
  uint64_t high_water_pool_reservation_count;
  // Current pool reservation bytes after matched reserve/release events.
  uint64_t current_pool_reserved_bytes;
  // Maximum pool reservation bytes observed while applying matched events.
  uint64_t high_water_pool_reserved_bytes;
  // Total bytes reserved by matched pool reserve events.
  uint64_t total_pool_reserved_bytes;
  // Total bytes released by matched pool release events.
  uint64_t total_pool_released_bytes;
  // Queue alloca events matched for this device.
  uint64_t queue_alloca_count;
  // Queue dealloca events matched for this device.
  uint64_t queue_dealloca_count;
  // Current live queue allocations after applying matched queue events.
  uint64_t current_queue_allocation_count;
  // Maximum live queue allocations observed while applying matched events.
  uint64_t high_water_queue_allocation_count;
  // Current slab bytes after applying matched slab events.
  uint64_t current_slab_bytes;
  // Maximum current slab bytes observed while applying matched slab events.
  uint64_t high_water_slab_bytes;
  // Total slab bytes acquired by matched slab events.
  uint64_t total_slab_acquired_bytes;
  // Total slab bytes released by matched slab events.
  uint64_t total_slab_released_bytes;
  // Current queue-allocation bytes after matched alloca/dealloca events.
  uint64_t current_queue_bytes;
  // Maximum queue-allocation bytes observed while applying matched events.
  uint64_t high_water_queue_bytes;
  // Total bytes allocated by matched queue alloca events.
  uint64_t total_queue_alloca_bytes;
  // Total bytes deallocated by matched queue dealloca events.
  uint64_t total_queue_dealloca_bytes;
  // Current synchronous buffer bytes after matched allocate/free events.
  uint64_t current_buffer_bytes;
  // Maximum synchronous buffer bytes observed while applying matched events.
  uint64_t high_water_buffer_bytes;
  // Total synchronous buffer bytes allocated by matched events.
  uint64_t total_buffer_allocate_bytes;
  // Total synchronous buffer bytes freed by matched events.
  uint64_t total_buffer_free_bytes;
} iree_profile_memory_device_t;

typedef enum iree_profile_memory_lifecycle_kind_e {
  // Slab-provider backing allocation lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB = 0,
  // Pool reservation lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION = 1,
  // Queue-visible transient allocation lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION = 2,
  // Direct synchronous HAL buffer lifecycle.
  IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION = 3,
} iree_profile_memory_lifecycle_kind_t;

typedef struct iree_profile_memory_pool_t {
  // Lifecycle kind responsible for this aggregate.
  iree_profile_memory_lifecycle_kind_t kind;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Producer-defined pool/provider identifier.
  uint64_t pool_id;
  // HAL memory type bits observed for this pool/provider.
  uint64_t memory_type;
  // HAL buffer usage bits observed for this pool/provider.
  uint64_t buffer_usage;
  // Matched events attributed to this pool/provider.
  uint64_t event_count;
  // Wait events attributed to this pool/provider.
  uint64_t wait_count;
  // Materialize events attributed to this pool/provider.
  uint64_t materialize_count;
  // Current live allocation/reservation/slab count.
  uint64_t current_allocation_count;
  // Maximum live allocation/reservation/slab count.
  uint64_t high_water_allocation_count;
  // Current live bytes in this pool/provider aggregate.
  uint64_t current_bytes;
  // Maximum live bytes in this pool/provider aggregate.
  uint64_t high_water_bytes;
  // Cumulative bytes acquired/allocated/reserved.
  uint64_t total_allocate_bytes;
  // Cumulative bytes released/freed/deallocated.
  uint64_t total_free_bytes;
} iree_profile_memory_pool_t;

typedef struct iree_profile_memory_allocation_t {
  // Lifecycle kind represented by this allocation row.
  iree_profile_memory_lifecycle_kind_t kind;
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Producer-defined allocation identifier.
  uint64_t allocation_id;
  // Producer-defined pool/provider identifier.
  uint64_t pool_id;
  // Producer-defined backing allocation or slab identifier.
  uint64_t backing_id;
  // HAL memory type bits observed for this allocation.
  uint64_t memory_type;
  // HAL buffer usage bits observed for this allocation.
  uint64_t buffer_usage;
  // First matched event id in this lifecycle.
  uint64_t first_event_id;
  // Last matched event id in this lifecycle.
  uint64_t last_event_id;
  // First matched event host timestamp.
  int64_t first_host_time_ns;
  // Last matched event host timestamp.
  int64_t last_host_time_ns;
  // First nonzero queue submission id associated with this allocation.
  uint64_t first_submission_id;
  // Last nonzero queue submission id associated with this allocation.
  uint64_t last_submission_id;
  // Number of matched events in this lifecycle.
  uint64_t event_count;
  // Wait events in this lifecycle.
  uint64_t wait_count;
  // Materialize events in this lifecycle.
  uint64_t materialize_count;
  // Current live bytes after matched events.
  uint64_t current_bytes;
  // Maximum live bytes observed after matched events.
  uint64_t high_water_bytes;
  // Cumulative bytes acquired/allocated/reserved.
  uint64_t total_allocate_bytes;
  // Cumulative bytes released/freed/deallocated.
  uint64_t total_free_bytes;
} iree_profile_memory_allocation_t;

typedef struct iree_profile_memory_context_t {
  // Host allocator used for dynamic memory summary arrays.
  iree_allocator_t host_allocator;
  // Dynamic array of per-device memory summaries.
  iree_profile_memory_device_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Dynamic array of per-pool/provider memory summaries.
  iree_profile_memory_pool_t* pools;
  // Number of valid entries in |pools|.
  iree_host_size_t pool_count;
  // Capacity of |pools| in entries.
  iree_host_size_t pool_capacity;
  // Dynamic array of per-allocation memory lifecycles.
  iree_profile_memory_allocation_t* allocations;
  // Number of valid entries in |allocations|.
  iree_host_size_t allocation_count;
  // Capacity of |allocations| in entries.
  iree_host_size_t allocation_capacity;
  // Total memory events parsed before filtering.
  uint64_t total_event_count;
  // Memory events matched by --id and --filter.
  uint64_t matched_event_count;
  // Matched memory events from truncated chunks.
  uint64_t truncated_event_count;
} iree_profile_memory_context_t;

typedef struct iree_profile_dispatch_export_t {
  // Producer-local executable identifier owning this export.
  uint64_t executable_id;
  // Export ordinal used by dispatch event records.
  uint32_t export_ordinal;
  // Number of HAL ABI constant words expected by this export.
  uint32_t constant_count;
  // Number of HAL ABI binding pointer slots expected by this export.
  uint32_t binding_count;
  // Number of reflected parameters associated with this export.
  uint32_t parameter_count;
  // Static workgroup size for each dispatch dimension.
  uint32_t workgroup_size[3];
  // Borrowed export name from the mapped profile bundle.
  iree_string_view_t name;
} iree_profile_dispatch_export_t;

typedef struct iree_profile_dispatch_executable_t {
  // Immutable executable metadata record borrowed from the profile bundle.
  iree_hal_profile_executable_record_t record;
} iree_profile_dispatch_executable_t;

typedef struct iree_profile_dispatch_command_buffer_t {
  // Immutable command-buffer metadata record borrowed from the profile bundle.
  iree_hal_profile_command_buffer_record_t record;
} iree_profile_dispatch_command_buffer_t;

typedef struct iree_profile_dispatch_command_operation_t {
  // Immutable command-operation metadata record copied from the profile bundle.
  iree_hal_profile_command_operation_record_t record;
} iree_profile_dispatch_command_operation_t;

typedef struct iree_profile_dispatch_queue_t {
  // Immutable queue metadata record borrowed from the profile bundle.
  iree_hal_profile_queue_record_t record;
} iree_profile_dispatch_queue_t;

typedef struct iree_profile_dispatch_queue_event_t {
  // Immutable queue event record copied from the profile bundle.
  iree_hal_profile_queue_event_t record;
} iree_profile_dispatch_queue_event_t;

typedef struct iree_profile_dispatch_device_t {
  // Session-local physical device ordinal.
  uint32_t physical_device_ordinal;
  // Number of clock-correlation samples seen for this physical device.
  uint64_t clock_sample_count;
  // First clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t first_clock_sample;
  // Last clock-correlation sample seen for this physical device.
  iree_hal_profile_clock_correlation_record_t last_clock_sample;
} iree_profile_dispatch_device_t;

typedef struct iree_profile_dispatch_aggregate_t {
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier for this aggregate row.
  uint64_t executable_id;
  // Export ordinal for this aggregate row.
  uint32_t export_ordinal;
  // Total dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start tick in this aggregate row.
  uint64_t earliest_start_tick;
  // Latest valid dispatch end tick in this aggregate row.
  uint64_t latest_end_tick;
  // Minimum valid dispatch duration in raw device ticks.
  uint64_t minimum_ticks;
  // Maximum valid dispatch duration in raw device ticks.
  uint64_t maximum_ticks;
  // Sum of valid dispatch durations in raw device ticks.
  double total_ticks;
  // Running mean of valid dispatch durations in raw device ticks.
  double mean_ticks;
  // Running sum of squares of differences from |mean_ticks|.
  double m2_ticks;
  // Last observed workgroup count for this dispatch key.
  uint32_t last_workgroup_count[3];
  // Last observed workgroup size for this dispatch key.
  uint32_t last_workgroup_size[3];
} iree_profile_dispatch_aggregate_t;

typedef struct iree_profile_dispatch_command_aggregate_t {
  // Producer-local command-buffer identifier for this aggregate row.
  uint64_t command_buffer_id;
  // Queue submission epoch containing this command-buffer execution.
  uint64_t submission_id;
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this aggregate row.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this aggregate row.
  uint64_t stream_id;
  // Total dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start tick in this aggregate row.
  uint64_t earliest_start_tick;
  // Latest valid dispatch end tick in this aggregate row.
  uint64_t latest_end_tick;
  // Sum of valid dispatch durations in raw device ticks.
  double total_ticks;
} iree_profile_dispatch_command_aggregate_t;

typedef struct iree_profile_dispatch_queue_aggregate_t {
  // Session-local physical device ordinal for this aggregate row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal for this aggregate row.
  uint32_t queue_ordinal;
  // Producer-defined stream identifier for this aggregate row.
  uint64_t stream_id;
  // Queue submission epoch for this aggregate row.
  uint64_t submission_id;
  // Total dispatch records matched for this aggregate row.
  uint64_t dispatch_count;
  // Dispatch records with valid start/end timestamps.
  uint64_t valid_count;
  // Dispatch records with missing or reversed timestamps.
  uint64_t invalid_count;
  // Earliest valid dispatch start tick in this aggregate row.
  uint64_t earliest_start_tick;
  // Latest valid dispatch end tick in this aggregate row.
  uint64_t latest_end_tick;
  // Sum of valid dispatch durations in raw device ticks.
  double total_ticks;
} iree_profile_dispatch_queue_aggregate_t;

typedef struct iree_profile_dispatch_context_t {
  // Host allocator used for dynamic side tables and aggregate rows.
  iree_allocator_t host_allocator;
  // Dynamic array of executable side-table entries.
  iree_profile_dispatch_executable_t* executables;
  // Number of valid entries in |executables|.
  iree_host_size_t executable_count;
  // Capacity of |executables| in entries.
  iree_host_size_t executable_capacity;
  // Dynamic array of executable export side-table entries.
  iree_profile_dispatch_export_t* exports;
  // Number of valid entries in |exports|.
  iree_host_size_t export_count;
  // Capacity of |exports| in entries.
  iree_host_size_t export_capacity;
  // Dynamic array of command-buffer side-table entries.
  iree_profile_dispatch_command_buffer_t* command_buffers;
  // Number of valid entries in |command_buffers|.
  iree_host_size_t command_buffer_count;
  // Capacity of |command_buffers| in entries.
  iree_host_size_t command_buffer_capacity;
  // Dynamic array of command-operation side-table entries.
  iree_profile_dispatch_command_operation_t* command_operations;
  // Number of valid entries in |command_operations|.
  iree_host_size_t command_operation_count;
  // Capacity of |command_operations| in entries.
  iree_host_size_t command_operation_capacity;
  // Dynamic array of queue side-table entries.
  iree_profile_dispatch_queue_t* queues;
  // Number of valid entries in |queues|.
  iree_host_size_t queue_count;
  // Capacity of |queues| in entries.
  iree_host_size_t queue_capacity;
  // Dynamic array of per-physical-device clock samples.
  iree_profile_dispatch_device_t* devices;
  // Number of valid entries in |devices|.
  iree_host_size_t device_count;
  // Capacity of |devices| in entries.
  iree_host_size_t device_capacity;
  // Dynamic array of aggregate dispatch rows.
  iree_profile_dispatch_aggregate_t* aggregates;
  // Number of valid entries in |aggregates|.
  iree_host_size_t aggregate_count;
  // Capacity of |aggregates| in entries.
  iree_host_size_t aggregate_capacity;
  // Dynamic array of command-buffer execution aggregate rows.
  iree_profile_dispatch_command_aggregate_t* command_aggregates;
  // Number of valid entries in |command_aggregates|.
  iree_host_size_t command_aggregate_count;
  // Capacity of |command_aggregates| in entries.
  iree_host_size_t command_aggregate_capacity;
  // Dynamic array of queue submission aggregate rows.
  iree_profile_dispatch_queue_aggregate_t* queue_aggregates;
  // Number of valid entries in |queue_aggregates|.
  iree_host_size_t queue_aggregate_count;
  // Capacity of |queue_aggregates| in entries.
  iree_host_size_t queue_aggregate_capacity;
  // Dynamic array of queue operation event rows.
  iree_profile_dispatch_queue_event_t* queue_events;
  // Number of valid entries in |queue_events|.
  iree_host_size_t queue_event_count;
  // Capacity of |queue_events| in entries.
  iree_host_size_t queue_event_capacity;
  // Total queue operation records parsed before filtering.
  uint64_t total_queue_event_count;
  // Queue operation records matched by the active filter.
  uint64_t matched_queue_event_count;
  // Total dispatch records parsed before filtering.
  uint64_t total_dispatch_count;
  // Dispatch records matched by the active filter.
  uint64_t matched_dispatch_count;
  // Matched dispatch records with valid timestamps.
  uint64_t valid_dispatch_count;
  // Matched dispatch records with missing or reversed timestamps.
  uint64_t invalid_dispatch_count;
} iree_profile_dispatch_context_t;

static const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      return "session_begin";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      return "chunk";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      return "session_end";
    default:
      return "unknown";
  }
}

static void iree_profile_fprint_json_string(FILE* file,
                                            iree_string_view_t value) {
  fputc('"', file);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (c < 0x20) {
          fprintf(file, "\\u%04x", c);
        } else {
          fputc(c, file);
        }
        break;
    }
  }
  fputc('"', file);
}

static void iree_profile_dump_header_text(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file, "IREE HAL profile bundle\n");
  fprintf(file, "version: %u.%u\n", header->version_major,
          header->version_minor);
  fprintf(file, "header_length: %u\n", header->header_length);
  fprintf(file, "flags: 0x%08x\n", header->flags);
  fprintf(file, "records:\n");
}

static void iree_profile_dump_record_text(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file,
          "[%" PRIhsz "] %s record_length=%" PRIu64 " payload_length=%" PRIu64
          "\n",
          record_index, iree_profile_record_type_name(header->record_type),
          header->record_length, header->payload_length);
  fprintf(file, "  content_type: %.*s\n", (int)record->content_type.size,
          record->content_type.data);
  fprintf(file, "  name: %.*s\n", (int)record->name.size, record->name.data);
  fprintf(file,
          "  session_id=%" PRIu64 " stream_id=%" PRIu64 " event_id=%" PRIu64
          "\n",
          header->session_id, header->stream_id, header->event_id);
  fprintf(file, "  executable_id=%" PRIu64 " command_buffer_id=%" PRIu64 "\n",
          header->executable_id, header->command_buffer_id);
  fprintf(file, "  physical_device_ordinal=%u queue_ordinal=%u\n",
          header->physical_device_ordinal, header->queue_ordinal);
  fprintf(file, "  chunk_flags=0x%016" PRIx64 " session_status_code=%u\n",
          header->chunk_flags, header->session_status_code);
}

static void iree_profile_dump_header_jsonl(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file,
          "{\"type\":\"file\",\"magic\":\"IRPF\","
          "\"version_major\":%u,\"version_minor\":%u,"
          "\"header_length\":%u,\"flags\":%u}\n",
          header->version_major, header->version_minor, header->header_length,
          header->flags);
}

static void iree_profile_dump_record_jsonl(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file, "{\"type\":\"record\",\"index\":%" PRIhsz, record_index);
  fprintf(file, ",\"record_type\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_record_type_name(header->record_type)));
  fprintf(file, ",\"record_type_value\":%u", header->record_type);
  fprintf(file, ",\"record_length\":%" PRIu64, header->record_length);
  fprintf(file, ",\"payload_length\":%" PRIu64, header->payload_length);
  fprintf(file, ",\"content_type\":");
  iree_profile_fprint_json_string(file, record->content_type);
  fprintf(file, ",\"name\":");
  iree_profile_fprint_json_string(file, record->name);
  fprintf(file, ",\"session_id\":%" PRIu64, header->session_id);
  fprintf(file, ",\"stream_id\":%" PRIu64, header->stream_id);
  fprintf(file, ",\"event_id\":%" PRIu64, header->event_id);
  fprintf(file, ",\"executable_id\":%" PRIu64, header->executable_id);
  fprintf(file, ",\"command_buffer_id\":%" PRIu64, header->command_buffer_id);
  fprintf(file, ",\"physical_device_ordinal\":%u",
          header->physical_device_ordinal);
  fprintf(file, ",\"queue_ordinal\":%u", header->queue_ordinal);
  fprintf(file, ",\"chunk_flags\":%" PRIu64, header->chunk_flags);
  fprintf(file, ",\"session_status_code\":%u", header->session_status_code);
  fputs("}\n", file);
}

static void iree_profile_summary_initialize(
    iree_allocator_t host_allocator, iree_profile_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  out_summary->host_allocator = host_allocator;
}

static void iree_profile_summary_deinitialize(iree_profile_summary_t* summary) {
  iree_allocator_free(summary->host_allocator, summary->devices);
  memset(summary, 0, sizeof(*summary));
}

static iree_status_t iree_profile_summary_get_device(
    iree_profile_summary_t* summary, uint32_t physical_device_ordinal,
    iree_profile_device_summary_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    if (summary->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &summary->devices[i];
      return iree_ok_status();
    }
  }

  if (summary->device_count + 1 > summary->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        summary->host_allocator,
        iree_max((iree_host_size_t)4, summary->device_count + 1),
        sizeof(summary->devices[0]), &summary->device_capacity,
        (void**)&summary->devices));
  }

  iree_profile_device_summary_t* device =
      &summary->devices[summary->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  device->minimum_clock_uncertainty_ns = INT64_MAX;
  device->earliest_dispatch_start_tick = UINT64_MAX;
  device->minimum_dispatch_ticks = UINT64_MAX;
  *out_device = device;
  return iree_ok_status();
}

static iree_status_t iree_profile_payload_record_length(
    iree_string_view_t content_type, iree_const_byte_span_t payload,
    iree_host_size_t payload_offset, iree_host_size_t minimum_record_length,
    iree_host_size_t* out_record_length) {
  *out_record_length = 0;

  const iree_host_size_t remaining_length =
      payload.data_length - payload_offset;
  if (remaining_length < minimum_record_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "profile chunk '%.*s' has a truncated typed record",
                            (int)content_type.size, content_type.data);
  }

  uint32_t record_length = 0;
  memcpy(&record_length, payload.data + payload_offset, sizeof(record_length));
  if (record_length < minimum_record_length ||
      record_length > remaining_length) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "profile chunk '%.*s' has invalid typed record length %u",
        (int)content_type.size, content_type.data, record_length);
  }

  *out_record_length = record_length;
  return iree_ok_status();
}

static double iree_profile_sqrt_f64(double value) {
  if (value <= 0.0) return 0.0;
  // Keep this standalone C tool free of libm linkage.
  double estimate = value >= 1.0 ? value : 1.0;
  for (int i = 0; i < 32; ++i) {
    estimate = 0.5 * (estimate + value / estimate);
  }
  return estimate;
}

static void iree_profile_dispatch_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_dispatch_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

static void iree_profile_dispatch_context_deinitialize(
    iree_profile_dispatch_context_t* context) {
  iree_allocator_free(context->host_allocator, context->queue_events);
  iree_allocator_free(context->host_allocator, context->queue_aggregates);
  iree_allocator_free(context->host_allocator, context->command_aggregates);
  iree_allocator_free(context->host_allocator, context->aggregates);
  iree_allocator_free(context->host_allocator, context->devices);
  iree_allocator_free(context->host_allocator, context->queues);
  iree_allocator_free(context->host_allocator, context->command_operations);
  iree_allocator_free(context->host_allocator, context->command_buffers);
  iree_allocator_free(context->host_allocator, context->exports);
  iree_allocator_free(context->host_allocator, context->executables);
  memset(context, 0, sizeof(*context));
}

static iree_status_t iree_profile_dispatch_get_device(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    iree_profile_dispatch_device_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &context->devices[i];
      return iree_ok_status();
    }
  }

  if (context->device_count + 1 > context->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->device_count + 1),
        sizeof(context->devices[0]), &context->device_capacity,
        (void**)&context->devices));
  }

  iree_profile_dispatch_device_t* device =
      &context->devices[context->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static void iree_profile_dispatch_record_clock_sample(
    iree_profile_dispatch_device_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;
}

static const iree_profile_dispatch_queue_t* iree_profile_dispatch_find_queue(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    uint64_t stream_id) {
  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_profile_dispatch_queue_t* queue_info = &context->queues[i];
    if (queue_info->record.physical_device_ordinal == physical_device_ordinal &&
        queue_info->record.queue_ordinal == queue_ordinal &&
        queue_info->record.stream_id == stream_id) {
      return queue_info;
    }
  }
  return NULL;
}

static const iree_profile_dispatch_executable_t*
iree_profile_dispatch_find_executable(
    const iree_profile_dispatch_context_t* context, uint64_t executable_id) {
  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_profile_dispatch_executable_t* executable_info =
        &context->executables[i];
    if (executable_info->record.executable_id == executable_id) {
      return executable_info;
    }
  }
  return NULL;
}

static const iree_profile_dispatch_command_buffer_t*
iree_profile_dispatch_find_command_buffer(
    const iree_profile_dispatch_context_t* context,
    uint64_t command_buffer_id) {
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_profile_dispatch_command_buffer_t* command_buffer_info =
        &context->command_buffers[i];
    if (command_buffer_info->record.command_buffer_id == command_buffer_id) {
      return command_buffer_info;
    }
  }
  return NULL;
}

static bool iree_profile_dispatch_device_try_fit_clock(
    const iree_profile_dispatch_device_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  if (!device || device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(
          first->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      !iree_all_bits_set(
          last->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP)) {
    return false;
  }
  if (last->device_tick <= first->device_tick ||
      last->host_cpu_timestamp_ns <= first->host_cpu_timestamp_ns) {
    return false;
  }

  const double device_delta_ticks =
      (double)(last->device_tick - first->device_tick);
  const double host_delta_ns =
      (double)(last->host_cpu_timestamp_ns - first->host_cpu_timestamp_ns);
  *out_ns_per_tick = host_delta_ns / device_delta_ticks;
  *out_tick_frequency_hz = 1000000000.0 / *out_ns_per_tick;
  return true;
}

static const iree_profile_dispatch_export_t* iree_profile_dispatch_find_export(
    const iree_profile_dispatch_context_t* context, uint64_t executable_id,
    uint32_t export_ordinal) {
  for (iree_host_size_t i = 0; i < context->export_count; ++i) {
    const iree_profile_dispatch_export_t* export_info = &context->exports[i];
    if (export_info->executable_id == executable_id &&
        export_info->export_ordinal == export_ordinal) {
      return export_info;
    }
  }
  return NULL;
}

static iree_string_view_t iree_profile_dispatch_format_numeric_key(
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* buffer, iree_host_size_t buffer_capacity) {
  if (buffer_capacity == 0) return iree_string_view_empty();
  int result = 0;
  if (physical_device_ordinal == UINT32_MAX) {
    result = snprintf(buffer, buffer_capacity, "executable%" PRIu64 "#%u",
                      executable_id, export_ordinal);
  } else {
    result =
        snprintf(buffer, buffer_capacity, "device%u/executable%" PRIu64 "#%u",
                 physical_device_ordinal, executable_id, export_ordinal);
  }
  if (result < 0) return iree_string_view_empty();
  iree_host_size_t length = (iree_host_size_t)result;
  if (length >= buffer_capacity) length = buffer_capacity - 1;
  return iree_make_string_view(buffer, length);
}

static iree_string_view_t iree_profile_dispatch_format_export_key(
    const iree_profile_dispatch_export_t* export_info,
    uint32_t physical_device_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity) {
  if (!iree_string_view_is_empty(export_info->name)) {
    return export_info->name;
  }
  return iree_profile_dispatch_format_numeric_key(
      physical_device_ordinal, export_info->executable_id,
      export_info->export_ordinal, numeric_buffer, numeric_buffer_capacity);
}

static iree_status_t iree_profile_dispatch_resolve_key(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal, uint64_t executable_id,
    uint32_t export_ordinal, char* numeric_buffer,
    iree_host_size_t numeric_buffer_capacity, iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  const iree_profile_dispatch_executable_t* executable_info =
      iree_profile_dispatch_find_executable(context, executable_id);
  if (!executable_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  const iree_profile_dispatch_export_t* export_info =
      iree_profile_dispatch_find_export(context, executable_id, export_ordinal);
  if (!export_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing executable export metadata "
        "device=%u executable=%" PRIu64 " export=%u",
        physical_device_ordinal, executable_id, export_ordinal);
  }
  if (!iree_string_view_is_empty(export_info->name)) {
    *out_key = export_info->name;
  } else {
    *out_key = iree_profile_dispatch_format_export_key(
        export_info, physical_device_ordinal, numeric_buffer,
        numeric_buffer_capacity);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_validate_event_metadata(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record,
    const iree_hal_profile_dispatch_event_t* event,
    iree_profile_projection_mode_t projection_mode) {
  const iree_profile_dispatch_queue_t* queue_info =
      iree_profile_dispatch_find_queue(
          context, record->header.physical_device_ordinal,
          record->header.queue_ordinal, record->header.stream_id);
  if (!queue_info) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing queue metadata "
        "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
        record->header.physical_device_ordinal, record->header.queue_ordinal,
        record->header.stream_id, event->submission_id);
  }
  if (projection_mode == IREE_PROFILE_PROJECTION_MODE_COMMAND &&
      event->command_buffer_id != 0 &&
      !iree_profile_dispatch_find_command_buffer(context,
                                                 event->command_buffer_id)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "dispatch event references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " submission=%" PRIu64,
        event->command_buffer_id, event->submission_id);
  }
  return iree_ok_status();
}

static bool iree_profile_dispatch_key_matches(iree_string_view_t key,
                                              iree_string_view_t filter) {
  if (iree_string_view_is_empty(filter) ||
      iree_string_view_equal(filter, IREE_SV("*"))) {
    return true;
  }
  return iree_string_view_match_pattern(key, filter);
}

static bool iree_profile_dispatch_event_matches_id(
    const iree_hal_profile_dispatch_event_t* event,
    iree_profile_projection_mode_t mode, int64_t id_filter) {
  if (id_filter < 0) return true;
  const uint64_t id = (uint64_t)id_filter;
  switch (mode) {
    case IREE_PROFILE_PROJECTION_MODE_DISPATCH:
      return event->event_id == id;
    case IREE_PROFILE_PROJECTION_MODE_EXECUTABLE:
      return event->executable_id == id;
    case IREE_PROFILE_PROJECTION_MODE_COMMAND:
      return event->command_buffer_id == id;
    case IREE_PROFILE_PROJECTION_MODE_QUEUE:
      return event->submission_id == id;
    default:
      return false;
  }
}

static iree_status_t iree_profile_dispatch_append_export(
    iree_profile_dispatch_context_t* context,
    const iree_profile_dispatch_export_t* export_info) {
  if (context->export_count + 1 > context->export_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->export_count + 1),
        sizeof(context->exports[0]), &context->export_capacity,
        (void**)&context->exports));
  }
  context->exports[context->export_count++] = *export_info;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_executable(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_executable_record_t* record) {
  if (context->executable_count + 1 > context->executable_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->executable_count + 1),
        sizeof(context->executables[0]), &context->executable_capacity,
        (void**)&context->executables));
  }
  iree_profile_dispatch_executable_t* executable_info =
      &context->executables[context->executable_count++];
  executable_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_command_buffer(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_buffer_record_t* record) {
  if (context->command_buffer_count + 1 > context->command_buffer_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->command_buffer_count + 1),
        sizeof(context->command_buffers[0]), &context->command_buffer_capacity,
        (void**)&context->command_buffers));
  }
  iree_profile_dispatch_command_buffer_t* command_buffer_info =
      &context->command_buffers[context->command_buffer_count++];
  command_buffer_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_command_operation(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* record) {
  if (context->command_operation_count + 1 >
      context->command_operation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)64, context->command_operation_count + 1),
        sizeof(context->command_operations[0]),
        &context->command_operation_capacity,
        (void**)&context->command_operations));
  }
  iree_profile_dispatch_command_operation_t* operation_info =
      &context->command_operations[context->command_operation_count++];
  operation_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_queue(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* record) {
  if (context->queue_count + 1 > context->queue_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->queue_count + 1),
        sizeof(context->queues[0]), &context->queue_capacity,
        (void**)&context->queues));
  }
  iree_profile_dispatch_queue_t* queue_info =
      &context->queues[context->queue_count++];
  queue_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_append_queue_event(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_event_t* record) {
  if (context->queue_event_count + 1 > context->queue_event_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)64, context->queue_event_count + 1),
        sizeof(context->queue_events[0]), &context->queue_event_capacity,
        (void**)&context->queue_events));
  }
  iree_profile_dispatch_queue_event_t* event_info =
      &context->queue_events[context->queue_event_count++];
  event_info->record = *record;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint64_t executable_id, uint32_t export_ordinal,
    iree_profile_dispatch_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    iree_profile_dispatch_aggregate_t* aggregate = &context->aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->executable_id == executable_id &&
        aggregate->export_ordinal == export_ordinal) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->aggregate_count + 1 > context->aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->aggregate_count + 1),
        sizeof(context->aggregates[0]), &context->aggregate_capacity,
        (void**)&context->aggregates));
  }

  iree_profile_dispatch_aggregate_t* aggregate =
      &context->aggregates[context->aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->executable_id = executable_id;
  aggregate->export_ordinal = export_ordinal;
  aggregate->earliest_start_tick = UINT64_MAX;
  aggregate->minimum_ticks = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_command_aggregate(
    iree_profile_dispatch_context_t* context, uint64_t command_buffer_id,
    uint64_t submission_id, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id,
    iree_profile_dispatch_command_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->command_aggregate_count; ++i) {
    iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (aggregate->command_buffer_id == command_buffer_id &&
        aggregate->submission_id == submission_id &&
        aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->command_aggregate_count + 1 >
      context->command_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->command_aggregate_count + 1),
        sizeof(context->command_aggregates[0]),
        &context->command_aggregate_capacity,
        (void**)&context->command_aggregates));
  }

  iree_profile_dispatch_command_aggregate_t* aggregate =
      &context->command_aggregates[context->command_aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->command_buffer_id = command_buffer_id;
  aggregate->submission_id = submission_id;
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_get_queue_aggregate(
    iree_profile_dispatch_context_t* context, uint32_t physical_device_ordinal,
    uint32_t queue_ordinal, uint64_t stream_id, uint64_t submission_id,
    iree_profile_dispatch_queue_aggregate_t** out_aggregate) {
  *out_aggregate = NULL;

  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal &&
        aggregate->queue_ordinal == queue_ordinal &&
        aggregate->stream_id == stream_id &&
        aggregate->submission_id == submission_id) {
      *out_aggregate = aggregate;
      return iree_ok_status();
    }
  }

  if (context->queue_aggregate_count + 1 > context->queue_aggregate_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->queue_aggregate_count + 1),
        sizeof(context->queue_aggregates[0]),
        &context->queue_aggregate_capacity,
        (void**)&context->queue_aggregates));
  }

  iree_profile_dispatch_queue_aggregate_t* aggregate =
      &context->queue_aggregates[context->queue_aggregate_count++];
  memset(aggregate, 0, sizeof(*aggregate));
  aggregate->physical_device_ordinal = physical_device_ordinal;
  aggregate->queue_ordinal = queue_ordinal;
  aggregate->stream_id = stream_id;
  aggregate->submission_id = submission_id;
  aggregate->earliest_start_tick = UINT64_MAX;
  *out_aggregate = aggregate;
  return iree_ok_status();
}

static void iree_profile_dispatch_record_aggregate_event(
    iree_profile_dispatch_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  memcpy(aggregate->last_workgroup_count, event->workgroup_count,
         sizeof(aggregate->last_workgroup_count));
  memcpy(aggregate->last_workgroup_size, event->workgroup_size,
         sizeof(aggregate->last_workgroup_size));

  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }

  const uint64_t duration_ticks = event->end_tick - event->start_tick;
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->minimum_ticks = iree_min(aggregate->minimum_ticks, duration_ticks);
  aggregate->maximum_ticks = iree_max(aggregate->maximum_ticks, duration_ticks);
  aggregate->total_ticks += (double)duration_ticks;

  const double duration = (double)duration_ticks;
  const double delta = duration - aggregate->mean_ticks;
  aggregate->mean_ticks += delta / (double)aggregate->valid_count;
  const double delta2 = duration - aggregate->mean_ticks;
  aggregate->m2_ticks += delta * delta2;
}

static void iree_profile_dispatch_record_command_aggregate_event(
    iree_profile_dispatch_command_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->total_ticks += (double)(event->end_tick - event->start_tick);
}

static void iree_profile_dispatch_record_queue_aggregate_event(
    iree_profile_dispatch_queue_aggregate_t* aggregate,
    const iree_hal_profile_dispatch_event_t* event) {
  ++aggregate->dispatch_count;
  if (event->start_tick == 0 || event->end_tick == 0 ||
      event->end_tick < event->start_tick) {
    ++aggregate->invalid_count;
    return;
  }
  ++aggregate->valid_count;
  aggregate->earliest_start_tick =
      iree_min(aggregate->earliest_start_tick, event->start_tick);
  aggregate->latest_end_tick =
      iree_max(aggregate->latest_end_tick, event->end_tick);
  aggregate->total_ticks += (double)(event->end_tick - event->start_tick);
}

static void iree_profile_summary_record_clock_sample(
    iree_profile_device_summary_t* device,
    const iree_hal_profile_clock_correlation_record_t* record) {
  if (device->clock_sample_count == 0) {
    device->first_clock_sample = *record;
  }
  device->last_clock_sample = *record;
  ++device->clock_sample_count;

  if (iree_all_bits_set(
          record->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET) &&
      record->host_time_end_ns >= record->host_time_begin_ns) {
    const int64_t uncertainty_ns =
        record->host_time_end_ns - record->host_time_begin_ns;
    device->minimum_clock_uncertainty_ns =
        iree_min(device->minimum_clock_uncertainty_ns, uncertainty_ns);
    device->maximum_clock_uncertainty_ns =
        iree_max(device->maximum_clock_uncertainty_ns, uncertainty_ns);
  }
}

static void iree_profile_summary_record_dispatch_event(
    iree_profile_device_summary_t* device,
    const iree_hal_profile_dispatch_event_t* record) {
  ++device->dispatch_event_count;
  if (record->start_tick == 0 || record->end_tick == 0 ||
      record->end_tick < record->start_tick) {
    ++device->invalid_dispatch_event_count;
    return;
  }

  const uint64_t duration_ticks = record->end_tick - record->start_tick;
  device->total_dispatch_ticks += (double)duration_ticks;
  device->earliest_dispatch_start_tick =
      iree_min(device->earliest_dispatch_start_tick, record->start_tick);
  device->latest_dispatch_end_tick =
      iree_max(device->latest_dispatch_end_tick, record->end_tick);
  device->minimum_dispatch_ticks =
      iree_min(device->minimum_dispatch_ticks, duration_ticks);
  device->maximum_dispatch_ticks =
      iree_max(device->maximum_dispatch_ticks, duration_ticks);
}

static iree_status_t iree_profile_summary_process_device_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_device_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_device_record_t device_record;
      memcpy(&device_record, record->payload.data + payload_offset,
             sizeof(device_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, device_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        ++device->device_record_count;
        device->queue_count = device_record.queue_count;
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_queue_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_record_t queue_record;
      memcpy(&queue_record, record->payload.data + payload_offset,
             sizeof(queue_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, queue_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        ++device->queue_record_count;
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->executable_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_executable_export_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_export_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->executable_export_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_command_buffer_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_buffer_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->command_buffer_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_command_operation_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_operation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->command_operation_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_clock_correlation_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_clock_correlation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_clock_correlation_record_t clock_record;
      memcpy(&clock_record, record->payload.data + payload_offset,
             sizeof(clock_record));

      iree_profile_device_summary_t* device = NULL;
      status = iree_profile_summary_get_device(
          summary, clock_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        iree_profile_summary_record_clock_sample(device, &clock_record);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_dispatch_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_profile_device_summary_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_summary_get_device(
      summary, record->header.physical_device_ordinal, &device));

  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_dispatch_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_dispatch_event_t dispatch_record;
      memcpy(&dispatch_record, record->payload.data + payload_offset,
             sizeof(dispatch_record));
      iree_profile_summary_record_dispatch_event(device, &dispatch_record);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_memory_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_memory_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->memory_event_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_queue_event_records(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      ++summary->queue_event_record_count;
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_summary_process_record(
    iree_profile_summary_t* summary,
    const iree_hal_profile_file_record_t* record) {
  ++summary->file_record_count;

  switch (record->header.record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      ++summary->session_begin_count;
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      ++summary->session_end_count;
      return iree_ok_status();
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      ++summary->chunk_count;
      break;
    default:
      ++summary->unknown_record_count;
      return iree_ok_status();
  }

  if (iree_any_bit_set(record->header.chunk_flags,
                       IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
    ++summary->truncated_chunk_count;
  }

  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    ++summary->device_chunk_count;
    return iree_profile_summary_process_device_records(summary, record);
  } else if (iree_string_view_equal(record->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    ++summary->queue_chunk_count;
    return iree_profile_summary_process_queue_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    ++summary->executable_chunk_count;
    return iree_profile_summary_process_executable_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    ++summary->executable_export_chunk_count;
    return iree_profile_summary_process_executable_export_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    ++summary->command_buffer_chunk_count;
    return iree_profile_summary_process_command_buffer_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    ++summary->command_operation_chunk_count;
    return iree_profile_summary_process_command_operation_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    ++summary->clock_correlation_chunk_count;
    return iree_profile_summary_process_clock_correlation_records(summary,
                                                                  record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    ++summary->dispatch_event_chunk_count;
    return iree_profile_summary_process_dispatch_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    ++summary->queue_event_chunk_count;
    return iree_profile_summary_process_queue_event_records(summary, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    ++summary->memory_event_chunk_count;
    return iree_profile_summary_process_memory_event_records(summary, record);
  }

  ++summary->unknown_chunk_count;
  return iree_ok_status();
}

static bool iree_profile_device_summary_try_fit_clock(
    const iree_profile_device_summary_t* device, double* out_ns_per_tick,
    double* out_tick_frequency_hz) {
  *out_ns_per_tick = 0.0;
  *out_tick_frequency_hz = 0.0;
  if (device->clock_sample_count < 2) return false;

  const iree_hal_profile_clock_correlation_record_t* first =
      &device->first_clock_sample;
  const iree_hal_profile_clock_correlation_record_t* last =
      &device->last_clock_sample;
  if (!iree_all_bits_set(
          first->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP) ||
      !iree_all_bits_set(
          last->flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP)) {
    return false;
  }
  if (last->device_tick <= first->device_tick ||
      last->host_cpu_timestamp_ns <= first->host_cpu_timestamp_ns) {
    return false;
  }

  const double device_delta_ticks =
      (double)(last->device_tick - first->device_tick);
  const double host_delta_ns =
      (double)(last->host_cpu_timestamp_ns - first->host_cpu_timestamp_ns);
  *out_ns_per_tick = host_delta_ns / device_delta_ticks;
  *out_tick_frequency_hz = 1000000000.0 / *out_ns_per_tick;
  return true;
}

static bool iree_profile_device_summary_clock_covers_dispatches(
    const iree_profile_device_summary_t* device) {
  const uint64_t valid_dispatch_count =
      device->dispatch_event_count - device->invalid_dispatch_event_count;
  if (device->clock_sample_count < 2 || valid_dispatch_count == 0) {
    return false;
  }
  if (!iree_all_bits_set(device->first_clock_sample.flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK) ||
      !iree_all_bits_set(device->last_clock_sample.flags,
                         IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
    return false;
  }
  return device->first_clock_sample.device_tick <=
             device->earliest_dispatch_start_tick &&
         device->latest_dispatch_end_tick <=
             device->last_clock_sample.device_tick;
}

static void iree_profile_print_summary_text(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(file, "IREE HAL profile summary\n");
  fprintf(file,
          "records: file=%" PRIu64 " session_begin=%" PRIu64 " chunks=%" PRIu64
          " session_end=%" PRIu64 " unknown=%" PRIu64 "\n",
          summary->file_record_count, summary->session_begin_count,
          summary->chunk_count, summary->session_end_count,
          summary->unknown_record_count);
  fprintf(
      file,
      "chunks: devices=%" PRIu64 " queues=%" PRIu64 " executables=%" PRIu64
      " executable_exports=%" PRIu64 " command_buffers=%" PRIu64
      " command_operations=%" PRIu64 " clock_correlations=%" PRIu64
      " dispatch_events=%" PRIu64 " queue_events=%" PRIu64
      " memory_events=%" PRIu64 " unknown=%" PRIu64 " truncated=%" PRIu64 "\n",
      summary->device_chunk_count, summary->queue_chunk_count,
      summary->executable_chunk_count, summary->executable_export_chunk_count,
      summary->command_buffer_chunk_count,
      summary->command_operation_chunk_count,
      summary->clock_correlation_chunk_count,
      summary->dispatch_event_chunk_count, summary->queue_event_chunk_count,
      summary->memory_event_chunk_count, summary->unknown_chunk_count,
      summary->truncated_chunk_count);
  fprintf(
      file,
      "metadata_records: executables=%" PRIu64 " executable_exports=%" PRIu64
      " command_buffers=%" PRIu64 " command_operations=%" PRIu64 "\n",
      summary->executable_record_count, summary->executable_export_record_count,
      summary->command_buffer_record_count,
      summary->command_operation_record_count);
  fprintf(file,
          "event_records: queue_events=%" PRIu64 " memory_events=%" PRIu64 "\n",
          summary->queue_event_record_count,
          summary->memory_event_record_count);
  fprintf(file, "devices:\n");

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    const iree_profile_device_summary_t* device = &summary->devices[i];
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_device_summary_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const bool clock_covers_dispatches =
        iree_profile_device_summary_clock_covers_dispatches(device);
    const uint64_t valid_dispatch_count =
        device->dispatch_event_count - device->invalid_dispatch_event_count;

    fprintf(file, "  device[%u]: device_records=%u queues=%u/%u\n",
            device->physical_device_ordinal, device->device_record_count,
            device->queue_record_count, device->queue_count);
    fprintf(file,
            "    clock_samples=%" PRIu64 " min_uncertainty_ns=%" PRId64
            " max_uncertainty_ns=%" PRId64 "\n",
            device->clock_sample_count,
            device->minimum_clock_uncertainty_ns == INT64_MAX
                ? 0
                : device->minimum_clock_uncertainty_ns,
            device->maximum_clock_uncertainty_ns);
    if (has_clock_fit) {
      fprintf(file,
              "    clock_fit: ns_per_tick=%.9f tick_frequency_hz=%.3f"
              " device_delta_ticks=%" PRIu64 " host_delta_ns=%" PRIu64 "\n",
              ns_per_tick, tick_frequency_hz,
              device->last_clock_sample.device_tick -
                  device->first_clock_sample.device_tick,
              device->last_clock_sample.host_cpu_timestamp_ns -
                  device->first_clock_sample.host_cpu_timestamp_ns);
    } else {
      fprintf(file, "    clock_fit: unavailable\n");
    }

    fprintf(file,
            "    dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
            "\n",
            device->dispatch_event_count, valid_dispatch_count,
            device->invalid_dispatch_event_count);
    if (valid_dispatch_count != 0) {
      const double average_ticks =
          device->total_dispatch_ticks / (double)valid_dispatch_count;
      fprintf(file,
              "    dispatch_tick_range: start=%" PRIu64 " end=%" PRIu64
              " covered_by_clock_samples=%s\n",
              device->earliest_dispatch_start_tick,
              device->latest_dispatch_end_tick,
              clock_covers_dispatches ? "true" : "false");
      fprintf(file,
              "    dispatch_ticks: min=%" PRIu64 " avg=%.3f max=%" PRIu64
              " total=%.3f\n",
              device->minimum_dispatch_ticks, average_ticks,
              device->maximum_dispatch_ticks, device->total_dispatch_ticks);
      if (has_clock_fit) {
        fprintf(file,
                "    dispatch_time_ns: min=%.3f avg=%.3f max=%.3f"
                " total=%.3f\n",
                (double)device->minimum_dispatch_ticks * ns_per_tick,
                average_ticks * ns_per_tick,
                (double)device->maximum_dispatch_ticks * ns_per_tick,
                device->total_dispatch_ticks * ns_per_tick);
      }
    }
  }
}

static void iree_profile_print_summary_jsonl(
    const iree_profile_summary_t* summary, FILE* file) {
  fprintf(
      file,
      "{\"type\":\"summary\",\"file_records\":%" PRIu64
      ",\"session_begin_records\":%" PRIu64 ",\"chunk_records\":%" PRIu64
      ",\"session_end_records\":%" PRIu64 ",\"unknown_records\":%" PRIu64
      ",\"device_chunks\":%" PRIu64 ",\"queue_chunks\":%" PRIu64
      ",\"executable_chunks\":%" PRIu64 ",\"executable_records\":%" PRIu64
      ",\"executable_export_chunks\":%" PRIu64
      ",\"executable_export_records\":%" PRIu64
      ",\"command_buffer_chunks\":%" PRIu64
      ",\"command_buffer_records\":%" PRIu64
      ",\"command_operation_chunks\":%" PRIu64
      ",\"command_operation_records\":%" PRIu64
      ",\"clock_correlation_chunks\":%" PRIu64
      ",\"dispatch_event_chunks\":%" PRIu64 ",\"queue_event_chunks\":%" PRIu64
      ",\"queue_event_records\":%" PRIu64 ",\"memory_event_chunks\":%" PRIu64
      ",\"memory_event_records\":%" PRIu64 ",\"unknown_chunks\":%" PRIu64
      ",\"truncated_chunks\":%" PRIu64 "}\n",
      summary->file_record_count, summary->session_begin_count,
      summary->chunk_count, summary->session_end_count,
      summary->unknown_record_count, summary->device_chunk_count,
      summary->queue_chunk_count, summary->executable_chunk_count,
      summary->executable_record_count, summary->executable_export_chunk_count,
      summary->executable_export_record_count,
      summary->command_buffer_chunk_count, summary->command_buffer_record_count,
      summary->command_operation_chunk_count,
      summary->command_operation_record_count,
      summary->clock_correlation_chunk_count,
      summary->dispatch_event_chunk_count, summary->queue_event_chunk_count,
      summary->queue_event_record_count, summary->memory_event_chunk_count,
      summary->memory_event_record_count, summary->unknown_chunk_count,
      summary->truncated_chunk_count);

  for (iree_host_size_t i = 0; i < summary->device_count; ++i) {
    const iree_profile_device_summary_t* device = &summary->devices[i];
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_device_summary_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const bool clock_covers_dispatches =
        iree_profile_device_summary_clock_covers_dispatches(device);
    const uint64_t valid_dispatch_count =
        device->dispatch_event_count - device->invalid_dispatch_event_count;
    const double average_ticks =
        valid_dispatch_count
            ? device->total_dispatch_ticks / (double)valid_dispatch_count
            : 0.0;
    fprintf(file,
            "{\"type\":\"device_summary\",\"physical_device_ordinal\":%u"
            ",\"device_records\":%u,\"queue_records\":%u,\"queues\":%u"
            ",\"clock_samples\":%" PRIu64
            ",\"clock_fit_available\":%s"
            ",\"ns_per_tick\":%.9f,\"tick_frequency_hz\":%.3f"
            ",\"min_clock_uncertainty_ns\":%" PRId64
            ",\"max_clock_uncertainty_ns\":%" PRId64 ",\"dispatches\":%" PRIu64
            ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
            ",\"min_dispatch_ticks\":%" PRIu64
            ",\"avg_dispatch_ticks\":%.3f"
            ",\"max_dispatch_ticks\":%" PRIu64
            ",\"total_dispatch_ticks\":%.3f"
            ",\"earliest_dispatch_start_tick\":%" PRIu64
            ",\"latest_dispatch_end_tick\":%" PRIu64
            ",\"dispatch_ticks_covered_by_clock_samples\":%s"
            ",\"min_dispatch_ns\":%.3f,\"avg_dispatch_ns\":%.3f"
            ",\"max_dispatch_ns\":%.3f,\"total_dispatch_ns\":%.3f}\n",
            device->physical_device_ordinal, device->device_record_count,
            device->queue_record_count, device->queue_count,
            device->clock_sample_count, has_clock_fit ? "true" : "false",
            ns_per_tick, tick_frequency_hz,
            device->minimum_clock_uncertainty_ns == INT64_MAX
                ? 0
                : device->minimum_clock_uncertainty_ns,
            device->maximum_clock_uncertainty_ns, device->dispatch_event_count,
            valid_dispatch_count, device->invalid_dispatch_event_count,
            valid_dispatch_count ? device->minimum_dispatch_ticks : 0,
            average_ticks,
            valid_dispatch_count ? device->maximum_dispatch_ticks : 0,
            device->total_dispatch_ticks,
            valid_dispatch_count ? device->earliest_dispatch_start_tick : 0,
            valid_dispatch_count ? device->latest_dispatch_end_tick : 0,
            clock_covers_dispatches ? "true" : "false",
            has_clock_fit && valid_dispatch_count
                ? (double)device->minimum_dispatch_ticks * ns_per_tick
                : 0.0,
            has_clock_fit && valid_dispatch_count ? average_ticks * ns_per_tick
                                                  : 0.0,
            has_clock_fit && valid_dispatch_count
                ? (double)device->maximum_dispatch_ticks * ns_per_tick
                : 0.0,
            has_clock_fit ? device->total_dispatch_ticks * ns_per_tick : 0.0);
  }
}

static iree_status_t iree_profile_dispatch_process_queue_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      status = iree_profile_dispatch_append_queue(context, &record_value);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_executable_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_executable_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      status = iree_profile_dispatch_append_executable(context, &record_value);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_export_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_executable_export_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_executable_export_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      if (record_value.name_length != record_length - sizeof(record_value)) {
        status =
            iree_make_status(IREE_STATUS_DATA_LOSS,
                             "executable export name length is inconsistent");
      }
      if (iree_status_is_ok(status)) {
        iree_profile_dispatch_export_t export_info = {
            .executable_id = record_value.executable_id,
            .export_ordinal = record_value.export_ordinal,
            .constant_count = record_value.constant_count,
            .binding_count = record_value.binding_count,
            .parameter_count = record_value.parameter_count,
            .workgroup_size = {record_value.workgroup_size[0],
                               record_value.workgroup_size[1],
                               record_value.workgroup_size[2]},
            .name =
                iree_make_string_view((const char*)record->payload.data +
                                          payload_offset + sizeof(record_value),
                                      record_value.name_length),
        };
        status = iree_profile_dispatch_append_export(context, &export_info);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_command_buffer_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_buffer_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_command_buffer_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      status =
          iree_profile_dispatch_append_command_buffer(context, &record_value);
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_command_operation_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_command_operation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_command_operation_record_t record_value;
      memcpy(&record_value, record->payload.data + payload_offset,
             sizeof(record_value));
      if (!iree_profile_dispatch_find_command_buffer(
              context, record_value.command_buffer_id)) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "command operation references missing command-buffer metadata "
            "command_buffer=%" PRIu64 " command_index=%u",
            record_value.command_buffer_id, record_value.command_index);
      }
      if (iree_status_is_ok(status)) {
        status = iree_profile_dispatch_append_command_operation(context,
                                                                &record_value);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_clock_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_clock_correlation_record_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_clock_correlation_record_t clock_record;
      memcpy(&clock_record, record->payload.data + payload_offset,
             sizeof(clock_record));

      iree_profile_dispatch_device_t* device = NULL;
      status = iree_profile_dispatch_get_device(
          context, clock_record.physical_device_ordinal, &device);
      if (iree_status_is_ok(status)) {
        iree_profile_dispatch_record_clock_sample(device, &clock_record);
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_metadata_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    return iree_profile_dispatch_process_queue_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    return iree_profile_dispatch_process_executable_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS)) {
    return iree_profile_dispatch_process_export_records(context, record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    return iree_profile_dispatch_process_command_buffer_records(context,
                                                                record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    return iree_profile_dispatch_process_command_operation_records(context,
                                                                   record);
  } else if (iree_string_view_equal(
                 record->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    return iree_profile_dispatch_process_clock_records(context, record);
  }
  return iree_ok_status();
}

static void iree_profile_dispatch_print_event_jsonl(
    const iree_hal_profile_file_record_t* file_record,
    const iree_hal_profile_dispatch_event_t* event, iree_string_view_t key,
    double ns_per_tick, bool has_clock_fit, FILE* file) {
  const bool is_valid = event->start_tick != 0 && event->end_tick != 0 &&
                        event->end_tick >= event->start_tick;
  const uint64_t duration_ticks =
      is_valid ? event->end_tick - event->start_tick : 0;

  fprintf(file,
          "{\"type\":\"dispatch_event\",\"physical_device_ordinal\":%u"
          ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
          ",\"event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64,
          file_record->header.physical_device_ordinal,
          file_record->header.queue_ordinal, file_record->header.stream_id,
          event->event_id, event->submission_id);
  fprintf(file,
          ",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u"
          ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u",
          event->command_buffer_id, event->command_index, event->executable_id,
          event->export_ordinal);
  fprintf(file, ",\"key\":");
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"flags\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]",
          event->flags, event->workgroup_count[0], event->workgroup_count[1],
          event->workgroup_count[2], event->workgroup_size[0],
          event->workgroup_size[1], event->workgroup_size[2]);
  fprintf(file,
          ",\"start_tick\":%" PRIu64 ",\"end_tick\":%" PRIu64
          ",\"duration_ticks\":%" PRIu64 ",\"valid\":%s",
          event->start_tick, event->end_tick, duration_ticks,
          is_valid ? "true" : "false");
  fprintf(file, ",\"clock_fit_available\":%s",
          has_clock_fit ? "true" : "false");
  fprintf(
      file, ",\"duration_ns\":%.3f",
      has_clock_fit && is_valid ? (double)duration_ticks * ns_per_tick : 0.0);
  fputs("}\n", file);
}

static const char* iree_profile_command_operation_type_name(
    iree_hal_profile_command_operation_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_PROFILE_MARKER:
      return "profile_marker";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH:
      return "branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH:
      return "cond_branch";
    case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN:
      return "return";
    default:
      return "unknown";
  }
}

static const char* iree_profile_queue_event_type_name(
    iree_hal_profile_queue_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER:
      return "barrier";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE:
      return "execute";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY:
      return "copy";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL:
      return "fill";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE:
      return "update";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ:
      return "read";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE:
      return "write";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA:
      return "alloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA:
      return "dealloca";
    case IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL:
      return "host_call";
    default:
      return "unknown";
  }
}

static const char* iree_profile_queue_dependency_strategy_name(
    iree_hal_profile_queue_dependency_strategy_t strategy) {
  switch (strategy) {
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE:
      return "none";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE:
      return "inline";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER:
      return "device_barrier";
    case IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER:
      return "software_defer";
    default:
      return "unknown";
  }
}

static bool iree_profile_queue_event_matches(
    const iree_hal_profile_queue_event_t* event, int64_t id_filter,
    iree_string_view_t filter) {
  if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_queue_event_type_name(event->type));
  return iree_profile_dispatch_key_matches(type_name, filter);
}

static iree_status_t iree_profile_dispatch_process_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file) {
  iree_profile_dispatch_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_dispatch_get_device(
      context, record->header.physical_device_ordinal, &device));
  double ns_per_tick = 0.0;
  double tick_frequency_hz = 0.0;
  const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
      device, &ns_per_tick, &tick_frequency_hz);
  (void)tick_frequency_hz;

  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_dispatch_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_dispatch_event_t event;
      memcpy(&event, record->payload.data + payload_offset, sizeof(event));
      ++context->total_dispatch_count;

      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      const bool id_matches = iree_profile_dispatch_event_matches_id(
          &event, projection_mode, id_filter);
      if (id_matches) {
        status = iree_profile_dispatch_validate_event_metadata(
            context, record, &event, projection_mode);
      }
      if (iree_status_is_ok(status) && id_matches) {
        status = iree_profile_dispatch_resolve_key(
            context, record->header.physical_device_ordinal,
            event.executable_id, event.export_ordinal, numeric_buffer,
            sizeof(numeric_buffer), &key);
      }
      if (iree_status_is_ok(status) && !iree_string_view_is_empty(key) &&
          iree_profile_dispatch_key_matches(key, filter)) {
        ++context->matched_dispatch_count;
        const bool is_valid = event.start_tick != 0 && event.end_tick != 0 &&
                              event.end_tick >= event.start_tick;
        if (is_valid) {
          ++context->valid_dispatch_count;
        } else {
          ++context->invalid_dispatch_count;
        }
        if (emit_events) {
          iree_profile_dispatch_print_event_jsonl(
              record, &event, key, ns_per_tick, has_clock_fit, file);
        } else {
          iree_profile_dispatch_aggregate_t* aggregate = NULL;
          status = iree_profile_dispatch_get_aggregate(
              context, record->header.physical_device_ordinal,
              event.executable_id, event.export_ordinal, &aggregate);
          if (iree_status_is_ok(status)) {
            iree_profile_dispatch_record_aggregate_event(aggregate, &event);
          }
          if (iree_status_is_ok(status) && event.command_buffer_id != 0) {
            iree_profile_dispatch_command_aggregate_t* command_aggregate = NULL;
            status = iree_profile_dispatch_get_command_aggregate(
                context, event.command_buffer_id, event.submission_id,
                record->header.physical_device_ordinal,
                record->header.queue_ordinal, record->header.stream_id,
                &command_aggregate);
            if (iree_status_is_ok(status)) {
              iree_profile_dispatch_record_command_aggregate_event(
                  command_aggregate, &event);
            }
          }
          if (iree_status_is_ok(status)) {
            iree_profile_dispatch_queue_aggregate_t* queue_aggregate = NULL;
            status = iree_profile_dispatch_get_queue_aggregate(
                context, record->header.physical_device_ordinal,
                record->header.queue_ordinal, record->header.stream_id,
                event.submission_id, &queue_aggregate);
            if (iree_status_is_ok(status)) {
              iree_profile_dispatch_record_queue_aggregate_event(
                  queue_aggregate, &event);
            }
          }
        }
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_queue_event_records(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter) {
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_queue_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_queue_event_t event;
      memcpy(&event, record->payload.data + payload_offset, sizeof(event));
      ++context->total_queue_event_count;
      if (iree_profile_queue_event_matches(&event, id_filter, filter)) {
        ++context->matched_queue_event_count;
        const iree_profile_dispatch_queue_t* queue_info =
            iree_profile_dispatch_find_queue(
                context, event.physical_device_ordinal, event.queue_ordinal,
                event.stream_id);
        if (!queue_info) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "queue event references missing queue metadata "
              "device=%u queue=%u stream=%" PRIu64 " submission=%" PRIu64,
              event.physical_device_ordinal, event.queue_ordinal,
              event.stream_id, event.submission_id);
        }
        if (iree_status_is_ok(status)) {
          status = iree_profile_dispatch_append_queue_event(context, &event);
        }
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_dispatch_process_events_record(
    iree_profile_dispatch_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    iree_profile_projection_mode_t projection_mode, int64_t id_filter,
    bool emit_events, FILE* file) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    if (projection_mode == IREE_PROFILE_PROJECTION_MODE_QUEUE &&
        iree_string_view_equal(record->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
      return iree_profile_dispatch_process_queue_event_records(
          context, record, filter, id_filter);
    }
    return iree_ok_status();
  }
  return iree_profile_dispatch_process_event_records(
      context, record, filter, projection_mode, id_filter, emit_events, file);
}

static const iree_profile_dispatch_device_t* iree_profile_dispatch_find_device(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal) {
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      return &context->devices[i];
    }
  }
  return NULL;
}

static double iree_profile_dispatch_span_ticks(uint64_t earliest_start_tick,
                                               uint64_t latest_end_tick) {
  if (earliest_start_tick == UINT64_MAX || latest_end_tick == 0 ||
      latest_end_tick < earliest_start_tick) {
    return 0.0;
  }
  return (double)(latest_end_tick - earliest_start_tick);
}

static iree_status_t iree_profile_command_operation_resolve_key(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    char* numeric_buffer, iree_host_size_t numeric_buffer_capacity,
    iree_string_view_t* out_key) {
  *out_key = iree_make_cstring_view(
      iree_profile_command_operation_type_name(operation->type));
  if (operation->type != IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH ||
      operation->executable_id == 0 ||
      operation->export_ordinal == UINT32_MAX) {
    return iree_ok_status();
  }

  const iree_profile_dispatch_command_buffer_t* command_buffer =
      iree_profile_dispatch_find_command_buffer(context,
                                                operation->command_buffer_id);
  if (!command_buffer) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "command operation references missing command-buffer metadata "
        "command_buffer=%" PRIu64 " command_index=%u",
        operation->command_buffer_id, operation->command_index);
  }
  return iree_profile_dispatch_resolve_key(
      context, command_buffer->record.physical_device_ordinal,
      operation->executable_id, operation->export_ordinal, numeric_buffer,
      numeric_buffer_capacity, out_key);
}

static bool iree_profile_command_operation_filter_matches(
    iree_string_view_t operation_name, iree_string_view_t key,
    iree_string_view_t filter) {
  return iree_profile_dispatch_key_matches(operation_name, filter) ||
         iree_profile_dispatch_key_matches(key, filter);
}

static iree_status_t iree_profile_command_count_matching_operations(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, iree_host_size_t* out_operation_count) {
  *out_operation_count = 0;
  iree_host_size_t operation_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->command_operation_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    const char* operation_name =
        iree_profile_command_operation_type_name(operation->type);
    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_command_operation_resolve_key(
        context, operation, numeric_buffer, sizeof(numeric_buffer), &key);
    if (iree_status_is_ok(status) &&
        iree_profile_command_operation_filter_matches(
            iree_make_cstring_view(operation_name), key, filter)) {
      ++operation_count;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_operation_count = operation_count;
  }
  return status;
}

static iree_status_t iree_profile_command_print_operation_text(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    iree_string_view_t filter, FILE* file) {
  const char* operation_name =
      iree_profile_command_operation_type_name(operation->type);
  char numeric_buffer[128];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_profile_command_operation_resolve_key(
      context, operation, numeric_buffer, sizeof(numeric_buffer), &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file, "    command[%u]: op=%s block=%u local=%u flags=0x%x",
          operation->command_index, operation_name, operation->block_ordinal,
          operation->block_command_ordinal, operation->flags);
  if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH) {
    fprintf(file,
            " executable=%" PRIu64
            " export=%u key=%.*s bindings=%u"
            " workgroups=[%u,%u,%u] workgroup_size=[%u,%u,%u]",
            operation->executable_id, operation->export_ordinal, (int)key.size,
            key.data, operation->binding_count, operation->workgroup_count[0],
            operation->workgroup_count[1], operation->workgroup_count[2],
            operation->workgroup_size[0], operation->workgroup_size[1],
            operation->workgroup_size[2]);
  } else if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL ||
             operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE) {
    fprintf(file, " target=%u target_offset=%" PRIu64 " length=%" PRIu64,
            operation->target_ordinal, operation->target_offset,
            operation->length);
  } else if (operation->type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY) {
    fprintf(file,
            " source=%u source_offset=%" PRIu64
            " target=%u target_offset=%" PRIu64 " length=%" PRIu64,
            operation->source_ordinal, operation->source_offset,
            operation->target_ordinal, operation->target_offset,
            operation->length);
  } else if (operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BRANCH ||
             operation->type ==
                 IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COND_BRANCH) {
    fprintf(file, " target_block=%u alternate_block=%u",
            operation->target_block_ordinal,
            operation->alternate_block_ordinal);
  }
  fputc('\n', file);
  return iree_ok_status();
}

static iree_status_t iree_profile_command_print_operation_jsonl(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_command_operation_record_t* operation,
    iree_string_view_t filter, FILE* file) {
  const char* operation_name =
      iree_profile_command_operation_type_name(operation->type);
  char numeric_buffer[128];
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_profile_command_operation_resolve_key(
      context, operation, numeric_buffer, sizeof(numeric_buffer), &key));
  if (!iree_profile_command_operation_filter_matches(
          iree_make_cstring_view(operation_name), key, filter)) {
    return iree_ok_status();
  }

  fprintf(file,
          "{\"type\":\"command_operation\",\"command_buffer_id\":%" PRIu64
          ",\"command_index\":%u,\"op\":\"%s\",\"flags\":%u"
          ",\"block_ordinal\":%u,\"block_command_ordinal\":%u",
          operation->command_buffer_id, operation->command_index,
          operation_name, operation->flags, operation->block_ordinal,
          operation->block_command_ordinal);
  fprintf(file, ",\"key\":");
  iree_profile_fprint_json_string(file, key);
  fprintf(file,
          ",\"executable_id\":%" PRIu64
          ",\"export_ordinal\":%u"
          ",\"binding_count\":%u,\"workgroup_count\":[%u,%u,%u]"
          ",\"workgroup_size\":[%u,%u,%u]"
          ",\"source_ordinal\":%u,\"target_ordinal\":%u"
          ",\"source_offset\":%" PRIu64 ",\"target_offset\":%" PRIu64
          ",\"length\":%" PRIu64
          ",\"target_block_ordinal\":%u,\"alternate_block_ordinal\":%u}\n",
          operation->executable_id, operation->export_ordinal,
          operation->binding_count, operation->workgroup_count[0],
          operation->workgroup_count[1], operation->workgroup_count[2],
          operation->workgroup_size[0], operation->workgroup_size[1],
          operation->workgroup_size[2], operation->source_ordinal,
          operation->target_ordinal, operation->source_offset,
          operation->target_offset, operation->length,
          operation->target_block_ordinal, operation->alternate_block_ordinal);
  return iree_ok_status();
}

static iree_status_t iree_profile_dispatch_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    FILE* file) {
  fprintf(file, "IREE HAL profile dispatch summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "dispatches: total=%" PRIu64 " matched=%" PRIu64 " valid=%" PRIu64
          " invalid=%" PRIu64 " groups=%" PRIhsz "\n",
          context->total_dispatch_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count,
          context->aggregate_count);
  fprintf(file, "groups:\n");

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    const iree_profile_dispatch_device_t* device = NULL;
    for (iree_host_size_t j = 0; j < context->device_count; ++j) {
      if (context->devices[j].physical_device_ordinal ==
          aggregate->physical_device_ordinal) {
        device = &context->devices[j];
        break;
      }
    }
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_dispatch_resolve_key(
        context, aggregate->physical_device_ordinal, aggregate->executable_id,
        aggregate->export_ordinal, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance_ticks =
          aggregate->valid_count > 1
              ? aggregate->m2_ticks / (double)(aggregate->valid_count - 1)
              : 0.0;
      const double stddev_ticks = iree_profile_sqrt_f64(variance_ticks);
      fprintf(file, "  %.*s\n", (int)key.size, key.data);
      fprintf(file,
              "    device=%u executable=%" PRIu64 " export=%u count=%" PRIu64
              " valid=%" PRIu64 " invalid=%" PRIu64 "\n",
              aggregate->physical_device_ordinal, aggregate->executable_id,
              aggregate->export_ordinal, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count);
      if (aggregate->valid_count != 0) {
        fprintf(file,
                "    ticks: min=%" PRIu64 " avg=%.3f stddev=%.3f max=%" PRIu64
                " total=%.3f\n",
                aggregate->minimum_ticks, aggregate->mean_ticks, stddev_ticks,
                aggregate->maximum_ticks, aggregate->total_ticks);
        if (has_clock_fit) {
          fprintf(file,
                  "    time_ns: min=%.3f avg=%.3f stddev=%.3f max=%.3f"
                  " total=%.3f\n",
                  (double)aggregate->minimum_ticks * ns_per_tick,
                  aggregate->mean_ticks * ns_per_tick,
                  stddev_ticks * ns_per_tick,
                  (double)aggregate->maximum_ticks * ns_per_tick,
                  aggregate->total_ticks * ns_per_tick);
        } else {
          fprintf(file, "    time_ns: unavailable\n");
        }
      }
      fprintf(
          file,
          "    last_geometry: workgroup_count=%ux%ux%u"
          " workgroup_size=%ux%ux%u\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
    }
  }
  return status;
}

static void iree_profile_dispatch_print_jsonl_summary(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    bool emit_events, FILE* file) {
  fprintf(file, "{\"type\":\"dispatch_summary\",\"mode\":\"%s\",\"filter\":",
          emit_events ? "events" : "aggregate");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"total_dispatches\":%" PRIu64 ",\"matched_dispatches\":%" PRIu64
          ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
          ",\"aggregate_groups\":%" PRIhsz "}\n",
          context->total_dispatch_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count,
          context->aggregate_count);
}

static iree_status_t iree_profile_dispatch_print_jsonl_aggregates(
    const iree_profile_dispatch_context_t* context, FILE* file) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < context->aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    const iree_profile_dispatch_device_t* device = NULL;
    for (iree_host_size_t j = 0; j < context->device_count; ++j) {
      if (context->devices[j].physical_device_ordinal ==
          aggregate->physical_device_ordinal) {
        device = &context->devices[j];
        break;
      }
    }
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    (void)tick_frequency_hz;

    char numeric_buffer[128];
    iree_string_view_t key = iree_string_view_empty();
    status = iree_profile_dispatch_resolve_key(
        context, aggregate->physical_device_ordinal, aggregate->executable_id,
        aggregate->export_ordinal, numeric_buffer, sizeof(numeric_buffer),
        &key);
    if (iree_status_is_ok(status)) {
      const double variance_ticks =
          aggregate->valid_count > 1
              ? aggregate->m2_ticks / (double)(aggregate->valid_count - 1)
              : 0.0;
      const double stddev_ticks = iree_profile_sqrt_f64(variance_ticks);
      fprintf(file,
              "{\"type\":\"dispatch_group\",\"physical_device_ordinal\":%u"
              ",\"executable_id\":%" PRIu64 ",\"export_ordinal\":%u,\"key\":",
              aggregate->physical_device_ordinal, aggregate->executable_id,
              aggregate->export_ordinal);
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"count\":%" PRIu64 ",\"valid\":%" PRIu64
              ",\"invalid\":%" PRIu64 ",\"min_ticks\":%" PRIu64
              ",\"avg_ticks\":%.3f,\"stddev_ticks\":%.3f"
              ",\"max_ticks\":%" PRIu64 ",\"total_ticks\":%.3f",
              aggregate->dispatch_count, aggregate->valid_count,
              aggregate->invalid_count,
              aggregate->valid_count ? aggregate->minimum_ticks : 0,
              aggregate->valid_count ? aggregate->mean_ticks : 0.0,
              stddev_ticks,
              aggregate->valid_count ? aggregate->maximum_ticks : 0,
              aggregate->total_ticks);
      fprintf(file, ",\"clock_fit_available\":%s",
              has_clock_fit ? "true" : "false");
      fprintf(file,
              ",\"min_ns\":%.3f,\"avg_ns\":%.3f,\"stddev_ns\":%.3f"
              ",\"max_ns\":%.3f,\"total_ns\":%.3f",
              has_clock_fit && aggregate->valid_count
                  ? (double)aggregate->minimum_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit && aggregate->valid_count
                  ? aggregate->mean_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit ? stddev_ticks * ns_per_tick : 0.0,
              has_clock_fit && aggregate->valid_count
                  ? (double)aggregate->maximum_ticks * ns_per_tick
                  : 0.0,
              has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
      fprintf(
          file,
          ",\"last_workgroup_count\":[%u,%u,%u]"
          ",\"last_workgroup_size\":[%u,%u,%u]}\n",
          aggregate->last_workgroup_count[0],
          aggregate->last_workgroup_count[1],
          aggregate->last_workgroup_count[2], aggregate->last_workgroup_size[0],
          aggregate->last_workgroup_size[1], aggregate->last_workgroup_size[2]);
    }
  }
  return status;
}

static iree_status_t iree_profile_executable_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile executable summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "executables=%" PRIhsz " exports=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->executable_count, context->export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->executables[i].record;
    if (id_filter >= 0 && executable->executable_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "executable %" PRIu64
            ": exports=%u flags=%u "
            "code_object_hash=%016" PRIx64 "%016" PRIx64 "\n",
            executable->executable_id, executable->export_count,
            executable->flags, executable->code_object_hash[0],
            executable->code_object_hash[1]);
    for (iree_host_size_t j = 0; j < context->export_count; ++j) {
      const iree_profile_dispatch_export_t* export_info = &context->exports[j];
      if (export_info->executable_id != executable->executable_id) continue;

      char numeric_buffer[128];
      iree_string_view_t key = iree_profile_dispatch_format_export_key(
          export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
      if (!iree_profile_dispatch_key_matches(key, filter)) continue;

      fprintf(file,
              "  export %u: %.*s constants=%u bindings=%u parameters=%u "
              "workgroup_size=%ux%ux%u\n",
              export_info->export_ordinal, (int)key.size, key.data,
              export_info->constant_count, export_info->binding_count,
              export_info->parameter_count, export_info->workgroup_size[0],
              export_info->workgroup_size[1], export_info->workgroup_size[2]);
      bool has_aggregate = false;
      for (iree_host_size_t k = 0; k < context->aggregate_count; ++k) {
        const iree_profile_dispatch_aggregate_t* aggregate =
            &context->aggregates[k];
        if (aggregate->executable_id != export_info->executable_id ||
            aggregate->export_ordinal != export_info->export_ordinal) {
          continue;
        }
        has_aggregate = true;
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(
                context, aggregate->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        const double average_ticks =
            aggregate->valid_count
                ? aggregate->total_ticks / (double)aggregate->valid_count
                : 0.0;
        fprintf(file,
                "    device=%u dispatches=%" PRIu64 " valid=%" PRIu64
                " invalid=%" PRIu64 " ticks[min/avg/max/total]=%" PRIu64
                "/%.3f/%" PRIu64 "/%.3f",
                aggregate->physical_device_ordinal, aggregate->dispatch_count,
                aggregate->valid_count, aggregate->invalid_count,
                aggregate->valid_count ? aggregate->minimum_ticks : 0,
                average_ticks,
                aggregate->valid_count ? aggregate->maximum_ticks : 0,
                aggregate->total_ticks);
        if (has_clock_fit) {
          fprintf(file, " ns[min/avg/max/total]=%.3f/%.3f/%.3f/%.3f",
                  aggregate->valid_count
                      ? (double)aggregate->minimum_ticks * ns_per_tick
                      : 0.0,
                  average_ticks * ns_per_tick,
                  aggregate->valid_count
                      ? (double)aggregate->maximum_ticks * ns_per_tick
                      : 0.0,
                  aggregate->total_ticks * ns_per_tick);
        }
        fputc('\n', file);
      }
      if (!has_aggregate) {
        fprintf(file, "    dispatches=0\n");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_executable_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "{\"type\":\"executable_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"executables\":%" PRIhsz ",\"exports\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->executable_count, context->export_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->executable_count; ++i) {
    const iree_hal_profile_executable_record_t* executable =
        &context->executables[i].record;
    if (id_filter >= 0 && executable->executable_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "{\"type\":\"executable\",\"executable_id\":%" PRIu64
            ",\"flags\":%u,\"export_count\":%u,\"code_object_hash\":[%" PRIu64
            ",%" PRIu64 "]}\n",
            executable->executable_id, executable->flags,
            executable->export_count, executable->code_object_hash[0],
            executable->code_object_hash[1]);
    for (iree_host_size_t j = 0; j < context->export_count; ++j) {
      const iree_profile_dispatch_export_t* export_info = &context->exports[j];
      if (export_info->executable_id != executable->executable_id) continue;

      char numeric_buffer[128];
      iree_string_view_t key = iree_profile_dispatch_format_export_key(
          export_info, UINT32_MAX, numeric_buffer, sizeof(numeric_buffer));
      if (!iree_profile_dispatch_key_matches(key, filter)) continue;

      fprintf(file,
              "{\"type\":\"executable_export\",\"executable_id\":%" PRIu64
              ",\"export_ordinal\":%u,\"key\":",
              export_info->executable_id, export_info->export_ordinal);
      iree_profile_fprint_json_string(file, key);
      fprintf(file,
              ",\"constant_count\":%u,\"binding_count\":%u"
              ",\"parameter_count\":%u,\"workgroup_size\":[%u,%u,%u]}\n",
              export_info->constant_count, export_info->binding_count,
              export_info->parameter_count, export_info->workgroup_size[0],
              export_info->workgroup_size[1], export_info->workgroup_size[2]);
      for (iree_host_size_t k = 0; k < context->aggregate_count; ++k) {
        const iree_profile_dispatch_aggregate_t* aggregate =
            &context->aggregates[k];
        if (aggregate->executable_id != export_info->executable_id ||
            aggregate->export_ordinal != export_info->export_ordinal) {
          continue;
        }
        const iree_profile_dispatch_device_t* device =
            iree_profile_dispatch_find_device(
                context, aggregate->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        const double average_ticks =
            aggregate->valid_count
                ? aggregate->total_ticks / (double)aggregate->valid_count
                : 0.0;
        fprintf(file,
                "{\"type\":\"executable_export_dispatch_group\""
                ",\"physical_device_ordinal\":%u"
                ",\"executable_id\":%" PRIu64
                ",\"export_ordinal\":%u"
                ",\"key\":",
                aggregate->physical_device_ordinal, aggregate->executable_id,
                aggregate->export_ordinal);
        iree_profile_fprint_json_string(file, key);
        fprintf(file,
                ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
                ",\"invalid\":%" PRIu64 ",\"min_ticks\":%" PRIu64
                ",\"avg_ticks\":%.3f,\"max_ticks\":%" PRIu64
                ",\"total_ticks\":%.3f,\"clock_fit_available\":%s"
                ",\"min_ns\":%.3f,\"avg_ns\":%.3f,\"max_ns\":%.3f"
                ",\"total_ns\":%.3f}\n",
                aggregate->dispatch_count, aggregate->valid_count,
                aggregate->invalid_count,
                aggregate->valid_count ? aggregate->minimum_ticks : 0,
                average_ticks,
                aggregate->valid_count ? aggregate->maximum_ticks : 0,
                aggregate->total_ticks, has_clock_fit ? "true" : "false",
                has_clock_fit && aggregate->valid_count
                    ? (double)aggregate->minimum_ticks * ns_per_tick
                    : 0.0,
                has_clock_fit ? average_ticks * ns_per_tick : 0.0,
                has_clock_fit && aggregate->valid_count
                    ? (double)aggregate->maximum_ticks * ns_per_tick
                    : 0.0,
                has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_command_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  fprintf(file, "IREE HAL profile command-buffer summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "command_buffers=%" PRIhsz " executions=%" PRIhsz
          " command_operations=%" PRIhsz " matched_command_operations=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->command_buffer_count, context->command_aggregate_count,
          context->command_operation_count, matched_command_operation_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->command_buffers[i].record;
    if (id_filter >= 0 &&
        command_buffer->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "command_buffer %" PRIu64 ": device=%u mode=%" PRIu64
            " categories=%" PRIu64 " queue_affinity=%" PRIu64 "\n",
            command_buffer->command_buffer_id,
            command_buffer->physical_device_ordinal, command_buffer->mode,
            command_buffer->command_categories, command_buffer->queue_affinity);
    for (iree_host_size_t j = 0;
         j < context->command_operation_count && iree_status_is_ok(status);
         ++j) {
      const iree_hal_profile_command_operation_record_t* operation =
          &context->command_operations[j].record;
      if (operation->command_buffer_id != command_buffer->command_buffer_id) {
        continue;
      }
      status = iree_profile_command_print_operation_text(context, operation,
                                                         filter, file);
    }
    for (iree_host_size_t j = 0;
         j < context->command_aggregate_count && iree_status_is_ok(status);
         ++j) {
      const iree_profile_dispatch_command_aggregate_t* aggregate =
          &context->command_aggregates[j];
      if (aggregate->command_buffer_id != command_buffer->command_buffer_id) {
        continue;
      }
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double span_ticks = iree_profile_dispatch_span_ticks(
          aggregate->earliest_start_tick, aggregate->latest_end_tick);
      fprintf(file,
              "  submission=%" PRIu64 " queue=%u stream=%" PRIu64
              " dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
              " span_ticks=%.3f total_dispatch_ticks=%.3f",
              aggregate->submission_id, aggregate->queue_ordinal,
              aggregate->stream_id, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count, span_ticks,
              aggregate->total_ticks);
      if (has_clock_fit) {
        fprintf(file, " span_ns=%.3f total_dispatch_ns=%.3f",
                span_ticks * ns_per_tick, aggregate->total_ticks * ns_per_tick);
      }
      fputc('\n', file);
    }
  }
  return status;
}

static iree_status_t iree_profile_command_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  iree_host_size_t matched_command_operation_count = 0;
  IREE_RETURN_IF_ERROR(iree_profile_command_count_matching_operations(
      context, filter, id_filter, &matched_command_operation_count));

  fprintf(file, "{\"type\":\"command_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"command_buffers\":%" PRIhsz ",\"executions\":%" PRIhsz
          ",\"command_operations\":%" PRIhsz
          ",\"matched_command_operations\":%" PRIhsz
          ",\"matched_dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
          ",\"invalid_dispatches\":%" PRIu64 "}\n",
          context->command_buffer_count, context->command_aggregate_count,
          context->command_operation_count, matched_command_operation_count,
          context->matched_dispatch_count, context->valid_dispatch_count,
          context->invalid_dispatch_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < context->command_buffer_count; ++i) {
    const iree_hal_profile_command_buffer_record_t* command_buffer =
        &context->command_buffers[i].record;
    if (id_filter >= 0 &&
        command_buffer->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(file,
            "{\"type\":\"command_buffer\",\"command_buffer_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"mode\":%" PRIu64
            ",\"command_categories\":%" PRIu64 ",\"queue_affinity\":%" PRIu64
            "}\n",
            command_buffer->command_buffer_id,
            command_buffer->physical_device_ordinal, command_buffer->mode,
            command_buffer->command_categories, command_buffer->queue_affinity);
  }
  for (iree_host_size_t i = 0;
       i < context->command_operation_count && iree_status_is_ok(status); ++i) {
    const iree_hal_profile_command_operation_record_t* operation =
        &context->command_operations[i].record;
    if (id_filter >= 0 && operation->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    status = iree_profile_command_print_operation_jsonl(context, operation,
                                                        filter, file);
  }
  for (iree_host_size_t i = 0;
       i < context->command_aggregate_count && iree_status_is_ok(status); ++i) {
    const iree_profile_dispatch_command_aggregate_t* aggregate =
        &context->command_aggregates[i];
    if (id_filter >= 0 && aggregate->command_buffer_id != (uint64_t)id_filter) {
      continue;
    }
    const iree_profile_dispatch_device_t* device =
        iree_profile_dispatch_find_device(context,
                                          aggregate->physical_device_ordinal);
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const double span_ticks = iree_profile_dispatch_span_ticks(
        aggregate->earliest_start_tick, aggregate->latest_end_tick);
    fprintf(file,
            "{\"type\":\"command_execution\",\"command_buffer_id\":%" PRIu64
            ",\"submission_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u"
            ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
            ",\"invalid\":%" PRIu64
            ",\"span_ticks\":%.3f"
            ",\"total_dispatch_ticks\":%.3f,\"clock_fit_available\":%s"
            ",\"span_ns\":%.3f,\"total_dispatch_ns\":%.3f}\n",
            aggregate->command_buffer_id, aggregate->submission_id,
            aggregate->physical_device_ordinal, aggregate->queue_ordinal,
            aggregate->stream_id, aggregate->dispatch_count,
            aggregate->valid_count, aggregate->invalid_count, span_ticks,
            aggregate->total_ticks, has_clock_fit ? "true" : "false",
            has_clock_fit ? span_ticks * ns_per_tick : 0.0,
            has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
  }
  return status;
}

static iree_status_t iree_profile_queue_print_text(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "IREE HAL profile queue summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  fprintf(file,
          "queues=%" PRIhsz " queue_events=%" PRIhsz " submissions=%" PRIhsz
          " matched_dispatches=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
          "\n",
          context->queue_count, context->queue_event_count,
          context->queue_aggregate_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue = &context->queues[i].record;
    fprintf(file, "queue device=%u ordinal=%u stream=%" PRIu64 "\n",
            queue->physical_device_ordinal, queue->queue_ordinal,
            queue->stream_id);
    for (iree_host_size_t j = 0; j < context->queue_aggregate_count; ++j) {
      const iree_profile_dispatch_queue_aggregate_t* aggregate =
          &context->queue_aggregates[j];
      if (aggregate->physical_device_ordinal !=
              queue->physical_device_ordinal ||
          aggregate->queue_ordinal != queue->queue_ordinal ||
          aggregate->stream_id != queue->stream_id) {
        continue;
      }
      if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
        continue;
      }
      const iree_profile_dispatch_device_t* device =
          iree_profile_dispatch_find_device(context,
                                            aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double span_ticks = iree_profile_dispatch_span_ticks(
          aggregate->earliest_start_tick, aggregate->latest_end_tick);
      fprintf(file,
              "  submission=%" PRIu64 " dispatches=%" PRIu64 " valid=%" PRIu64
              " invalid=%" PRIu64 " span_ticks=%.3f total_dispatch_ticks=%.3f",
              aggregate->submission_id, aggregate->dispatch_count,
              aggregate->valid_count, aggregate->invalid_count, span_ticks,
              aggregate->total_ticks);
      if (has_clock_fit) {
        fprintf(file, " span_ns=%.3f total_dispatch_ns=%.3f",
                span_ticks * ns_per_tick, aggregate->total_ticks * ns_per_tick);
      }
      fputc('\n', file);
    }
    for (iree_host_size_t j = 0; j < context->queue_event_count; ++j) {
      const iree_hal_profile_queue_event_t* event =
          &context->queue_events[j].record;
      if (event->physical_device_ordinal != queue->physical_device_ordinal ||
          event->queue_ordinal != queue->queue_ordinal ||
          event->stream_id != queue->stream_id) {
        continue;
      }
      if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
        continue;
      }
      fprintf(
          file,
          "  event=%" PRIu64 " type=%s submission=%" PRIu64
          " strategy=%s waits=%u signals=%u barriers=%u ops=%u bytes=%" PRIu64
          "\n",
          event->event_id, iree_profile_queue_event_type_name(event->type),
          event->submission_id,
          iree_profile_queue_dependency_strategy_name(
              event->dependency_strategy),
          event->wait_count, event->signal_count, event->barrier_count,
          event->operation_count, event->payload_length);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_queue_print_jsonl(
    const iree_profile_dispatch_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  fprintf(file, "{\"type\":\"queue_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"queues\":%" PRIhsz ",\"queue_events\":%" PRIhsz
          ",\"submissions\":%" PRIhsz ",\"matched_dispatches\":%" PRIu64
          ",\"valid_dispatches\":%" PRIu64 ",\"invalid_dispatches\":%" PRIu64
          "}\n",
          context->queue_count, context->queue_event_count,
          context->queue_aggregate_count, context->matched_dispatch_count,
          context->valid_dispatch_count, context->invalid_dispatch_count);

  for (iree_host_size_t i = 0; i < context->queue_count; ++i) {
    const iree_hal_profile_queue_record_t* queue = &context->queues[i].record;
    fprintf(file,
            "{\"type\":\"queue\",\"physical_device_ordinal\":%u"
            ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64 "}\n",
            queue->physical_device_ordinal, queue->queue_ordinal,
            queue->stream_id);
  }
  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    const iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (id_filter >= 0 && aggregate->submission_id != (uint64_t)id_filter) {
      continue;
    }
    const iree_profile_dispatch_device_t* device =
        iree_profile_dispatch_find_device(context,
                                          aggregate->physical_device_ordinal);
    double ns_per_tick = 0.0;
    double tick_frequency_hz = 0.0;
    const bool has_clock_fit = iree_profile_dispatch_device_try_fit_clock(
        device, &ns_per_tick, &tick_frequency_hz);
    const double span_ticks = iree_profile_dispatch_span_ticks(
        aggregate->earliest_start_tick, aggregate->latest_end_tick);
    fprintf(file,
            "{\"type\":\"queue_submission\",\"submission_id\":%" PRIu64
            ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
            ",\"stream_id\":%" PRIu64 ",\"dispatches\":%" PRIu64
            ",\"valid\":%" PRIu64 ",\"invalid\":%" PRIu64
            ",\"span_ticks\":%.3f,\"total_dispatch_ticks\":%.3f"
            ",\"clock_fit_available\":%s,\"span_ns\":%.3f"
            ",\"total_dispatch_ns\":%.3f}\n",
            aggregate->submission_id, aggregate->physical_device_ordinal,
            aggregate->queue_ordinal, aggregate->stream_id,
            aggregate->dispatch_count, aggregate->valid_count,
            aggregate->invalid_count, span_ticks, aggregate->total_ticks,
            has_clock_fit ? "true" : "false",
            has_clock_fit ? span_ticks * ns_per_tick : 0.0,
            has_clock_fit ? aggregate->total_ticks * ns_per_tick : 0.0);
  }
  for (iree_host_size_t i = 0; i < context->queue_event_count; ++i) {
    const iree_hal_profile_queue_event_t* event =
        &context->queue_events[i].record;
    if (id_filter >= 0 && event->submission_id != (uint64_t)id_filter) {
      continue;
    }
    fprintf(
        file,
        "{\"type\":\"queue_event\",\"event_id\":%" PRIu64
        ",\"op\":\"%s\",\"flags\":%u,\"dependency_strategy\":\"%s\""
        ",\"submission_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
        ",\"allocation_id\":%" PRIu64
        ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
        ",\"stream_id\":%" PRIu64 ",\"host_time_ns\":%" PRId64
        ",\"wait_count\":%u,\"signal_count\":%u,\"barrier_count\":%u"
        ",\"operation_count\":%u,\"payload_length\":%" PRIu64 "}\n",
        event->event_id, iree_profile_queue_event_type_name(event->type),
        event->flags,
        iree_profile_queue_dependency_strategy_name(event->dependency_strategy),
        event->submission_id, event->command_buffer_id, event->allocation_id,
        event->physical_device_ordinal, event->queue_ordinal, event->stream_id,
        event->host_time_ns, event->wait_count, event->signal_count,
        event->barrier_count, event->operation_count, event->payload_length);
  }
  return iree_ok_status();
}

static const char* iree_profile_memory_event_type_name(
    iree_hal_profile_memory_event_type_t type) {
  switch (type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      return "slab_acquire";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      return "slab_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return "pool_reserve";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      return "pool_materialize";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      return "pool_release";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      return "pool_wait";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return "queue_alloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      return "queue_dealloca";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      return "buffer_allocate";
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return "buffer_free";
    default:
      return "unknown";
  }
}

static const char* iree_profile_memory_lifecycle_kind_name(
    iree_profile_memory_lifecycle_kind_t kind) {
  switch (kind) {
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB:
      return "slab";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION:
      return "pool_reservation";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION:
      return "queue_allocation";
    case IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION:
      return "buffer_allocation";
    default:
      return "unknown";
  }
}

static bool iree_profile_memory_event_allocation_kind(
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t* out_kind) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      if (iree_all_bits_set(
              event->flags,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
        *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION;
        return true;
      }
      return false;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION;
      return true;
    default:
      return false;
  }
}

static bool iree_profile_memory_event_pool_kind(
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t* out_kind) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_SLAB;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION;
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      *out_kind = IREE_PROFILE_MEMORY_LIFECYCLE_KIND_BUFFER_ALLOCATION;
      return true;
    default:
      return false;
  }
}

static bool iree_profile_memory_event_increases_live_bytes(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      return true;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      return iree_all_bits_set(
          event->flags, IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION);
    default:
      return false;
  }
}

static bool iree_profile_memory_event_decreases_live_bytes(
    const iree_hal_profile_memory_event_t* event) {
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      return true;
    default:
      return false;
  }
}

static void iree_profile_memory_context_initialize(
    iree_allocator_t host_allocator,
    iree_profile_memory_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;
}

static void iree_profile_memory_context_deinitialize(
    iree_profile_memory_context_t* context) {
  iree_allocator_free(context->host_allocator, context->devices);
  iree_allocator_free(context->host_allocator, context->pools);
  iree_allocator_free(context->host_allocator, context->allocations);
  memset(context, 0, sizeof(*context));
}

static iree_status_t iree_profile_memory_get_device(
    iree_profile_memory_context_t* context, uint32_t physical_device_ordinal,
    iree_profile_memory_device_t** out_device) {
  *out_device = NULL;

  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    if (context->devices[i].physical_device_ordinal ==
        physical_device_ordinal) {
      *out_device = &context->devices[i];
      return iree_ok_status();
    }
  }

  if (context->device_count + 1 > context->device_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)4, context->device_count + 1),
        sizeof(context->devices[0]), &context->device_capacity,
        (void**)&context->devices));
  }

  iree_profile_memory_device_t* device =
      &context->devices[context->device_count++];
  memset(device, 0, sizeof(*device));
  device->physical_device_ordinal = physical_device_ordinal;
  *out_device = device;
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_get_pool(
    iree_profile_memory_context_t* context,
    iree_profile_memory_lifecycle_kind_t kind, uint32_t physical_device_ordinal,
    uint64_t pool_id, uint64_t memory_type,
    iree_profile_memory_pool_t** out_pool) {
  *out_pool = NULL;

  for (iree_host_size_t i = context->pool_count; i > 0; --i) {
    iree_profile_memory_pool_t* pool = &context->pools[i - 1];
    if (pool->kind == kind &&
        pool->physical_device_ordinal == physical_device_ordinal &&
        pool->pool_id == pool_id && pool->memory_type == memory_type) {
      *out_pool = pool;
      return iree_ok_status();
    }
  }

  if (context->pool_count + 1 > context->pool_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)8, context->pool_count + 1),
        sizeof(context->pools[0]), &context->pool_capacity,
        (void**)&context->pools));
  }

  iree_profile_memory_pool_t* pool = &context->pools[context->pool_count++];
  memset(pool, 0, sizeof(*pool));
  pool->kind = kind;
  pool->physical_device_ordinal = physical_device_ordinal;
  pool->pool_id = pool_id;
  pool->memory_type = memory_type;
  *out_pool = pool;
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_get_allocation(
    iree_profile_memory_context_t* context,
    iree_profile_memory_lifecycle_kind_t kind, uint32_t physical_device_ordinal,
    uint64_t allocation_id, uint64_t pool_id,
    iree_profile_memory_allocation_t** out_allocation) {
  *out_allocation = NULL;

  for (iree_host_size_t i = context->allocation_count; i > 0; --i) {
    iree_profile_memory_allocation_t* allocation = &context->allocations[i - 1];
    if (allocation->kind == kind &&
        allocation->physical_device_ordinal == physical_device_ordinal &&
        allocation->allocation_id == allocation_id &&
        allocation->pool_id == pool_id) {
      *out_allocation = allocation;
      return iree_ok_status();
    }
  }

  if (context->allocation_count + 1 > context->allocation_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        context->host_allocator,
        iree_max((iree_host_size_t)16, context->allocation_count + 1),
        sizeof(context->allocations[0]), &context->allocation_capacity,
        (void**)&context->allocations));
  }

  iree_profile_memory_allocation_t* allocation =
      &context->allocations[context->allocation_count++];
  memset(allocation, 0, sizeof(*allocation));
  allocation->kind = kind;
  allocation->physical_device_ordinal = physical_device_ordinal;
  allocation->allocation_id = allocation_id;
  allocation->pool_id = pool_id;
  *out_allocation = allocation;
  return iree_ok_status();
}

static uint64_t iree_profile_memory_resolve_pool_id(
    const iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event,
    iree_profile_memory_lifecycle_kind_t kind) {
  if (event->pool_id != 0) return event->pool_id;
  if (kind != IREE_PROFILE_MEMORY_LIFECYCLE_KIND_QUEUE_ALLOCATION) {
    return event->pool_id;
  }

  for (iree_host_size_t i = context->allocation_count; i > 0; --i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i - 1];
    if (allocation->kind == kind &&
        allocation->physical_device_ordinal == event->physical_device_ordinal &&
        allocation->allocation_id == event->allocation_id) {
      return allocation->pool_id;
    }
  }
  return event->pool_id;
}

static bool iree_profile_memory_event_matches(
    const iree_hal_profile_memory_event_t* event, int64_t id_filter,
    iree_string_view_t filter) {
  if (id_filter >= 0 && event->event_id != (uint64_t)id_filter &&
      event->allocation_id != (uint64_t)id_filter) {
    return false;
  }
  iree_string_view_t type_name =
      iree_make_cstring_view(iree_profile_memory_event_type_name(event->type));
  return iree_profile_dispatch_key_matches(type_name, filter);
}

static void iree_profile_memory_record_event(
    iree_profile_memory_device_t* device,
    const iree_hal_profile_memory_event_t* event) {
  ++device->event_count;
  switch (event->type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_ACQUIRE:
      ++device->slab_acquire_count;
      ++device->current_slab_allocation_count;
      device->high_water_slab_allocation_count =
          iree_max(device->high_water_slab_allocation_count,
                   device->current_slab_allocation_count);
      device->total_slab_acquired_bytes += event->length;
      device->current_slab_bytes += event->length;
      device->high_water_slab_bytes =
          iree_max(device->high_water_slab_bytes, device->current_slab_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_SLAB_RELEASE:
      ++device->slab_release_count;
      device->current_slab_allocation_count =
          device->current_slab_allocation_count == 0
              ? 0
              : device->current_slab_allocation_count - 1;
      device->total_slab_released_bytes += event->length;
      device->current_slab_bytes =
          event->length > device->current_slab_bytes
              ? 0
              : device->current_slab_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
      ++device->pool_reserve_count;
      if (iree_all_bits_set(
              event->flags,
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION)) {
        ++device->current_pool_reservation_count;
        device->high_water_pool_reservation_count =
            iree_max(device->high_water_pool_reservation_count,
                     device->current_pool_reservation_count);
        device->total_pool_reserved_bytes += event->length;
        device->current_pool_reserved_bytes += event->length;
        device->high_water_pool_reserved_bytes =
            iree_max(device->high_water_pool_reserved_bytes,
                     device->current_pool_reserved_bytes);
      }
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
      ++device->pool_materialize_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      ++device->pool_release_count;
      device->current_pool_reservation_count =
          device->current_pool_reservation_count == 0
              ? 0
              : device->current_pool_reservation_count - 1;
      device->total_pool_released_bytes += event->length;
      device->current_pool_reserved_bytes =
          event->length > device->current_pool_reserved_bytes
              ? 0
              : device->current_pool_reserved_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT:
      ++device->pool_wait_count;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE:
      ++device->buffer_allocate_count;
      ++device->current_buffer_allocation_count;
      device->high_water_buffer_allocation_count =
          iree_max(device->high_water_buffer_allocation_count,
                   device->current_buffer_allocation_count);
      device->total_buffer_allocate_bytes += event->length;
      device->current_buffer_bytes += event->length;
      device->high_water_buffer_bytes = iree_max(
          device->high_water_buffer_bytes, device->current_buffer_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE:
      ++device->buffer_free_count;
      device->current_buffer_allocation_count =
          device->current_buffer_allocation_count == 0
              ? 0
              : device->current_buffer_allocation_count - 1;
      device->total_buffer_free_bytes += event->length;
      device->current_buffer_bytes =
          event->length > device->current_buffer_bytes
              ? 0
              : device->current_buffer_bytes - event->length;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
      ++device->queue_alloca_count;
      ++device->current_queue_allocation_count;
      device->high_water_queue_allocation_count =
          iree_max(device->high_water_queue_allocation_count,
                   device->current_queue_allocation_count);
      device->total_queue_alloca_bytes += event->length;
      device->current_queue_bytes += event->length;
      device->high_water_queue_bytes =
          iree_max(device->high_water_queue_bytes, device->current_queue_bytes);
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      ++device->queue_dealloca_count;
      device->current_queue_allocation_count =
          device->current_queue_allocation_count == 0
              ? 0
              : device->current_queue_allocation_count - 1;
      device->total_queue_dealloca_bytes += event->length;
      device->current_queue_bytes =
          event->length > device->current_queue_bytes
              ? 0
              : device->current_queue_bytes - event->length;
      break;
    default:
      break;
  }
}

static void iree_profile_memory_apply_live_increase(
    uint64_t length, uint64_t* current_count, uint64_t* high_water_count,
    uint64_t* current_bytes, uint64_t* high_water_bytes,
    uint64_t* total_allocate_bytes) {
  *current_count += 1;
  *high_water_count = iree_max(*high_water_count, *current_count);
  *current_bytes += length;
  *high_water_bytes = iree_max(*high_water_bytes, *current_bytes);
  *total_allocate_bytes += length;
}

static void iree_profile_memory_apply_live_decrease(
    uint64_t length, uint64_t* current_count, uint64_t* current_bytes,
    uint64_t* total_free_bytes) {
  *current_count = *current_count == 0 ? 0 : *current_count - 1;
  *current_bytes = length > *current_bytes ? 0 : *current_bytes - length;
  *total_free_bytes += length;
}

static iree_status_t iree_profile_memory_record_pool_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
  iree_profile_memory_lifecycle_kind_t kind = 0;
  if (!iree_profile_memory_event_pool_kind(event, &kind)) {
    return iree_ok_status();
  }

  const uint64_t pool_id =
      iree_profile_memory_resolve_pool_id(context, event, kind);
  iree_profile_memory_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_pool(
      context, kind, event->physical_device_ordinal, pool_id,
      event->memory_type, &pool));
  ++pool->event_count;
  pool->buffer_usage |= event->buffer_usage;
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT) {
    ++pool->wait_count;
  } else if (event->type ==
             IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    ++pool->materialize_count;
  }

  if (iree_profile_memory_event_increases_live_bytes(event)) {
    iree_profile_memory_apply_live_increase(
        event->length, &pool->current_allocation_count,
        &pool->high_water_allocation_count, &pool->current_bytes,
        &pool->high_water_bytes, &pool->total_allocate_bytes);
  } else if (iree_profile_memory_event_decreases_live_bytes(event)) {
    iree_profile_memory_apply_live_decrease(
        event->length, &pool->current_allocation_count, &pool->current_bytes,
        &pool->total_free_bytes);
  }
  return iree_ok_status();
}

static iree_status_t iree_profile_memory_record_allocation_event(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_memory_event_t* event) {
  iree_profile_memory_lifecycle_kind_t kind = 0;
  if (!iree_profile_memory_event_allocation_kind(event, &kind)) {
    return iree_ok_status();
  }

  const uint64_t pool_id =
      iree_profile_memory_resolve_pool_id(context, event, kind);
  iree_profile_memory_allocation_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_profile_memory_get_allocation(
      context, kind, event->physical_device_ordinal, event->allocation_id,
      pool_id, &allocation));
  if (allocation->first_event_id == 0) {
    allocation->first_event_id = event->event_id;
    allocation->first_host_time_ns = event->host_time_ns;
  }
  allocation->last_event_id = event->event_id;
  allocation->last_host_time_ns = event->host_time_ns;
  if (allocation->first_submission_id == 0 && event->submission_id != 0) {
    allocation->first_submission_id = event->submission_id;
  }
  if (event->submission_id != 0) {
    allocation->last_submission_id = event->submission_id;
  }
  allocation->backing_id =
      allocation->backing_id ? allocation->backing_id : event->backing_id;
  allocation->memory_type |= event->memory_type;
  allocation->buffer_usage |= event->buffer_usage;
  ++allocation->event_count;
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT) {
    ++allocation->wait_count;
  } else if (event->type ==
             IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE) {
    ++allocation->materialize_count;
  }

  if (iree_profile_memory_event_increases_live_bytes(event)) {
    allocation->total_allocate_bytes += event->length;
    allocation->current_bytes += event->length;
    allocation->high_water_bytes =
        iree_max(allocation->high_water_bytes, allocation->current_bytes);
  } else if (iree_profile_memory_event_decreases_live_bytes(event)) {
    allocation->total_free_bytes += event->length;
    allocation->current_bytes = event->length > allocation->current_bytes
                                    ? 0
                                    : allocation->current_bytes - event->length;
  }
  return iree_ok_status();
}

static void iree_profile_memory_print_event_jsonl(
    const iree_hal_profile_memory_event_t* event, FILE* file) {
  fprintf(file,
          "{\"type\":\"memory_event\",\"event_id\":%" PRIu64 ",\"event_type\":",
          event->event_id);
  iree_profile_fprint_json_string(
      file,
      iree_make_cstring_view(iree_profile_memory_event_type_name(event->type)));
  fprintf(file,
          ",\"event_type_value\":%u,\"flags\":%u,\"result\":%u"
          ",\"host_time_ns\":%" PRId64 ",\"allocation_id\":%" PRIu64
          ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
          ",\"submission_id\":%" PRIu64
          ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
          ",\"frontier_entry_count\":%u,\"memory_type\":%" PRIu64
          ",\"buffer_usage\":%" PRIu64 ",\"offset\":%" PRIu64
          ",\"length\":%" PRIu64 ",\"alignment\":%" PRIu64 "}\n",
          event->type, event->flags, event->result, event->host_time_ns,
          event->allocation_id, event->pool_id, event->backing_id,
          event->submission_id, event->physical_device_ordinal,
          event->queue_ordinal, event->frontier_entry_count, event->memory_type,
          event->buffer_usage, event->offset, event->length, event->alignment);
}

static iree_status_t iree_profile_memory_process_event_records(
    iree_profile_memory_context_t* context,
    const iree_hal_profile_file_record_t* record, iree_string_view_t filter,
    int64_t id_filter, bool emit_events, FILE* file) {
  if (record->header.record_type != IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK) {
    return iree_ok_status();
  }
  if (!iree_string_view_equal(record->content_type,
                              IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    return iree_ok_status();
  }

  const bool is_truncated = iree_any_bit_set(
      record->header.chunk_flags, IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  iree_host_size_t payload_offset = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         payload_offset < record->payload.data_length) {
    iree_host_size_t record_length = 0;
    status = iree_profile_payload_record_length(
        record->content_type, record->payload, payload_offset,
        sizeof(iree_hal_profile_memory_event_t), &record_length);
    if (iree_status_is_ok(status)) {
      iree_hal_profile_memory_event_t event;
      memcpy(&event, record->payload.data + payload_offset, sizeof(event));
      ++context->total_event_count;
      if (iree_profile_memory_event_matches(&event, id_filter, filter)) {
        ++context->matched_event_count;
        if (is_truncated) ++context->truncated_event_count;
        iree_profile_memory_device_t* device = NULL;
        status = iree_profile_memory_get_device(
            context, event.physical_device_ordinal, &device);
        if (iree_status_is_ok(status)) {
          iree_profile_memory_record_event(device, &event);
          status = iree_profile_memory_record_pool_event(context, &event);
        }
        if (iree_status_is_ok(status)) {
          status = iree_profile_memory_record_allocation_event(context, &event);
        }
        if (iree_status_is_ok(status)) {
          if (emit_events) {
            iree_profile_memory_print_event_jsonl(&event, file);
          }
        }
      }
      payload_offset += record_length;
    }
  }
  return status;
}

static iree_status_t iree_profile_memory_print_text(
    const iree_profile_memory_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  uint64_t live_allocation_count = 0;
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    if (context->allocations[i].current_bytes != 0) {
      ++live_allocation_count;
    }
  }

  fprintf(file, "IREE HAL profile memory summary\n");
  fprintf(file, "filter: %.*s\n", (int)filter.size, filter.data);
  if (id_filter >= 0) {
    fprintf(file, "id_filter: %" PRId64 "\n", id_filter);
  }
  fprintf(file,
          "events: total=%" PRIu64 " matched=%" PRIu64
          " truncated_matched=%" PRIu64 " devices=%" PRIhsz " pools=%" PRIhsz
          " allocation_lifecycles=%" PRIhsz " live_lifecycles=%" PRIu64 "\n",
          context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          live_allocation_count);
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(file,
            "device[%u]: events=%" PRIu64 " slab_acquire/release=%" PRIu64
            "/%" PRIu64 " pool_reserve/materialize/release/wait=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " queue_alloca/dealloca=%" PRIu64 "/%" PRIu64
            " buffer_allocate/free=%" PRIu64 "/%" PRIu64 "\n",
            device->physical_device_ordinal, device->event_count,
            device->slab_acquire_count, device->slab_release_count,
            device->pool_reserve_count, device->pool_materialize_count,
            device->pool_release_count, device->pool_wait_count,
            device->queue_alloca_count, device->queue_dealloca_count,
            device->buffer_allocate_count, device->buffer_free_count);
    fprintf(file,
            "  slab_provider_events: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " acquired_bytes=%" PRIu64 " released_bytes=%" PRIu64 "\n",
            device->current_slab_allocation_count,
            device->high_water_slab_allocation_count,
            device->current_slab_bytes, device->high_water_slab_bytes,
            device->total_slab_acquired_bytes,
            device->total_slab_released_bytes);
    fprintf(file,
            "  pool_reservations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " reserved_bytes=%" PRIu64 " released_bytes=%" PRIu64 "\n",
            device->current_pool_reservation_count,
            device->high_water_pool_reservation_count,
            device->current_pool_reserved_bytes,
            device->high_water_pool_reserved_bytes,
            device->total_pool_reserved_bytes,
            device->total_pool_released_bytes);
    fprintf(file,
            "  queue_allocations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " alloca_bytes=%" PRIu64 " dealloca_bytes=%" PRIu64 "\n",
            device->current_queue_allocation_count,
            device->high_water_queue_allocation_count,
            device->current_queue_bytes, device->high_water_queue_bytes,
            device->total_queue_alloca_bytes,
            device->total_queue_dealloca_bytes);
    fprintf(file,
            "  buffer_allocations: live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " allocate_bytes=%" PRIu64 " free_bytes=%" PRIu64 "\n",
            device->current_buffer_allocation_count,
            device->high_water_buffer_allocation_count,
            device->current_buffer_bytes, device->high_water_buffer_bytes,
            device->total_buffer_allocate_bytes,
            device->total_buffer_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->pool_count; ++i) {
    const iree_profile_memory_pool_t* pool = &context->pools[i];
    fprintf(file,
            "pool[%s device=%u id=%" PRIu64 " memory_type=%" PRIu64
            "]: events=%" PRIu64 " waits=%" PRIu64 " materializes=%" PRIu64
            " live_at_end=%" PRIu64 " peak_live=%" PRIu64
            " current_bytes=%" PRIu64 " high_water_bytes=%" PRIu64
            " allocate_bytes=%" PRIu64 " free_bytes=%" PRIu64 "\n",
            iree_profile_memory_lifecycle_kind_name(pool->kind),
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->event_count, pool->wait_count, pool->materialize_count,
            pool->current_allocation_count, pool->high_water_allocation_count,
            pool->current_bytes, pool->high_water_bytes,
            pool->total_allocate_bytes, pool->total_free_bytes);
  }
  if (id_filter >= 0) {
    for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
      const iree_profile_memory_allocation_t* allocation =
          &context->allocations[i];
      const int64_t duration_ns =
          allocation->last_host_time_ns >= allocation->first_host_time_ns
              ? allocation->last_host_time_ns - allocation->first_host_time_ns
              : 0;
      fprintf(
          file,
          "allocation[%s device=%u id=%" PRIu64 " pool=%" PRIu64
          " backing=%" PRIu64 "]: events=%" PRIu64 " waits=%" PRIu64
          " materializes=%" PRIu64 " live_at_end=%s current_bytes=%" PRIu64
          " high_water_bytes=%" PRIu64 " allocate_bytes=%" PRIu64
          " free_bytes=%" PRIu64 " first_event=%" PRIu64 " last_event=%" PRIu64
          " duration_ns=%" PRId64 "\n",
          iree_profile_memory_lifecycle_kind_name(allocation->kind),
          allocation->physical_device_ordinal, allocation->allocation_id,
          allocation->pool_id, allocation->backing_id, allocation->event_count,
          allocation->wait_count, allocation->materialize_count,
          allocation->current_bytes != 0 ? "true" : "false",
          allocation->current_bytes, allocation->high_water_bytes,
          allocation->total_allocate_bytes, allocation->total_free_bytes,
          allocation->first_event_id, allocation->last_event_id, duration_ns);
    }
  }
  return iree_ok_status();
}

static void iree_profile_memory_print_jsonl_summary(
    const iree_profile_memory_context_t* context, iree_string_view_t filter,
    int64_t id_filter, FILE* file) {
  uint64_t live_allocation_count = 0;
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    if (context->allocations[i].current_bytes != 0) {
      ++live_allocation_count;
    }
  }

  fprintf(file, "{\"type\":\"memory_summary\",\"filter\":");
  iree_profile_fprint_json_string(file, filter);
  fprintf(file,
          ",\"id_filter\":%" PRId64 ",\"total_events\":%" PRIu64
          ",\"matched_events\":%" PRIu64
          ",\"truncated_matched_events\":%" PRIu64 ",\"devices\":%" PRIhsz
          ",\"pools\":%" PRIhsz ",\"allocation_lifecycles\":%" PRIhsz
          ",\"live_allocation_lifecycles\":%" PRIu64 "}\n",
          id_filter, context->total_event_count, context->matched_event_count,
          context->truncated_event_count, context->device_count,
          context->pool_count, context->allocation_count,
          live_allocation_count);
  for (iree_host_size_t i = 0; i < context->device_count; ++i) {
    const iree_profile_memory_device_t* device = &context->devices[i];
    fprintf(
        file,
        "{\"type\":\"memory_device\",\"physical_device_ordinal\":%u"
        ",\"events\":%" PRIu64 ",\"slab_acquires\":%" PRIu64
        ",\"slab_releases\":%" PRIu64 ",\"pool_reserves\":%" PRIu64
        ",\"pool_materializes\":%" PRIu64 ",\"pool_releases\":%" PRIu64
        ",\"pool_waits\":%" PRIu64 ",\"queue_allocas\":%" PRIu64
        ",\"queue_deallocas\":%" PRIu64 ",\"buffer_allocates\":%" PRIu64
        ",\"buffer_frees\":%" PRIu64 ",\"current_slab_allocations\":%" PRIu64
        ",\"high_water_slab_allocations\":%" PRIu64
        ",\"current_slab_bytes\":%" PRIu64 ",\"high_water_slab_bytes\":%" PRIu64
        ",\"total_slab_acquired_bytes\":%" PRIu64
        ",\"total_slab_released_bytes\":%" PRIu64
        ",\"current_pool_reservations\":%" PRIu64
        ",\"high_water_pool_reservations\":%" PRIu64
        ",\"current_pool_reserved_bytes\":%" PRIu64
        ",\"high_water_pool_reserved_bytes\":%" PRIu64
        ",\"total_pool_reserved_bytes\":%" PRIu64
        ",\"total_pool_released_bytes\":%" PRIu64
        ",\"current_queue_allocations\":%" PRIu64
        ",\"high_water_queue_allocations\":%" PRIu64
        ",\"current_queue_bytes\":%" PRIu64
        ",\"high_water_queue_bytes\":%" PRIu64
        ",\"total_queue_alloca_bytes\":%" PRIu64
        ",\"total_queue_dealloca_bytes\":%" PRIu64
        ",\"current_buffer_allocations\":%" PRIu64
        ",\"high_water_buffer_allocations\":%" PRIu64
        ",\"current_buffer_bytes\":%" PRIu64
        ",\"high_water_buffer_bytes\":%" PRIu64
        ",\"total_buffer_allocate_bytes\":%" PRIu64
        ",\"total_buffer_free_bytes\":%" PRIu64 "}\n",
        device->physical_device_ordinal, device->event_count,
        device->slab_acquire_count, device->slab_release_count,
        device->pool_reserve_count, device->pool_materialize_count,
        device->pool_release_count, device->pool_wait_count,
        device->queue_alloca_count, device->queue_dealloca_count,
        device->buffer_allocate_count, device->buffer_free_count,
        device->current_slab_allocation_count,
        device->high_water_slab_allocation_count, device->current_slab_bytes,
        device->high_water_slab_bytes, device->total_slab_acquired_bytes,
        device->total_slab_released_bytes,
        device->current_pool_reservation_count,
        device->high_water_pool_reservation_count,
        device->current_pool_reserved_bytes,
        device->high_water_pool_reserved_bytes,
        device->total_pool_reserved_bytes, device->total_pool_released_bytes,
        device->current_queue_allocation_count,
        device->high_water_queue_allocation_count, device->current_queue_bytes,
        device->high_water_queue_bytes, device->total_queue_alloca_bytes,
        device->total_queue_dealloca_bytes,
        device->current_buffer_allocation_count,
        device->high_water_buffer_allocation_count,
        device->current_buffer_bytes, device->high_water_buffer_bytes,
        device->total_buffer_allocate_bytes, device->total_buffer_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->pool_count; ++i) {
    const iree_profile_memory_pool_t* pool = &context->pools[i];
    fprintf(file, "{\"type\":\"memory_pool\",\"kind\":");
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_lifecycle_kind_name(pool->kind)));
    fprintf(file,
            ",\"physical_device_ordinal\":%u,\"pool_id\":%" PRIu64
            ",\"memory_type\":%" PRIu64 ",\"buffer_usage\":%" PRIu64
            ",\"events\":%" PRIu64 ",\"waits\":%" PRIu64
            ",\"materializes\":%" PRIu64 ",\"current_allocations\":%" PRIu64
            ",\"high_water_allocations\":%" PRIu64 ",\"current_bytes\":%" PRIu64
            ",\"high_water_bytes\":%" PRIu64
            ",\"total_allocate_bytes\":%" PRIu64
            ",\"total_free_bytes\":%" PRIu64 "}\n",
            pool->physical_device_ordinal, pool->pool_id, pool->memory_type,
            pool->buffer_usage, pool->event_count, pool->wait_count,
            pool->materialize_count, pool->current_allocation_count,
            pool->high_water_allocation_count, pool->current_bytes,
            pool->high_water_bytes, pool->total_allocate_bytes,
            pool->total_free_bytes);
  }
  for (iree_host_size_t i = 0; i < context->allocation_count; ++i) {
    const iree_profile_memory_allocation_t* allocation =
        &context->allocations[i];
    const int64_t duration_ns =
        allocation->last_host_time_ns >= allocation->first_host_time_ns
            ? allocation->last_host_time_ns - allocation->first_host_time_ns
            : 0;
    fprintf(file, "{\"type\":\"memory_allocation\",\"kind\":");
    iree_profile_fprint_json_string(
        file, iree_make_cstring_view(
                  iree_profile_memory_lifecycle_kind_name(allocation->kind)));
    fprintf(
        file,
        ",\"physical_device_ordinal\":%u,\"allocation_id\":%" PRIu64
        ",\"pool_id\":%" PRIu64 ",\"backing_id\":%" PRIu64
        ",\"memory_type\":%" PRIu64 ",\"buffer_usage\":%" PRIu64
        ",\"events\":%" PRIu64 ",\"waits\":%" PRIu64
        ",\"materializes\":%" PRIu64
        ",\"live_at_end\":%s"
        ",\"current_bytes\":%" PRIu64 ",\"high_water_bytes\":%" PRIu64
        ",\"total_allocate_bytes\":%" PRIu64 ",\"total_free_bytes\":%" PRIu64
        ",\"first_event_id\":%" PRIu64 ",\"last_event_id\":%" PRIu64
        ",\"first_host_time_ns\":%" PRId64 ",\"last_host_time_ns\":%" PRId64
        ",\"duration_ns\":%" PRId64 ",\"first_submission_id\":%" PRIu64
        ",\"last_submission_id\":%" PRIu64 "}\n",
        allocation->physical_device_ordinal, allocation->allocation_id,
        allocation->pool_id, allocation->backing_id, allocation->memory_type,
        allocation->buffer_usage, allocation->event_count,
        allocation->wait_count, allocation->materialize_count,
        allocation->current_bytes != 0 ? "true" : "false",
        allocation->current_bytes, allocation->high_water_bytes,
        allocation->total_allocate_bytes, allocation->total_free_bytes,
        allocation->first_event_id, allocation->last_event_id,
        allocation->first_host_time_ns, allocation->last_host_time_ns,
        duration_ns, allocation->first_submission_id,
        allocation->last_submission_id);
  }
}

static iree_status_t iree_profile_memory_file(iree_string_view_t path,
                                              iree_string_view_t format,
                                              iree_string_view_t filter,
                                              int64_t id_filter, FILE* file,
                                              iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(host_allocator, &context);
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_memory_process_event_records(
          &context, &record, filter, id_filter, is_jsonl, file);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      status =
          iree_profile_memory_print_text(&context, filter, id_filter, file);
    } else {
      iree_profile_memory_print_jsonl_summary(&context, filter, id_filter,
                                              file);
    }
  }

  iree_profile_memory_context_deinitialize(&context);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t iree_profile_summary_file(
    iree_string_view_t path, iree_string_view_t format, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(host_allocator, &summary);
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_summary_process_record(&summary, &record);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_print_summary_text(&summary, file);
    } else {
      iree_profile_print_summary_jsonl(&summary, file);
    }
  }

  iree_profile_summary_deinitialize(&summary);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t iree_profile_projection_file(
    iree_string_view_t path, iree_string_view_t format,
    iree_string_view_t filter, iree_profile_projection_mode_t projection_mode,
    int64_t id_filter, bool emit_events, FILE* file,
    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }
  if (emit_events && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--dispatch_events requires --format=jsonl");
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t first_record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &first_record_offset);
  }

  iree_profile_dispatch_context_t context;
  iree_profile_dispatch_context_initialize(host_allocator, &context);

  iree_host_size_t record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_metadata_record(&context, &record);
      record_offset = next_record_offset;
    }
  }

  record_offset = first_record_offset;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      status = iree_profile_dispatch_process_events_record(
          &context, &record, filter, projection_mode, id_filter, emit_events,
          file);
      record_offset = next_record_offset;
    }
  }

  if (iree_status_is_ok(status)) {
    switch (projection_mode) {
      case IREE_PROFILE_PROJECTION_MODE_DISPATCH:
        if (is_text) {
          status = iree_profile_dispatch_print_text(&context, filter, file);
        } else {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
          if (!emit_events) {
            status =
                iree_profile_dispatch_print_jsonl_aggregates(&context, file);
          }
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_EXECUTABLE:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status = iree_profile_executable_print_text(&context, filter,
                                                      id_filter, file);
        } else {
          status = iree_profile_executable_print_jsonl(&context, filter,
                                                       id_filter, file);
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_COMMAND:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status = iree_profile_command_print_text(&context, filter, id_filter,
                                                   file);
        } else {
          status = iree_profile_command_print_jsonl(&context, filter, id_filter,
                                                    file);
        }
        break;
      case IREE_PROFILE_PROJECTION_MODE_QUEUE:
        if (emit_events) {
          iree_profile_dispatch_print_jsonl_summary(&context, filter,
                                                    emit_events, file);
        } else if (is_text) {
          status =
              iree_profile_queue_print_text(&context, filter, id_filter, file);
        } else {
          status =
              iree_profile_queue_print_jsonl(&context, filter, id_filter, file);
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported profile projection mode %d",
                                  (int)projection_mode);
        break;
    }
  }

  iree_profile_dispatch_context_deinitialize(&context);
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t iree_profile_cat_file(iree_string_view_t path,
                                           iree_string_view_t format,
                                           FILE* file,
                                           iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }
  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_dump_header_text(&header, file);
    } else {
      iree_profile_dump_header_jsonl(&header, file);
    }
  }

  iree_host_size_t record_index = 0;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      if (is_text) {
        iree_profile_dump_record_text(record_index, &record, file);
      } else {
        iree_profile_dump_record_jsonl(record_index, &record, file);
      }
      record_offset = next_record_offset;
      ++record_index;
    }
  }

  iree_io_file_contents_free(file_contents);
  return status;
}

static const char kIreeProfileUsage[] =
    "Inspects IREE HAL profile bundles.\n"
    "\n"
    "A .ireeprof bundle is a HAL-native profiling artifact. It contains\n"
    "session metadata, physical-device and queue metadata, executable/export\n"
    "metadata, command-buffer metadata and operations, clock-correlation\n"
    "samples, host queue-operation events, device-timestamped dispatch "
    "events,\n"
    "and optional memory lifecycle events. The projection commands below let "
    "you enter "
    "from\n"
    "the object you care about and then cross-reference ids in JSONL.\n"
    "\n"
    "Usage:\n"
    "  iree-profile summary [--format=text|jsonl] <file.ireeprof>\n"
    "  iree-profile executable [--format=text|jsonl] [--id=executable_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile dispatch [--format=text|jsonl] [--id=event_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile command [--format=text|jsonl] [--id=command_buffer_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile memory [--format=text|jsonl] "
    "[--id=event_or_allocation_id]\n"
    "      [--filter=event_type_pattern] <file.ireeprof>\n"
    "  iree-profile queue [--format=text|jsonl] [--id=submission_id]\n"
    "      [--filter=pattern] [--dispatch_events] <file.ireeprof>\n"
    "  iree-profile cat [--format=text|jsonl] <file.ireeprof>\n"
    "  iree-profile --agent_md\n"
    "\n"
    "Commands:\n"
    "  summary      Bundle health, metadata counts, clock fit, and per-device\n"
    "               dispatch timing totals.\n"
    "  executable   Executable/export catalog joined with dispatch timing. "
    "Use\n"
    "               this when optimizing a specific kernel name or export.\n"
    "  dispatch     Per-export dispatch timing aggregates, or individual "
    "events\n"
    "               with --dispatch_events --format=jsonl.\n"
    "  command      Recorded command-buffer operations, metadata, and "
    "per-execution\n"
    "               dispatch spans. Use this to inspect copy/fill/update/"
    "barrier\n"
    "               structure and drill into dispatch timings.\n"
    "  queue        Queue operation events and dispatch-derived submission "
    "spans.\n"
    "  memory       Memory lifecycle events and high-water summaries for "
    "slabs,\n"
    "               pools, async queue allocations, and HAL buffers. "
    "Live-at-end counts\n"
    "               are capture-window state, not required leaks.\n"
    "  cat          Raw bundle record dump for format archaeology/debugging.\n"
    "\n"
    "Important flags:\n"
    "  --format=text|jsonl     Text for humans, JSONL for tools and agents.\n"
    "  --filter=pattern        Wildcard over dispatch/export keys, command op\n"
    "                          names/keys, or queue/memory event type names,\n"
    "                          such as '*softmax*' or 'copy'.\n"
    "  --id=N                  dispatch: event_id; executable: executable_id;\n"
    "                          command: command_buffer_id; memory: "
    "event_id or\n"
    "                          allocation_id; queue: submission_id.\n"
    "  --dispatch_events       Stream individual dispatch_event rows instead "
    "of\n"
    "                          only aggregate projection rows. Requires "
    "JSONL.\n"
    "  --agent_md              Print a Markdown guide optimized for "
    "AGENTS.md.\n"
    "\n"
    "Capture examples:\n"
    "  iree-benchmark-module --device=amdgpu --module=model.vmfb \\\n"
    "      --function=main --benchmark_min_time=20x \\\n"
    "      --device_profiling_mode=queue \\\n"
    "      --device_profiling_output=/tmp/model.ireeprof\n"
    "\n"
    "Embedding examples:\n"
    "  Use iree_hal_profile_file_sink_create to create a file-backed sink,\n"
    "  set iree_hal_device_profiling_options_t::sink, then call\n"
    "  iree_hal_device_profiling_begin/end around the work to capture.\n"
    "\n"
    "Analysis examples:\n"
    "  iree-profile summary /tmp/model.ireeprof\n"
    "  iree-profile executable --filter='*matmul*' /tmp/model.ireeprof\n"
    "  iree-profile dispatch --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type==\"dispatch_group\") | {key,avg_ns,count}'\n"
    "  iree-profile queue --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type==\"queue_event\" or "
    ".type==\"queue_submission\")'\n"
    "  iree-profile memory --format=jsonl /tmp/model.ireeprof | \\\n"
    "      jq 'select(.type==\"memory_pool\") | \\\n"
    "          {kind,physical_device_ordinal,pool_id,"
    "high_water_bytes,waits}'\n"
    "  iree-profile command --id=1 --format=jsonl --dispatch_events \\\n"
    "      /tmp/model.ireeprof | jq 'select(.type==\"dispatch_event\")'\n"
    "\n"
    "Use `iree-profile --agent_md` for a Markdown playbook with JSONL record\n"
    "types and cross-reference recipes.\n";

static void iree_profile_fprint_usage(FILE* file) {
  fputs("iree-profile\n", file);
  fputs(kIreeProfileUsage, file);
}

static void iree_profile_print_agent_markdown(FILE* file) {
  fputs(
      "# IREE HAL Profiling With iree-profile\n"
      "\n"
      "Use `iree-profile` to inspect `.ireeprof` bundles emitted by IREE HAL\n"
      "device profiling. Prefer `--format=jsonl` for automated analysis: each\n"
      "line is one independent JSON record that can be filtered with `jq`.\n"
      "\n"
      "## Capture\n"
      "\n"
      "Capture from `iree-benchmark-module` with HAL queue profiling "
      "enabled:\n"
      "\n"
      "```bash\n"
      "iree-benchmark-module --device=amdgpu --module=model.vmfb \\\n"
      "  --function=main --benchmark_min_time=20x \\\n"
      "  --device_profiling_mode=queue \\\n"
      "  --device_profiling_output=/tmp/model.ireeprof\n"
      "```\n"
      "\n"
      "Embedding applications can capture the same bundle through the HAL "
      "API:\n"
      "\n"
      "```c\n"
      "iree_io_file_handle_t* file_handle = NULL;\n"
      "iree_hal_profile_sink_t* sink = NULL;\n"
      "IREE_RETURN_IF_ERROR(iree_io_file_handle_create(\n"
      "    IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN,\n"
      "    IREE_SV(\"/tmp/model.ireeprof\"), 0, host_allocator, "
      "&file_handle));\n"
      "IREE_RETURN_IF_ERROR(iree_hal_profile_file_sink_create(\n"
      "    file_handle, host_allocator, &sink));\n"
      "iree_hal_device_profiling_options_t options = {0};\n"
      "options.mode = IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;\n"
      "options.sink = sink;\n"
      "IREE_RETURN_IF_ERROR(iree_hal_device_profiling_begin(device, "
      "&options));\n"
      "/* Run workload. */\n"
      "IREE_RETURN_IF_ERROR(iree_hal_device_profiling_end(device));\n"
      "iree_hal_profile_sink_release(sink);\n"
      "iree_io_file_handle_release(file_handle);\n"
      "```\n"
      "\n"
      "## Commands\n"
      "\n"
      "- `summary` checks bundle health, metadata counts, dispatch counts, "
      "clock\n"
      "  correlation, and per-device timing totals.\n"
      "- `executable` lists executable/export metadata and joins export ids "
      "to\n"
      "  dispatch timing aggregates.\n"
      "- `dispatch` groups timings by executable export, or emits every "
      "dispatch\n"
      "  event with `--dispatch_events`.\n"
      "- `command` lists command-buffer metadata, static operation records, "
      "and\n"
      "  execution spans grouped by `command_buffer_id` and `submission_id`.\n"
      "- `queue` groups dispatch-derived queue submissions by physical queue "
      "and\n"
      "  `submission_id`, and emits host queue-operation events when the "
      "producer\n"
      "  provides them. Queue events include operation type, wait/signal "
      "counts,\n"
      "  dependency strategy, payload bytes, and related object ids.\n"
      "- `memory` summarizes slab/provider events, pool reservation events, "
      "async\n"
      "  queue alloca/dealloca, and synchronous HAL buffer allocation/free "
      "high-water behavior. With `--format=jsonl` it emits individual\n"
      "  `memory_event` rows before the summary and then emits `memory_pool` "
      "and\n"
      "  `memory_allocation` rows for pool/provider and lifecycle drilldown. "
      "Live\n"
      "  allocations at profile end are capture-window state and may be "
      "normal\n"
      "  for retained outputs or embedding-managed buffers.\n"
      "- `cat` dumps raw bundle records and is mainly for format debugging.\n"
      "\n"
      "## JSONL Record Types\n"
      "\n"
      "- `summary` emits `summary` and `device_summary`.\n"
      "- `executable --format=jsonl` emits `executable_summary`, "
      "`executable`,\n"
      "  `executable_export`, and `executable_export_dispatch_group`.\n"
      "- `dispatch --format=jsonl` emits `dispatch_summary` and "
      "`dispatch_group`.\n"
      "- `dispatch --format=jsonl --dispatch_events` emits `dispatch_event` "
      "rows\n"
      "  followed by `dispatch_summary`.\n"
      "- `command --format=jsonl` emits `command_summary`, `command_buffer`,\n"
      "  `command_operation`, and `command_execution`.\n"
      "- `queue --format=jsonl` emits `queue_summary`, `queue`,\n"
      "  `queue_submission`, and `queue_event`.\n"
      "- `memory --format=jsonl` emits `memory_event`, `memory_summary`,\n"
      "  `memory_device`, `memory_pool`, and `memory_allocation`.\n"
      "\n"
      "## Cross-Reference Keys\n"
      "\n"
      "- `dispatch_event.event_id` uniquely identifies one dispatch event in "
      "the\n"
      "  profile stream. Use `dispatch --id=<event_id>` to isolate it.\n"
      "- `dispatch_event.submission_id` joins dispatches to "
      "`queue_submission`.\n"
      "  Use `queue --id=<submission_id>` or "
      "`queue --id=<submission_id> --dispatch_events` to inspect a "
      "submission.\n"
      "- `dispatch_event.command_buffer_id` joins dispatches to "
      "`command_buffer`\n"
      "  and `command_execution`. Use `command --id=<command_buffer_id>`.\n"
      "- `dispatch_event.command_index` is the ordinal of the dispatch within "
      "the\n"
      "  command buffer when the event came from reusable command-buffer "
      "replay.\n"
      "- `command_operation.command_buffer_id` and "
      "`command_operation.command_index`\n"
      "  join recorded command-buffer operations to dispatch events. "
      "`op` and\n"
      "  `key` can be filtered with `command --filter=<pattern>`.\n"
      "- `dispatch_event.executable_id` plus `dispatch_event.export_ordinal` "
      "joins\n"
      "  to `executable_export`. The `key` field is the export name when "
      "present,\n"
      "  otherwise a stable numeric fallback.\n"
      "- `physical_device_ordinal`, `queue_ordinal`, and `stream_id` identify "
      "the\n"
      "  producing physical queue within the session.\n"
      "- `queue_event.submission_id` joins queue operation metadata to "
      "`queue_submission`\n"
      "  and dispatch events. `queue_event.command_buffer_id` joins execute "
      "events\n"
      "  to `command_buffer`; `queue_event.allocation_id` joins alloca/"
      "dealloca\n"
      "  events to memory events.\n"
      "- `memory_event.submission_id` joins queue alloca/dealloca events to "
      "queue\n"
      "  submissions when nonzero. `memory_event.allocation_id` joins all "
      "events\n"
      "  associated with one producer-defined allocation handle. "
      "`memory_pool`\n"
      "  groups by producer pool/provider id and memory type. "
      "`memory_allocation`\n"
      "  groups capture-window lifecycle rows by kind, allocation id, and "
      "pool id.\n"
      "  `memory_device`\n"
      "  live/high-water fields are computed over the capture window, so "
      "they\n"
      "  need not balance to zero if buffers outlive profiling.\n"
      "\n"
      "## Recipes\n"
      "\n"
      "Find the slowest kernel/export groups:\n"
      "\n"
      "```bash\n"
      "iree-profile dispatch --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"dispatch_group\")) | \\\n"
      "         sort_by(.total_ns) | reverse[:20] | \\\n"
      "         map({key,count,avg_ns,total_ns,max_ns})'\n"
      "```\n"
      "\n"
      "Show every dispatch of one kernel name pattern:\n"
      "\n"
      "```bash\n"
      "iree-profile dispatch --format=jsonl --dispatch_events \\\n"
      "  --filter='*softmax*' /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"dispatch_event\") | \\\n"
      "      {event_id,submission_id,command_buffer_id,command_index,key,"
      "duration_ns}'\n"
      "```\n"
      "\n"
      "Find expensive command-buffer executions:\n"
      "\n"
      "```bash\n"
      "iree-profile command --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"command_execution\")) | \\\n"
      "         sort_by(.span_ns) | reverse[:20] | \\\n"
      "         map({command_buffer_id,submission_id,span_ns,dispatches})'\n"
      "```\n"
      "\n"
      "Drill into all dispatches for command buffer 1:\n"
      "\n"
      "```bash\n"
      "iree-profile command --id=1 --format=jsonl --dispatch_events \\\n"
      "  /tmp/model.ireeprof | jq 'select(.type==\"dispatch_event\")'\n"
      "```\n"
      "\n"
      "Show the recorded operation stream for command buffer 1:\n"
      "\n"
      "```bash\n"
      "iree-profile command --id=1 --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"command_operation\") | \\\n"
      "      {command_index,op,key,length,workgroup_count}'\n"
      "```\n"
      "\n"
      "Find queue submissions with long device spans:\n"
      "\n"
      "```bash\n"
      "iree-profile queue --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"queue_submission\")) | \\\n"
      "         sort_by(.span_ns) | reverse[:20] | \\\n"
      "         map({submission_id,physical_device_ordinal,queue_ordinal,"
      "span_ns,dispatches})'\n"
      "```\n"
      "\n"
      "Drill from queue submission 4 to its dispatch events:\n"
      "\n"
      "```bash\n"
      "iree-profile queue --id=4 --format=jsonl --dispatch_events \\\n"
      "  /tmp/model.ireeprof | jq 'select(.type==\"dispatch_event\")'\n"
      "```\n"
      "\n"
      "Inspect allocation high-water by physical device:\n"
      "\n"
      "```bash\n"
      "iree-profile memory --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"memory_device\") | \\\n"
      "      {physical_device_ordinal,current_queue_allocations,"
      "high_water_queue_bytes,current_buffer_allocations,"
      "high_water_buffer_bytes}'\n"
      "```\n"
      "\n"
      "Find pools/providers with the largest memory high-water:\n"
      "\n"
      "```bash\n"
      "iree-profile memory --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq -s 'map(select(.type==\"memory_pool\")) | \\\n"
      "         sort_by(.high_water_bytes) | reverse[:20] | \\\n"
      "         map({kind,physical_device_ordinal,pool_id,memory_type,"
      "high_water_bytes,waits})'\n"
      "```\n"
      "\n"
      "Trace one allocation lifecycle:\n"
      "\n"
      "```bash\n"
      "iree-profile memory --format=jsonl --id=<allocation_id> \\\n"
      "  /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"memory_event\" or "
      ".type==\"memory_allocation\")'\n"
      "```\n"
      "\n"
      "Map executable ids and export ordinals to names:\n"
      "\n"
      "```bash\n"
      "iree-profile executable --format=jsonl /tmp/model.ireeprof | \\\n"
      "  jq 'select(.type==\"executable_export\") | \\\n"
      "      {executable_id,export_ordinal,key,binding_count,workgroup_size}'\n"
      "```\n"
      "\n"
      "## Notes For Agents\n"
      "\n"
      "Treat `summary` as the first sanity check: if clock fitting is "
      "unavailable\n"
      "or dispatch timestamps are invalid, timing conclusions are suspect. "
      "For\n"
      "automated analysis, prefer entering through `queue`, `command`,\n"
      "`executable`, or `dispatch` with JSONL and then drilling into\n"
      "`--dispatch_events` using the ids above. Do not infer queue ordering "
      "from\n"
      "record order alone; HAL queue ordering is established by semaphore "
      "edges,\n"
      "and full queue-edge records require producer support beyond the "
      "current\n"
      "dispatch-derived projections.\n",
      file);
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage("iree-profile", kIreeProfileUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agent_md) {
    iree_profile_print_agent_markdown(stdout);
    fflush(stdout);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(exit_code);
    return exit_code;
  }

  iree_string_view_t command = IREE_SV("cat");
  iree_string_view_t path = iree_string_view_empty();
  if (argc == 2) {
    path = iree_make_cstring_view(argv[1]);
  } else if (argc == 3) {
    command = iree_make_cstring_view(argv[1]);
    path = iree_make_cstring_view(argv[2]);
  }

  iree_status_t status = iree_ok_status();
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "Error: expected profile bundle path.\n");
    iree_profile_fprint_usage(stderr);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected profile bundle path");
  } else if (iree_string_view_is_empty(path)) {
    fprintf(stderr, "Error: missing profile bundle path.\n");
    iree_profile_fprint_usage(stderr);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing profile bundle path");
  } else if (!iree_string_view_equal(command, IREE_SV("cat")) &&
             !iree_string_view_equal(command, IREE_SV("command")) &&
             !iree_string_view_equal(command, IREE_SV("dispatch")) &&
             !iree_string_view_equal(command, IREE_SV("executable")) &&
             !iree_string_view_equal(command, IREE_SV("memory")) &&
             !iree_string_view_equal(command, IREE_SV("queue")) &&
             !iree_string_view_equal(command, IREE_SV("summary"))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported iree-profile command '%.*s'",
                              (int)command.size, command.data);
  }

  if (iree_status_is_ok(status)) {
    if (iree_string_view_equal(command, IREE_SV("summary"))) {
      status = iree_profile_summary_file(
          path, iree_make_cstring_view(FLAG_format), stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("command"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_COMMAND, FLAG_id, FLAG_dispatch_events,
          stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("dispatch"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_DISPATCH, FLAG_id, FLAG_dispatch_events,
          stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("executable"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_EXECUTABLE, FLAG_id,
          FLAG_dispatch_events, stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("memory"))) {
      status = iree_profile_memory_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter), FLAG_id, stdout, host_allocator);
    } else if (iree_string_view_equal(command, IREE_SV("queue"))) {
      status = iree_profile_projection_file(
          path, iree_make_cstring_view(FLAG_format),
          iree_make_cstring_view(FLAG_filter),
          IREE_PROFILE_PROJECTION_MODE_QUEUE, FLAG_id, FLAG_dispatch_events,
          stdout, host_allocator);
    } else {
      status = iree_profile_cat_file(path, iree_make_cstring_view(FLAG_format),
                                     stdout, host_allocator);
    }
  }

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
