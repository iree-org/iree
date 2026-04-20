// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_PROFILE_OPTIONS_H_
#define IREE_HAL_PROFILE_OPTIONS_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_profile_sink_t iree_hal_profile_sink_t;

// Bitfield selecting HAL-native structured profiling data families.
//
// These bits are not mutually-exclusive modes. Each bit requests one family of
// records or artifacts emitted through iree_hal_profile_sink_t. Not all
// implementations support all families.
typedef uint64_t iree_hal_device_profiling_data_families_t;
enum iree_hal_device_profiling_data_family_bits_t {
  IREE_HAL_DEVICE_PROFILING_DATA_NONE = 0u,

  // Host-timestamped queue operation records such as submissions, dependency
  // strategy, and encoded operation counts. Producers may retain these as an
  // aggregate lossy stream and report dropped records with TRUNCATED chunks.
  IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS = 1ull << 0,

  // Host-timestamped execution spans for work performed by the host, such as
  // CPU/local dispatch bodies or host-side command buffer replay.
  IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS = 1ull << 1,

  // Device-timestamped queue operation spans showing when queue-visible work
  // started and completed in the device timestamp domain. These are precise
  // execution timeline records: producers should fail the profiled operation or
  // session when they cannot retain complete selected events.
  IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS = 1ull << 2,

  // Device-timestamped dispatch execution events. This does not request
  // hardware/software counter samples by itself. These are precise execution
  // timeline records: producers should fail the profiled operation or session
  // when they cannot retain complete selected events.
  IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS = 1ull << 3,

  // Explicitly selected hardware/software counter samples. Requested counters
  // are described by |counter_sets|.
  IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES = 1ull << 4,

  // Executable/code-object/export metadata needed for offline analysis. Some
  // producers also use this as the cheap metadata family for command-buffer
  // records needed to interpret command-index joins. Producers may emit this
  // implicitly when another requested family references executable ids, but
  // this bit lets callers request metadata by itself.
  IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA = 1ull << 5,

  // Heavyweight executable trace artifacts such as instruction/thread traces
  // for selected dispatches or command-buffer ranges. This can allocate large
  // device buffers and inject additional queue packets and should only be
  // enabled with a narrow capture filter.
  IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES = 1ull << 6,

  // Memory allocation and reservation lifecycle records. Producers may retain
  // these as an aggregate lossy stream and report dropped records with
  // TRUNCATED chunks.
  IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS = 1ull << 7,

  // Periodic physical-device metric samples such as clocks, temperature,
  // power, memory occupancy, utilization, and bandwidth. Producers should emit
  // source and descriptor metadata so profile bundles remain self-describing.
  IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS = 1ull << 8,
};

// Bitfield selecting producer-side profiling behavior that is not itself a
// durable profile record family.
typedef uint32_t iree_hal_device_profiling_flags_t;
enum iree_hal_device_profiling_flag_bits_t {
  IREE_HAL_DEVICE_PROFILING_FLAG_NONE = 0u,

  // Requests the producer's cheapest useful execution-statistics record set.
  // Producers expand this into ordinary profile chunks such as queue, dispatch,
  // host-execution, and executable metadata records. No statistics-specific
  // profile chunks are emitted.
  IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS = 1u << 0,
};

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

// Controls profiling options.
//
// All pointer and string-view fields are borrowed and only need to remain valid
// until iree_hal_device_profiling_begin returns. Implementations that need a
// value after returning success must retain, copy, or resolve it into
// implementation-owned session state before returning.
typedef struct iree_hal_device_profiling_options_t {
  // Flags selecting producer-side profiling behavior.
  iree_hal_device_profiling_flags_t flags;

  // HAL-native structured data families requested by the caller.
  iree_hal_device_profiling_data_families_t data_families;

  // Programmatic sink receiving HAL-native profiling chunks.
  // The caller retains ownership of the sink for the duration of the
  // profiling_begin call. Implementations that keep the sink beyond the call
  // must retain it and release it during profiling_end or teardown.
  iree_hal_profile_sink_t* sink;

  // Optional borrowed filter selecting operations that should emit heavy
  // profile artifacts. A zero-initialized filter matches all operations.
  // Implementations that retain the filter for session matching must copy any
  // string views it contains before returning from profiling_begin.
  iree_hal_profile_capture_filter_t capture_filter;

  // Number of explicitly requested hardware/software counter sets.
  // Must be nonzero when requesting
  // IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES.
  iree_host_size_t counter_set_count;

  // Borrowed begin-call-only array of explicitly requested counter sets.
  // Implementations must either capture every requested counter set exactly or
  // fail profiling_begin; silently dropping counters would make the profile
  // bundle misleading.
  const iree_hal_profile_counter_set_selection_t* counter_sets;
} iree_hal_device_profiling_options_t;

// Opaque storage backing cloned profiling options.
typedef struct iree_hal_device_profiling_options_storage_t
    iree_hal_device_profiling_options_storage_t;

// Clones |source_options| and all borrowed nested storage into
// |host_allocator|.
//
// |out_options| receives a value type whose pointer/string fields reference
// storage owned by |out_storage|. If |source_options->sink| is non-NULL it is
// retained and will be released by
// iree_hal_device_profiling_options_storage_free. Callers must not separately
// release any pointers in |out_options|.
IREE_API_EXPORT iree_status_t iree_hal_device_profiling_options_clone(
    const iree_hal_device_profiling_options_t* source_options,
    iree_allocator_t host_allocator,
    iree_hal_device_profiling_options_t* out_options,
    iree_hal_device_profiling_options_storage_t** out_storage);

// Frees storage returned by iree_hal_device_profiling_options_clone.
IREE_API_EXPORT void iree_hal_device_profiling_options_storage_free(
    iree_hal_device_profiling_options_storage_t* storage,
    iree_allocator_t host_allocator);

// Returns true when |options| requests any bits in |data_families|.
static inline bool iree_hal_device_profiling_options_requests_data(
    const iree_hal_device_profiling_options_t* options,
    iree_hal_device_profiling_data_families_t data_families) {
  return iree_any_bit_set(options->data_families, data_families);
}

// Returns true when producers should select their lightweight statistics mode.
static inline bool
iree_hal_device_profiling_options_requests_lightweight_statistics(
    const iree_hal_device_profiling_options_t* options) {
  return iree_all_bits_set(
      options->flags, IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS);
}

// Returns true when |options| requests explicit hardware counter capture.
static inline bool iree_hal_device_profiling_options_requests_counter_samples(
    const iree_hal_device_profiling_options_t* options) {
  return iree_hal_device_profiling_options_requests_data(
      options, IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES);
}

// Returns true when |options| requests executable trace artifacts.
static inline bool iree_hal_device_profiling_options_requests_executable_traces(
    const iree_hal_device_profiling_options_t* options) {
  return iree_hal_device_profiling_options_requests_data(
      options, IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES);
}

// Returns true when |options| requests periodic device metrics.
static inline bool iree_hal_device_profiling_options_requests_device_metrics(
    const iree_hal_device_profiling_options_t* options) {
  return iree_hal_device_profiling_options_requests_data(
      options, IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_PROFILE_OPTIONS_H_
