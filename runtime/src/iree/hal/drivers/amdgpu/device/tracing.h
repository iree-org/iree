// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_TRACING_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_TRACING_H_

#include "iree/hal/drivers/amdgpu/device/support/common.h"
#include "iree/hal/drivers/amdgpu/device/support/signal.h"

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_* Flags and Options
//===----------------------------------------------------------------------===//

// Enables IREE_AMDGPU_TRACE_* macros for instrumented tracing.
#define IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION (1 << 0)

// Enables instrumentation of command buffer control (dispatches, DMA, etc).
// This can have significant code size and runtime overhead and should only be
// used when specifically tracing device-side execution.
#define IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL (1 << 1)

// Enables instrumentation of command buffer execution (dispatches, DMA, etc).
// This can have significant code size and runtime overhead and should only be
// used when specifically tracing device-side execution.
#define IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION \
  ((1 << 2) | IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)

// Tracks all device allocations.
#define IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING (1 << 3)

// Forwards log messages to traces, which will be visible under "Messages" in
// the Tracy UI.
#define IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES (1 << 4)

// Enables the IREE_HAL_AMDGPU_DBG print macros. May massively increase binary
// size and decrease performance.
#define IREE_HAL_AMDGPU_TRACING_FEATURE_DEBUG_MESSAGES \
  ((1 << 5) | IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES)

// TODO(benvanik): expose as a friendly option matching the host mode.
// For now we need the compilation to match and there are extra flags required
// for that.
#if 0
#define IREE_HAL_AMDGPU_TRACING_FEATURES                 \
  (IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION |     \
   IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL |      \
   IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION |    \
   IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING | \
   IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES |        \
   IREE_HAL_AMDGPU_TRACING_FEATURE_DEBUG_MESSAGES)
#else
#define IREE_HAL_AMDGPU_TRACING_FEATURES 0
#endif

// Tests whether one or more tracing features have been enabled in the build.
//
// Example:
//  #if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
//     IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES)
//  <<code that should only run when LOG_MESSAGES is enabled>>
//  #endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES
#define IREE_HAL_AMDGPU_HAS_TRACING_FEATURE(feature_bits) \
  IREE_AMDGPU_ALL_BITS_SET(IREE_HAL_AMDGPU_TRACING_FEATURES, (feature_bits))

//===----------------------------------------------------------------------===//
// Tracing Buffer Definitions
//===----------------------------------------------------------------------===//

// A timestamp in the domain of the agent who owns the buffer the trace event
// is recorded in. Each agent may have differing times that need to be converted
// into the system domain on the host.
typedef uint64_t iree_hal_amdgpu_trace_agent_timestamp_t;

// A time range bounded by two timestamps.
typedef struct iree_hal_amdgpu_trace_agent_time_range_t {
  iree_hal_amdgpu_trace_agent_timestamp_t begin;
  iree_hal_amdgpu_trace_agent_timestamp_t end;
} iree_hal_amdgpu_trace_agent_time_range_t;

// A process-unique ID assigned to an agent that executes execution zones.
// In Tracy this is a GPU context.
typedef uint8_t iree_hal_amdgpu_trace_executor_id_t;

// An outstanding execution zone query ID.
// As execution events are issued an ID is reserved and at some point in the
// future after execution has completed the ID is used to match up the acquired
// timing information for the event. Being 16-bit we have a limited number of
// outstanding IDs but as we scope them per trace buffer we should be ok.
typedef uint16_t iree_hal_amdgpu_trace_execution_query_id_t;

// Indicates that a query ID is not used.
#define IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID \
  ((iree_hal_amdgpu_trace_execution_query_id_t)0xFFFF)

// An 0xAABBGGRR color used when presenting messages and zones in a tracing UI.
// 0x0 can (usually) be used to indicate "default". Alpha may be ignored but
// should be 0xFF in most cases.
typedef uint32_t iree_hal_amdgpu_trace_color_t;

// A pointer that lives within the read-only data segment of the device library
// code object.
//
// The target memory may not be accessible (or may be slow) as the code object
// is loaded onto the GPU agent. The host tracing infrastructure creates a
// shadow copy of the code in host memory and adjusts the address from the
// device into that shadow such that it can access it locally.
//
// A special case is when the pointer is outside of the loaded code object
// range. When translating the host will pass-through any such pointers without
// modifying them to allow for host pointers to be round-tripped from the host.
// In this way calling a pointer an iree_hal_amdgpu_trace_rodata_ptr_t really
// just means the host must try to translate it before dereferencing it instead
// of strictly saying it's in code object memory.
//
// Example translation:
//  uint64_t code_base = 0;
//  err = loader.hsa_ven_amd_loader_loaded_code_object_get_info(
//      loaded_code_object,
//      HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &code_base);
//  const uint8_t* code_shadow = malloc(...); // + copy
//  iree_hal_amdgpu_trace_rodata_ptr_t device_ptr = ...;
//  const uint8_t* host_ptr = (const uint8_t*)(device_ptr - code_base +
//                                             (uint64_t)code_shadow);
typedef uint64_t iree_hal_amdgpu_trace_rodata_ptr_t;

// A NUL-terminated string literal stored in the code object data segment.
// This must be translated into the code object shadow copy prior to using on
// the host.
typedef iree_hal_amdgpu_trace_rodata_ptr_t
    iree_hal_amdgpu_trace_string_literal_ptr_t;

// Static information about a trace zone source location.
// Tracy and other tools require the source location and its contained strings
// to have process lifetime. Since the code object rodata segment they are
// stored in will be unloaded as HSA is shut down we create a shadow copy that
// we can persist in host memory until the process exits so that tracing tools
// can access it.
//
// NOTE: this matches the Tracy expected source location structure exactly so
// that we can pass it unmodified. Tracy uses the pointer of the source location
// for several lookup tables.
typedef struct iree_hal_amdgpu_trace_src_loc_t {
  const char* name;
  const char* function;
  const char* file;
  uint32_t line;
  iree_hal_amdgpu_trace_color_t color;
} iree_hal_amdgpu_trace_src_loc_t;

// A iree_hal_amdgpu_trace_rodata_ptr_t that specifically references a static
// iree_hal_amdgpu_trace_src_loc_t structure in the rodata segment. Note that
// any pointers nested within the target src_loc are also in the rodata segment.
typedef iree_hal_amdgpu_trace_rodata_ptr_t iree_hal_amdgpu_trace_src_loc_ptr_t;

// Matches Tracy's PlotFormatType enum.
typedef uint8_t iree_hal_amdgpu_trace_plot_type_t;
enum iree_hal_amdgpu_trace_plot_type_e {
  // Values will be displayed as plain numbers.
  IREE_HAL_AMDGPU_TRACE_PLOT_TYPE_NUMBER = 0,
  // Treats the values as memory sizes. Will display kilobytes, megabytes, etc.
  IREE_HAL_AMDGPU_TRACE_PLOT_TYPE_MEMORY = 1,
  // Values will be displayed as percentage with value 100 being equal to 100%.
  IREE_HAL_AMDGPU_TRACE_PLOT_TYPE_PERCENTAGE = 2,
};

// Controls plot display and accumulation behavior.
typedef uint8_t iree_hal_amdgpu_trace_plot_flags_t;
enum iree_hal_amdgpu_trace_plot_flag_bits_e {
  // Plot has discrete steps instead of being interpolated/smooth.
  IREE_HAL_AMDGPU_TRACE_PLOT_FLAG_DISCRETE = 1u << 0,
  // Plot has its display area filled with a solid color.
  IREE_HAL_AMDGPU_TRACE_PLOT_FLAG_FILL = 1u << 1,
};

// Event type used to interpret the remainder of the event data.
typedef uint8_t iree_hal_amdgpu_trace_event_type_t;
enum {
  IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN = 0u,
  IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END,
  IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64,
  IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL,
  IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC,
  IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN,
  IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END,
  IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH,
  IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY_BATCH,
  IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC,
  IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE,
  IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL,
  IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC,
  IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG,
  IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64,
};

// Begins a trace zone and pushes it onto the zone stack.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_zone_begin_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN
  iree_hal_amdgpu_trace_event_type_t event_type;
  uint8_t reserved[7];  // may be uninitialized
  // Timestamp the zone begins at.
  iree_hal_amdgpu_trace_agent_timestamp_t timestamp;
  // Source location of the zone being entered.
  iree_hal_amdgpu_trace_src_loc_ptr_t src_loc;
} iree_hal_amdgpu_trace_zone_begin_t;

// Ends the trace zone on the top of the zone stack.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_zone_end_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END
  iree_hal_amdgpu_trace_event_type_t event_type;
  uint8_t reserved[7];  // may be uninitialized
  // Timestamp the zone ends at.
  iree_hal_amdgpu_trace_agent_timestamp_t timestamp;
} iree_hal_amdgpu_trace_zone_end_t;

// Appends an i64 value to the zone on the top of the zone stack.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_zone_value_i64_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64
  iree_hal_amdgpu_trace_event_type_t event_type;
  uint8_t reserved[7];  // may be uninitialized
  // Payload value attached to the zone.
  uint64_t value;
} iree_hal_amdgpu_trace_zone_value_i64_t;

// Appends a string value to the zone on the top of the zone stack.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_trace_zone_value_text_literal_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL
  iree_hal_amdgpu_trace_event_type_t event_type;
  uint8_t reserved[3];
  // Payload value attached to the zone.
  // NUL terminated. Must be stored in the code object data segment.
  iree_hal_amdgpu_trace_string_literal_ptr_t value;
} iree_hal_amdgpu_trace_zone_value_text_literal_t;

// Appends a string value to the zone on the top of the zone stack.
// The contents are embedded in the trace buffer to support dynamically
// generated values.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_trace_zone_value_text_dynamic_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC
  iree_hal_amdgpu_trace_event_type_t event_type;
  uint8_t reserved[3];
  // Length of the value in characters.
  uint32_t length;
  // Payload value attached to the zone. Not NUL terminated.
  char value[/*length*/];
} iree_hal_amdgpu_trace_zone_value_text_dynamic_t;

// Begins a device execution zone.
// This captures the timestamp the zone is issued as well as a query_id used to
// correlate a future update of the timing when available.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_trace_execution_zone_begin_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN
  iree_hal_amdgpu_trace_event_type_t event_type;
  // Execution trace ID used to distinguish different execution units. This is
  // assigned on the host when the execution context is configured.
  iree_hal_amdgpu_trace_executor_id_t executor_id;
  uint8_t reserved0[2];  // may be uninitialized
  // A query ID used to feed back the timestamp once the execution has
  // completed.
  iree_hal_amdgpu_trace_execution_query_id_t execution_query_id;
  uint8_t reserved1[2];  // may be uninitialized
  // Timestamp the zone begin was issued at.
  // Note that this need not be in order with any other timestamps.
  iree_hal_amdgpu_trace_agent_timestamp_t issue_timestamp;
  // Source location of the zone being entered.
  iree_hal_amdgpu_trace_src_loc_ptr_t src_loc;
} iree_hal_amdgpu_trace_execution_zone_begin_t;

// Ends a device execution zone.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_trace_execution_zone_end_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END
  iree_hal_amdgpu_trace_event_type_t event_type;
  // Execution trace ID used to distinguish different execution units. This is
  // assigned on the host when the execution context is configured.
  iree_hal_amdgpu_trace_executor_id_t executor_id;
  uint8_t reserved0[2];  // may be uninitialized
  // A query ID used to feed back the timestamp once the execution has
  // completed.
  iree_hal_amdgpu_trace_execution_query_id_t execution_query_id;
  uint8_t reserved1[2];  // may be uninitialized
  // Timestamp the zone end was issued at.
  // Note that this need not be in order with any other timestamps.
  iree_hal_amdgpu_trace_agent_timestamp_t issue_timestamp;
} iree_hal_amdgpu_trace_execution_zone_end_t;

// Defines the type of an execution zone dispatch.
typedef uint8_t iree_hal_amdgpu_trace_execution_zone_type_t;
enum iree_hal_amdgpu_trace_execution_zone_type_e {
  // Indicates an executable export dispatch (kernel launch).
  // The export_loc will be populated with the value defined by the host.
  IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_DISPATCH = 0u,
  // Indicates an indirect executable export dispatch.
  // The export_loc will be populated with the value defined by the host.
  // The total time will span both the indirect preparation dispatch (if
  // required) and the dispatch itself.
  IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_DISPATCH_INDIRECT = 0u,
  // Indicates a DMA copy operation. export_loc may be 0.
  IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_COPY,
  // Indicates a DMA fill operation. export_loc may be 0.
  IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_FILL,
  // Indicates an internal bookkeeping dispatch. export_loc may be 0.
  IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_INTERNAL,
};

// Represents a leaf device execution zone.
// This is the same as emitting an execution zone begin and end pair but has
// less overhead for the common cases of leaf zones (dispatches, etc).
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_trace_execution_zone_dispatch_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH
  iree_hal_amdgpu_trace_event_type_t event_type;
  // Defines what kind of dispatch operation this is and how the export_loc is
  // interpreted (if it is at all).
  iree_hal_amdgpu_trace_execution_zone_type_t zone_type;
  // Execution trace ID used to distinguish different execution units. This is
  // assigned on the host when the execution context is configured.
  iree_hal_amdgpu_trace_executor_id_t executor_id;
  uint8_t reserved0[1];  // may be uninitialized
  // A query ID used to feed back the timestamp once the execution has
  // completed. In indirect dispatches with multiple device dispatches this is
  // used only for the primary dispatch.
  // Note that we have space for another ID but allocating those is annoying.
  iree_hal_amdgpu_trace_execution_query_id_t execution_query_id;
  uint8_t reserved1[2];
  // A reference to the interned export source location in host memory.
  // The host queries this information and preserves it with process lifetime
  // so that we can quickly look it up when feeding it to the trace sink.
  // In other approaches we'd have to allocate the source location each time we
  // recorded an event to it or deal with tracking such information on the
  // device.
  uint64_t export_loc;
  // Timestamp the dispatch was issued at.
  // Note that this need not be in order with any other timestamps.
  // To save on space we only record the instantaneous timestamp of the issue
  // and apply a fixed duration. We do the issues in parallel and the timings
  // would be messy anyway. A base timestamp is enough to calculate latency from
  // issue to execution.
  //
  // NOTE: this relies on the current tracy behavior of using the information
  // only to hint at where in the global timeline the issue occurred. If it was
  // actually trying to assign issues to zones then we'd likely have a problem
  // and need to either serialize all issues when tracing or do some funny math.
  iree_hal_amdgpu_trace_agent_timestamp_t issue_timestamp;
} iree_hal_amdgpu_trace_execution_zone_dispatch_t;

// Notifies the trace sink of a batch of completed queries.
// All queries must have contiguous IDs starting at the specified base ID.
typedef struct IREE_AMDGPU_ALIGNAS(8)
    iree_hal_amdgpu_trace_execution_zone_notify_batch_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY_BATCH
  iree_hal_amdgpu_trace_event_type_t event_type;
  // Execution trace ID used to distinguish different execution units. This is
  // assigned on the host when the execution context is configured.
  iree_hal_amdgpu_trace_executor_id_t executor_id;
  uint8_t reserved0[2];  // may be uninitialized
  // The base query ID for all queries in the batch.
  iree_hal_amdgpu_trace_execution_query_id_t execution_query_id_base;
  // The total number of queries in the batch.
  uint16_t execution_query_count;
  // Timestamp ranges of the queries as executed.
  iree_hal_amdgpu_trace_agent_time_range_t
      execution_time_ranges[/*execution_query_count*/];
} iree_hal_amdgpu_trace_execution_zone_notify_batch_t;

// Records the allocation of a block of memory from a named pool.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_memory_alloc_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC
  iree_hal_amdgpu_trace_event_type_t event_type;
  // TODO(benvanik): try to see if we can get the memory pool name in 7 bytes -
  // if so we can shrink the packet to 24 bytes.
  uint8_t reserved[7];
  // Pool name used as both a title for the pool and the unique ID for
  // correlating alloc/free events.
  iree_hal_amdgpu_trace_string_literal_ptr_t pool;
  // Timestamp the allocation was made.
  iree_hal_amdgpu_trace_agent_timestamp_t timestamp;
  // Pointer in whatever memory space the pool defines.
  uint64_t ptr;
  // Size of the allocation in bytes.
  uint64_t size;
} iree_hal_amdgpu_trace_memory_alloc_t;

// Records the freeing of a block of memory from a named pool.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_memory_free_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE
  iree_hal_amdgpu_trace_event_type_t event_type;
  uint8_t reserved[7];
  // Pool name used as both a title for the pool and the unique ID for
  // correlating alloc/free events.
  iree_hal_amdgpu_trace_string_literal_ptr_t pool;
  // Timestamp the allocation was freed.
  iree_hal_amdgpu_trace_agent_timestamp_t timestamp;
  // Pointer in whatever memory space the pool defines. Must have previously
  // been used in a memory allocation event.
  uint64_t ptr;
} iree_hal_amdgpu_trace_memory_free_t;

// Logs a string message.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_message_literal_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL
  iree_hal_amdgpu_trace_event_type_t event_type;
  // TODO(benvanik): try to see if we can get the literal in 7 bytes - if so
  // we can shrink the packet to 16 bytes.
  uint8_t reserved[7];
  // Timestamp the message was emitted.
  iree_hal_amdgpu_trace_agent_timestamp_t timestamp;
  // Message payload. Not NUL terminated.
  iree_hal_amdgpu_trace_string_literal_ptr_t value;
} iree_hal_amdgpu_trace_message_literal_t;

// Logs a string message.
// The contents are embedded in the trace buffer to support dynamically
// generated values.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_message_dynamic_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC
  iree_hal_amdgpu_trace_event_type_t event_type;
  uint8_t reserved[3];
  // Length of the value in characters.
  uint32_t length;
  // Timestamp the message was emitted.
  iree_hal_amdgpu_trace_agent_timestamp_t timestamp;
  // Message payload. Not NUL terminated.
  char value[/*length*/];
} iree_hal_amdgpu_trace_message_dynamic_t;

// Defines a plot.
// This must be called prior to
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_plot_config_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG
  iree_hal_amdgpu_trace_event_type_t event_type;
  // Defines the plot type.
  iree_hal_amdgpu_trace_plot_type_t plot_type;
  // Controls plot display and accumulation behavior.
  iree_hal_amdgpu_trace_plot_flags_t plot_flags;
  // Base color of the plot (line and fill will be derived from this).
  iree_hal_amdgpu_trace_color_t color;
  // Plot name displayed as a title.
  // The pointer value is used as a key for future plot data.
  iree_hal_amdgpu_trace_string_literal_ptr_t name;
} iree_hal_amdgpu_trace_plot_config_t;

// Records an i64 plot value change.
typedef struct IREE_AMDGPU_ALIGNAS(8) iree_hal_amdgpu_trace_plot_value_i64_t {
  // IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64
  iree_hal_amdgpu_trace_event_type_t event_type;
  // TODO(benvanik): try to see if we can get the plot name in 7 bytes - if so
  // we can shrink the packet to 24 bytes.
  uint8_t reserved[7];
  // Uniqued name of the plot as used during configuration.
  iree_hal_amdgpu_trace_string_literal_ptr_t plot_name;
  // Time the plot value was emitted.
  iree_hal_amdgpu_trace_agent_timestamp_t timestamp;
  // Plot value as interpreted by the plot type.
  int64_t value;
} iree_hal_amdgpu_trace_plot_value_i64_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_query_ringbuffer_t
//===----------------------------------------------------------------------===//

// Total number of query signals allocated to a trace buffer ringbuffer.
// This preallocates the signals as part of the trace buffer structure.
//
// We generally trade off some fixed device memory consumption by allocating a
// large pool instead of trying to handle cases of exhaustion. This could be
// lowered but at the point you have a few hundred GB of GPU memory 4MB is a
// drop in a very large bucket.
//
// Due to tracy behavior we have to reserve query indices for begin/end of
// dispatches even though we only need one signal. We use the upper bit to
// differentiate when reporting the signals to tracy.
//
// Must be a power-of-two.
#define IREE_HAL_AMDGPU_DEVICE_QUERY_RINGBUFFER_CAPACITY (0xFFFFu >> 1)

// A ringbuffer of device-only query signals that can be acquired/released in
// large blocks. Query signals are not full HSA signals and cannot be used on
// the host - there's no backing mailbox/doorbell for raising interrupts and
// attempting to cast them to hsa_signal_t (on the host) will fail.
//
// This exploits the fact that signals are just iree_amd_signal_t structs in
// memory from the perspective of the device - only the host cares if they are
// wrapped in ROCR/HSA types.
//
// Signals are maintained at rest with a value of 1. Those acquiring can change
// this value after acquiring them if needed.
//
// Thread-compatible; only the thread owning the trace buffer will acquire and
// release from the ringbuffer so there's no need to make it safer.
typedef struct iree_hal_amdgpu_device_query_ringbuffer_t {
  uint64_t read_index;
  uint64_t write_index;
  iree_amd_signal_t signals[IREE_HAL_AMDGPU_DEVICE_QUERY_RINGBUFFER_CAPACITY];
} iree_hal_amdgpu_device_query_ringbuffer_t;

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Acquires a signal from the ringbuffer and returns the execution query_id for
// it. Callers must use iree_hal_amdgpu_device_query_ringbuffer_signal_for_id to
// get the signal handle that can be provided in packets.
iree_hal_amdgpu_trace_execution_query_id_t
iree_hal_amdgpu_device_query_ringbuffer_acquire(
    iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT ringbuffer);

// Acquires |count| signals from the ringbuffer and returns the base index in
// the absolute ringbuffer domain. Callers must use
// iree_hal_amdgpu_device_query_ringbuffer_signal to get the signal handle that
// can be provided in packets.
uint64_t iree_hal_amdgpu_device_query_ringbuffer_acquire_range(
    iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint16_t count);

// Releases the oldest acquired batch of |count| signals back to the ringbuffer.
// The signals may immediately be overwritten/reused and must have no
// outstanding references by either the caller or the hardware queues.
void iree_hal_amdgpu_device_query_ringbuffer_release_range(
    iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint16_t count);

// Returns the tracing ID used for the signal at the given absolute ringbuffer
// index.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE
    iree_hal_amdgpu_trace_execution_query_id_t
    iree_hal_amdgpu_device_query_ringbuffer_query_id(
        const iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT
            ringbuffer,
        uint64_t index) {
  return (
      iree_hal_amdgpu_trace_execution_query_id_t)(index &
                                                  (IREE_AMDGPU_ARRAYSIZE(
                                                       ringbuffer->signals) -
                                                   1));
}

// Returns a device-only HSA signal handle for the query signal at the given
// absolute ringbuffer index.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE iree_hsa_signal_t
iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
    const iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT
        ringbuffer,
    iree_hal_amdgpu_trace_execution_query_id_t query_id) {
  return (iree_hsa_signal_t){(uint64_t)&ringbuffer->signals[query_id]};
}

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_trace_buffer_t
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES

// Single-producer/single-consumer ringbuffer with mapping tricks.
// Trace events are emitted by the scheduler in batches by having the scheduler
// mark the start of a reservation, populating that with as many events as it
// wants, and then committing at the end of its written range. The host side is
// responsible for processing the range defined from the last read commit offset
// to the last write commit offset it receives when running.
//
// Writes must check for overflow by ensuring that there is sufficient capacity
// for their reservation. An example write sequence:
// if (write_reserve_offset + requested_size - read_commit_offset >= capacity) {
//    iree_amdgpu_yield();
// }
// memcpy(ringbuffer_base + write_reserve_offset, contents, requested_size);
// write_reserve_offset += requested_size;
//
// This presents as a ringbuffer that does not need any special logic for
// wrapping from base offsets used when copying in memory. It follows the
// approach documented in of virtual memory mapping the buffer multiple times:
// https://github.com/google/wuffs/blob/main/script/mmap-ring-buffer.c
// We use SVM to allocate the physical memory of the ringbuffer and then stitch
// together 3 virtual memory ranges in one contiguous virtual allocation that
// alias the physical allocation. By treating the middle range as the base
// buffer pointer we are then able to freely dereference both before and after
// the base pointer by up to the ringbuffer size in length.
//   physical: <ringbuffer size> --+------+------+
//                                 v      v      v
//                        virtual: [prev] [base] [next]
//                                        ^
//                                        +-- base_ptr
//
// Because of the mapping trick we have a maximum outstanding ringbuffer size
// equal to the ringbuffer capacity (modulo alignment requirements). We flush
// after each major phase of work (when the scheduler goes idle, a command
// buffer block completes execution, etc) and need a minimum capacity enough to
// store all of the data produced during those phases. Command buffers are the
// riskiest and with a Tracy-imposed uint16_t signal query ringbuffer we have to
// chunk those anyway and can ensure we have enough space for the
// iree_hal_amdgpu_trace_execution_zone_dispatch_t and corresponding
// iree_hal_amdgpu_trace_execution_zone_notify_batch_t packets (+ margin for the
// scheduler).
//
// Thread-compatible; single-producer/single-consumer. The scheduler that owns
// the trace buffer is the "thread" and is the only one allowed to write to it.
// The paired host command processor is the only one allowed to read from it
// when marshaling to the native host tracing APIs. This allows us to use
// relatively simple data structures and commit at fixed intervals: we reserve
// from the write index while producing data and only commit to make it host
// visible at reasonable flush points.
//
// Each trace buffer represents a single thread of execution (the scheduler)
// plus one or more additional device executors (the hardware queues executing
// commands) so we don't need to track thread IDs or other TLS information on
// events. The host processing the events will assign the appropriate tracing
// IDs when transcribing the events.
//
// Pointers embedded within the trace events are treated as either opaque (such
// as the pointer used during an allocation event) or as special read-only data
// segment pointers. See iree_hal_amdgpu_trace_rodata_ptr_t for more
// information. The host must translate these pointers before dereferencing them
// or providing them to the trace sink that will (for example to get source
// location name strings).
typedef struct iree_hal_amdgpu_device_trace_buffer_t {
  // Base ringbuffer pointer used for all relative addressing.
  // Pointers must always be within the range of
  // (ringbuffer_base-ringbuffer_capacity, ringbuffer_base+ringbuffer_capacity).
  uint8_t* IREE_AMDGPU_RESTRICT ringbuffer_base;
  // Total size in bytes of the trace data ringbuffer.
  // Note that this is the size of the underlying physical allocation and the
  // virtual range is 3x that. Must be a power of two.
  uint64_t ringbuffer_capacity;
  // Process-unique executor ID used for tracing command execution.
  // NOTE: this assumes only one executor per trace buffer; we may want to
  // support multiple and make users pass them in for cases where a single
  // command buffer may execute on multiple hardware execution queues.
  iree_hal_amdgpu_trace_executor_id_t executor_id;
  uint8_t reserved0[7];  // may be uninitialized
  // iree_hal_amdgpu_trace_buffer_t* used to route flush requests back to the
  // owning host resource.
  uint64_t host_trace_buffer;
  // Used by the host to indicate where it has completed reading to. The host
  // should atomically bump read_commit_offset when it has completed reading a
  // chunk from the ringbuffer. If the capacity is reached then the device may
  // spin until the host has caught up.
  // Absolute offset - must be masked with ringbuffer_mask.
  IREE_AMDGPU_ALIGNAS(iree_amdgpu_destructive_interference_size)
  iree_amdgpu_scoped_atomic_uint64_t read_commit_offset;
  // Exclusively used by the scheduler to mark the start of its current
  // reservation. This is assigned after each flush and only advanced with each
  // trace event recorded.
  // Absolute offset - must be masked with ringbuffer_mask.
  IREE_AMDGPU_ALIGNAS(iree_amdgpu_destructive_interference_size)
  iree_amdgpu_scoped_atomic_uint64_t write_reserve_offset;
  // Used by both the host and scheduler to track the current committed write
  // range. Always <= the write_reserve_offset (with == indicating that there
  // are no pending events).
  // Absolute offset - must be masked with ringbuffer_mask.
  IREE_AMDGPU_ALIGNAS(iree_amdgpu_destructive_interference_size)
  iree_amdgpu_scoped_atomic_uint64_t write_commit_offset;
  // Ringbuffer of device-only signals that can be used to get timestamps from
  // the packet processor. Only the scheduler that owns the trace buffer is
  // allowed to acquire/release from the ringbuffer and it is sized to fit only
  // a single command buffer block worth of operations.
  iree_hal_amdgpu_device_query_ringbuffer_t query_ringbuffer;
} iree_hal_amdgpu_device_trace_buffer_t;

// Mask used to wrap an absolute ringbuffer offset into a base pointer offset.
#define iree_hal_amdgpu_device_trace_buffer_mask(trace_buffer) \
  ((trace_buffer)->ringbuffer_capacity - 1)

#else

typedef struct iree_hal_amdgpu_device_trace_buffer_t {
  int reserved;
} iree_hal_amdgpu_device_trace_buffer_t;

#define IREE_HAL_AMDGPU_TRACE_BUFFER_KERNARG_SIZE 0

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURES

// Control kernargs used when launching the trace buffer kernels.
typedef struct iree_hal_amdgpu_device_trace_buffer_kernargs_t {
  IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_t* trace_buffer;
} iree_hal_amdgpu_device_trace_buffer_kernargs_t;

//===----------------------------------------------------------------------===//
// Tracing Macros/Support
//===----------------------------------------------------------------------===//

typedef uint32_t iree_hal_amdgpu_zone_id_t;

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Colors used for messages based on the level provided to the macro.
enum {
  IREE_AMDGPU_TRACE_MESSAGE_LEVEL_ERROR = 0xFF0000u,
  IREE_AMDGPU_TRACE_MESSAGE_LEVEL_WARNING = 0xFFFF00u,
  IREE_AMDGPU_TRACE_MESSAGE_LEVEL_INFO = 0xFFFFFFu,
  IREE_AMDGPU_TRACE_MESSAGE_LEVEL_VERBOSE = 0xC0C0C0u,
  IREE_AMDGPU_TRACE_MESSAGE_LEVEL_DEBUG = 0x00FF00u,
};

#if IREE_HAL_AMDGPU_TRACING_FEATURES

#define IREE_AMDGPU_TRACE_BUFFER_SCOPE(trace_buffer)                  \
  iree_hal_amdgpu_device_trace_buffer_t* __iree_amdgpu_trace_buffer = \
      (trace_buffer)
#define IREE_AMDGPU_TRACE_BUFFER() (__iree_amdgpu_trace_buffer)

#else

#define IREE_AMDGPU_TRACE_BUFFER_SCOPE(...)
#define IREE_AMDGPU_TRACE_BUFFER() NULL

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURES

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION)

#define IREE_AMDGPU_TRACE_CONCAT_(x, y) IREE_AMDGPU_TRACE_CONCAT_INDIRECT_(x, y)
#define IREE_AMDGPU_TRACE_CONCAT_INDIRECT_(x, y) x##y

// Begins a new zone with the parent function name.
#define IREE_AMDGPU_TRACE_ZONE_BEGIN(zone_id) \
  IREE_AMDGPU_TRACE_ZONE_BEGIN_NAMED(zone_id, NULL)

// Begins a new zone with the given compile-time |name_literal|.
// The literal must be static const and will be embedded in the trace buffer by
// reference.
#define IREE_AMDGPU_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) \
  IREE_AMDGPU_TRACE_ZONE_BEGIN_NAMED_COLORED(zone_id, name_literal, 0)

// Begins a new zone with the given compile-time |name_literal| and color.
// The literal must be static const and will be embedded in the trace buffer by
// reference.
#define IREE_AMDGPU_TRACE_ZONE_BEGIN_NAMED_COLORED(zone_id, name_literal,      \
                                                   color)                      \
  static const iree_hal_amdgpu_trace_src_loc_t IREE_AMDGPU_TRACE_CONCAT_(      \
      __iree_amdgpu_trace_src_loc, __LINE__) = {                               \
      (name_literal), __FUNCTION__, __FILE__, (uint32_t)__LINE__, (color),     \
  };                                                                           \
  iree_hal_amdgpu_zone_id_t zone_id = iree_hal_amdgpu_device_trace_zone_begin( \
      IREE_AMDGPU_TRACE_BUFFER(),                                              \
      (iree_hal_amdgpu_trace_src_loc_ptr_t) &                                  \
          IREE_AMDGPU_TRACE_CONCAT_(__iree_amdgpu_trace_src_loc, __LINE__));

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_AMDGPU_TRACE_ZONE_END(zone_id) \
  iree_hal_amdgpu_device_trace_zone_end(IREE_AMDGPU_TRACE_BUFFER())

// Appends an int64_t value to the parent zone. May be called multiple times.
#define IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(zone_id, value) \
  (void)(zone_id);                                              \
  iree_hal_amdgpu_device_trace_zone_append_value_i64(           \
      IREE_AMDGPU_TRACE_BUFFER(), (int64_t)(value))

// Appends a string literal value to the parent zone. May be called multiple
// times. The provided NUL-terminated C string will be referenced directly.
#define IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(zone_id, value_literal) \
  (void)(zone_id);                                                         \
  iree_hal_amdgpu_device_trace_zone_append_text_literal(                   \
      IREE_AMDGPU_TRACE_BUFFER(),                                          \
      (iree_hal_amdgpu_trace_string_literal_ptr_t)(value_literal))

// Appends a string value to the parent zone. May be called multiple times.
// The provided NUL-terminated C string or string view will be copied into the
// trace buffer.
#define IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_DYNAMIC(...)                     \
  IREE_AMDGPU_TRACE_IMPL_GET_VARIADIC_(                                     \
      (__VA_ARGS__, IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_STRING_VIEW_DYNAMIC, \
       IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_CSTRING_DYNAMIC))                 \
  (__VA_ARGS__)
#define IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_CSTRING_DYNAMIC(zone_id, value)   \
  IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_STRING_VIEW_DYNAMIC((zone_id), (value), \
                                                         sizeof(value))
#define IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_STRING_VIEW_DYNAMIC(zone_id, value, \
                                                               value_length)   \
  (void)(zone_id);                                                             \
  iree_hal_amdgpu_device_trace_zone_append_text_dynamic(                       \
      IREE_AMDGPU_TRACE_BUFFER(), (value), (value_length))

// Configures the named plot with iree_hal_amdgpu_trace_plot_type_t data and
// iree_hal_amdgpu_trace_plot_flags_t controlling the display.
#define IREE_AMDGPU_TRACE_PLOT_CONFIGURE(name_literal, type, flags, color) \
  iree_hal_amdgpu_device_trace_plot_configure(                             \
      IREE_AMDGPU_TRACE_BUFFER(),                                          \
      (iree_hal_amdgpu_trace_string_literal_ptr_t)(name_literal), (type),  \
      (flags), (color))
// Plots a value in the named plot group as an int64_t.
#define IREE_AMDGPU_TRACE_PLOT_VALUE_I64(name_literal, value) \
  iree_hal_amdgpu_device_trace_plot_value_i64(                \
      IREE_AMDGPU_TRACE_BUFFER(),                             \
      (iree_hal_amdgpu_trace_string_literal_ptr_t)(name_literal), (value))

// Utilities:
#define IREE_AMDGPU_TRACE_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME
#define IREE_AMDGPU_TRACE_IMPL_GET_VARIADIC_(args) \
  IREE_AMDGPU_TRACE_IMPL_GET_VARIADIC_HELPER_ args

#else

#define IREE_AMDGPU_TRACE_ZONE_BEGIN(zone_id) \
  iree_hal_amdgpu_zone_id_t zone_id = 0;      \
  (void)zone_id;
#define IREE_AMDGPU_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) \
  IREE_AMDGPU_TRACE_ZONE_BEGIN(zone_id)
#define IREE_AMDGPU_TRACE_ZONE_BEGIN_NAMED_COLORED(zone_id, name_literal, \
                                                   color)                 \
  IREE_AMDGPU_TRACE_ZONE_BEGIN(zone_id)
#define IREE_AMDGPU_TRACE_ZONE_END(zone_id) (void)(zone_id)

#define IREE_AMDGPU_TRACE_ZONE_APPEND_VALUE_I64(zone_id, value)
#define IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_LITERAL(zone_id, value_literal)
#define IREE_AMDGPU_TRACE_ZONE_APPEND_TEXT_DYNAMIC(zone_id, ...)

#define IREE_AMDGPU_TRACE_PLOT_CONFIGURE(name_literal, type, flags, color)
#define IREE_AMDGPU_TRACE_PLOT_VALUE_I64(name_literal, value)

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING)

// Traces a new memory allocation in a named memory pool.
// Reallocations must be recorded as an
// IREE_AMDGPU_TRACE_ALLOC_NAMED/IREE_AMDGPU_TRACE_FREE_NAMED pair.
#define IREE_AMDGPU_TRACE_ALLOC_NAMED(name_literal, ptr, size)    \
  iree_hal_amdgpu_device_trace_memory_alloc(                      \
      IREE_AMDGPU_TRACE_BUFFER(),                                 \
      (iree_hal_amdgpu_trace_string_literal_ptr_t)(name_literal), \
      (uint64_t)(ptr), (size))

// Traces a free of an existing allocation traced with
// IREE_AMDGPU_TRACE_ALLOC_NAMED.
#define IREE_AMDGPU_TRACE_FREE_NAMED(name_literal, ptr)           \
  iree_hal_amdgpu_device_trace_memory_free(                       \
      IREE_AMDGPU_TRACE_BUFFER(),                                 \
      (iree_hal_amdgpu_trace_string_literal_ptr_t)(name_literal), \
      (uint64_t)(ptr))

#else

#define IREE_AMDGPU_TRACE_ALLOC_NAMED(name_literal, ptr, size)
#define IREE_AMDGPU_TRACE_FREE_NAMED(name_literal, ptr)

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES)

// Logs a message at the given logging level to the trace.
// The message text must be a compile-time string literal.
#define IREE_AMDGPU_TRACE_MESSAGE_LITERAL(level, value_literal) \
  IREE_AMDGPU_TRACE_MESSAGE_LITERAL_COLORED(                    \
      IREE_AMDGPU_TRACE_MESSAGE_LEVEL_##level, (value_literal))

// Logs a message with the given color to the trace.
// Standard colors are defined as IREE_AMDGPU_TRACE_MESSAGE_LEVEL_* values.
// The message text must be a compile-time string literal.
#define IREE_AMDGPU_TRACE_MESSAGE_LITERAL_COLORED(color, value_literal) \
  iree_hal_amdgpu_device_trace_message_literal(                         \
      IREE_AMDGPU_TRACE_BUFFER(), (color),                              \
      (iree_hal_amdgpu_trace_string_literal_ptr_t)(value_literal))

// Logs a dynamically-allocated message at the given logging level to the trace.
// The string |value| will be copied into the trace buffer.
#define IREE_AMDGPU_TRACE_MESSAGE_DYNAMIC(level, value, value_length) \
  IREE_AMDGPU_TRACE_MESSAGE_DYNAMIC_COLORED(                          \
      IREE_AMDGPU_TRACE_MESSAGE_LEVEL_##level, (value), (value_length))

// Logs a dynamically-allocated message with the given color to the trace.
// Standard colors are defined as IREE_AMDGPU_TRACE_MESSAGE_LEVEL_* values.
// The string |value| will be copied into the trace buffer.
#define IREE_AMDGPU_TRACE_MESSAGE_DYNAMIC_COLORED(color, value, value_length) \
  iree_hal_amdgpu_device_trace_message_dynamic(                               \
      IREE_AMDGPU_TRACE_BUFFER(), (color), (value), (value_length))

#else

#define IREE_AMDGPU_TRACE_MESSAGE_LITERAL(level, value_literal)
#define IREE_AMDGPU_TRACE_MESSAGE_LITERAL_COLORED(color, value_literal)
#define IREE_AMDGPU_TRACE_MESSAGE_DYNAMIC(level, value, value_length)
#define IREE_AMDGPU_TRACE_MESSAGE_DYNAMIC_COLORED(color, value, value_length)

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEBUG_MESSAGES)

// Logs a message formatted with an extremely basic sprintf-like function.
// Supported format specifiers:
//   %% (escape for `%`)
//   %c (single `char`)
//   %s (NUL-terminated string)
//   %u/%lu (uint32_t/uint64_t in base 10)
//   %x/%lx (uint32_t/uint64_t in base 16)
//   %p (pointer)
#define IREE_AMDGPU_DBG(format, ...)                                       \
  iree_hal_amdgpu_device_trace_debug(IREE_AMDGPU_TRACE_BUFFER(), (format), \
                                     __VA_ARGS__)

#else

#define IREE_AMDGPU_DBG(format, ...)

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEBUG_MESSAGES

#endif  // IREE_AMDGPU_TARGET_DEVICE

//===----------------------------------------------------------------------===//
// Device-side API
//===----------------------------------------------------------------------===//

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Commits the current write reservation to the ringbuffer so that the host can
// begin reading it. Callers must notify the host that new data is available via
// a host interrupt if this returns true.
bool iree_hal_amdgpu_device_trace_commit_range(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer);

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION)

iree_hal_amdgpu_zone_id_t iree_hal_amdgpu_device_trace_zone_begin(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_src_loc_ptr_t src_loc);
void iree_hal_amdgpu_device_trace_zone_end(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer);

void iree_hal_amdgpu_device_trace_zone_append_value_i64(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    int64_t value);
void iree_hal_amdgpu_device_trace_zone_append_text_literal(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t value_literal);
void iree_hal_amdgpu_device_trace_zone_append_text_dynamic(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    const char* IREE_AMDGPU_RESTRICT value, size_t value_length);

void iree_hal_amdgpu_device_trace_plot_configure(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal,
    iree_hal_amdgpu_trace_plot_type_t type,
    iree_hal_amdgpu_trace_plot_flags_t flags,
    iree_hal_amdgpu_trace_color_t color);
void iree_hal_amdgpu_device_trace_plot_value_i64(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, int64_t value);

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_begin(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id,
    iree_hal_amdgpu_trace_src_loc_ptr_t src_loc);
iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_end(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id);
void iree_hal_amdgpu_device_trace_execution_zone_notify(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id,
    iree_hal_amdgpu_trace_agent_time_range_t time_range);
iree_hal_amdgpu_trace_agent_time_range_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_trace_execution_zone_notify_batch(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id_base,
    uint16_t execution_query_count);

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_dispatch(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_zone_type_t zone_type, uint64_t export_loc,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id);

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING)

void iree_hal_amdgpu_device_trace_memory_alloc(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, uint64_t ptr,
    uint64_t size);
void iree_hal_amdgpu_device_trace_memory_free(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, uint64_t ptr);

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES)

void iree_hal_amdgpu_device_trace_message_literal(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_color_t color,
    iree_hal_amdgpu_trace_string_literal_ptr_t value_literal);
void iree_hal_amdgpu_device_trace_message_dynamic(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_color_t color, const char* IREE_AMDGPU_RESTRICT value,
    size_t value_length);

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEBUG_MESSAGES)

void iree_hal_amdgpu_device_trace_debug(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    const char* IREE_AMDGPU_RESTRICT format, ...);

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEBUG_MESSAGES

#endif  // IREE_AMDGPU_TARGET_DEVICE

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_TRACING_H_
