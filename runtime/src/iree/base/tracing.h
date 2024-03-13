// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Utilities for runtime tracing support.
// These allow the various runtime subsystems to insert trace events, attach
// metadata to events or allocations, and control tracing verbosity.
//
// Tracing features can be enabled with either an IREE_TRACING_MODE define that
// allows predefined tracing modes or individual IREE_TRACING_FEATURE_* flags
// set on IREE_TRACING_FEATURES when a more custom set of features is
// required. Exact feature support may vary on platform and toolchain.
//
// The tracing infrastructure is currently designed to target the Tracy
// profiler: https://github.com/wolfpld/tracy
// Tracy's profiler UI allowing for streaming captures and analysis can be
// downloaded from: https://github.com/wolfpld/tracy/releases
// The manual provided on the releases page contains more information about how
// Tracy works, its limitations, and how to operate the UI.
//
// NOTE: this header is used both from C and C++ code and only conditionally
// enables the C++ when in a valid context. Do not use C++ features or include
// other files that are not C-compatible.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"

#ifndef IREE_BASE_TRACING_H_
#define IREE_BASE_TRACING_H_

//===----------------------------------------------------------------------===//
// IREE_TRACING_FEATURE_* flags and options
//===----------------------------------------------------------------------===//

// Enables IREE_TRACE_* macros for instrumented tracing.
#define IREE_TRACING_FEATURE_INSTRUMENTATION (1 << 0)

// Captures callstacks up to IREE_TRACING_MAX_CALLSTACK_DEPTH at all
// IREE_TRACE_* events. This has a significant performance impact and should
// only be enabled when tracking down missing instrumentation.
#define IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS (1 << 1)

// Enables instrumentation of external device APIs (GPUs, etc) when supported.
// This can have significant code size and runtime overhead and should only be
// used when specifically tracing device-side execution.
#define IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE (1 << 2)

// Tracks all allocations (we know about) via new/delete/malloc/free.
// This allows fine-grained allocation and usage tracking down to the code that
// performed the allocations. Allocations or frees that are performed outside of
// the IREE API or runtime library will not be tracked and unbalanced usage
// (allocating with IREE's API then freeing with stdlib free, for example) will
// cause Tracy to become very unhappy.
#define IREE_TRACING_FEATURE_ALLOCATION_TRACKING (1 << 3)

// Captures callstacks up to IREE_TRACING_MAX_CALLSTACK_DEPTH at all allocation
// events when allocation tracking is enabled.
#define IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS (1 << 4)

// Tracks fast locks in all cases (both contended and uncontended).
// This may introduce contention where there would otherwise be none as what
// would be a handful of instructions and little memory access may become
// hundreds. To see only locks under contention use
// IREE_TRACING_FEATURE_SLOW_LOCKS.
#define IREE_TRACING_FEATURE_FAST_LOCKS (1 << 5)

// Tracks slow locks that end up going to the OS for waits/wakes in futexes.
// Uncontended locks will not be displayed and only waits will be visible in the
// Tracy UI.
#define IREE_TRACING_FEATURE_SLOW_LOCKS (1 << 6)

// Forwards log messages to traces, which will be visible under "Messages" in
// the Tracy UI.
#define IREE_TRACING_FEATURE_LOG_MESSAGES (1 << 7)

// Enables fiber support in the Tracy UI.
// Comes with a per-event overhead (less efficient queue insertion) but is
// required when running with asynchronous VM invocations.
#define IREE_TRACING_FEATURE_FIBERS (1 << 8)

#if !defined(IREE_TRACING_MAX_CALLSTACK_DEPTH)
// Tracing functions that capture stack traces will only capture up to N frames.
// The overhead for stack walking scales linearly with the number of frames
// captured and can increase the cost of an event capture by orders of
// magnitude.
// Minimum: 0 (disable)
// Maximum: 62
#define IREE_TRACING_MAX_CALLSTACK_DEPTH 16
#endif  // IREE_TRACING_MAX_CALLSTACK_DEPTH

//===----------------------------------------------------------------------===//
// IREE_TRACING_MODE simple setting
//===----------------------------------------------------------------------===//
// This defines IREE_TRACING_FEATURES_REQUESTED (which may also be provided in
// the user config.h) to select which tracing features are requested. Providers
// may not implement all of the features and are expected to filter out the
// requested features into ones they do before defining the primary
// IREE_TRACING_FEATURES value.

#if !defined(IREE_TRACING_FEATURES_REQUESTED)
#if defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 1
#define IREE_TRACING_FEATURES_REQUESTED \
  (IREE_TRACING_FEATURE_INSTRUMENTATION | IREE_TRACING_FEATURE_LOG_MESSAGES)
#undef IREE_TRACING_MAX_CALLSTACK_DEPTH
#define IREE_TRACING_MAX_CALLSTACK_DEPTH 0
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 2
#define IREE_TRACING_FEATURES_REQUESTED          \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |        \
   IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE | \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING |    \
   IREE_TRACING_FEATURE_LOG_MESSAGES)
// TODO(#9627): make tracy fibers faster; too slow for on-by-default!
// | IREE_TRACING_FEATURE_FIBERS)
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 3
#define IREE_TRACING_FEATURES_REQUESTED          \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |        \
   IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE | \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING |    \
   IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS |  \
   IREE_TRACING_FEATURE_LOG_MESSAGES | IREE_TRACING_FEATURE_FIBERS)
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE >= 4
#define IREE_TRACING_FEATURES_REQUESTED              \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |            \
   IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS | \
   IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE |     \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING |        \
   IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS |      \
   IREE_TRACING_FEATURE_LOG_MESSAGES | IREE_TRACING_FEATURE_FIBERS)
#else
#define IREE_TRACING_FEATURES_REQUESTED 0
#endif  // IREE_TRACING_MODE
#endif  // !IREE_TRACING_FEATURES

//===----------------------------------------------------------------------===//
// Tracing implementation
//===----------------------------------------------------------------------===//

// Include the actual tracing implementation used when a tracing mode is set.
// This will either come from build options like IREE_TRACING_PROVIDER or can
// be overridden during compilation with
// -DIREE_TRACING_PROVIDER_H="my_provider.h".
#if defined(IREE_TRACING_PROVIDER_H)

#include IREE_TRACING_PROVIDER_H

#if !defined(IREE_TRACING_FEATURES)
#error \
    "Tracing provider must define IREE_TRACING_FEATURES based on the requested bits in IREE_TRACING_FEATURES_REQUESTED and what it supports"
#endif  // !IREE_TRACING_FEATURES

#else
#define IREE_TRACING_FEATURES 0
typedef uint32_t iree_zone_id_t;
#endif  // IREE_TRACING_PROVIDER_H

//===----------------------------------------------------------------------===//
// Instrumentation macros (C)
//===----------------------------------------------------------------------===//

// Matches Tracy's PlotFormatType enum.
enum {
  // Values will be displayed as plain numbers.
  IREE_TRACING_PLOT_TYPE_NUMBER = 0,
  // Treats the values as memory sizes. Will display kilobytes, megabytes, etc.
  IREE_TRACING_PLOT_TYPE_MEMORY = 1,
  // Values will be displayed as percentage with value 100 being equal to 100%.
  IREE_TRACING_PLOT_TYPE_PERCENTAGE = 2,
};

// Colors used for messages based on the level provided to the macro.
enum {
  IREE_TRACING_MESSAGE_LEVEL_ERROR = 0xFF0000u,
  IREE_TRACING_MESSAGE_LEVEL_WARNING = 0xFFFF00u,
  IREE_TRACING_MESSAGE_LEVEL_INFO = 0xFFFFFFu,
  IREE_TRACING_MESSAGE_LEVEL_VERBOSE = 0xC0C0C0u,
  IREE_TRACING_MESSAGE_LEVEL_DEBUG = 0x00FF00u,
};

#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION)

// Evaluates the expression code only if tracing is enabled.
//
// Example:
//  struct {
//    IREE_TRACE(uint32_t trace_only_value);
//  } my_object;
//  IREE_TRACE(my_object.trace_only_value = 5);
#define IREE_TRACE(expr)

// Notifies the tracing implementation the app is about to start.
// No tracing APIs should be called prior to this point as implementations may
// use this to perform initialization.
#define IREE_TRACE_APP_ENTER()

// Notifies the tracing implementation that the app is about to exit.
// This allows implementations to flush their buffers. |exit_code| may be
// provided to indicate the application exit code if available.
// No tracing APIs should be called after this point as implementations may use
// this to perform deinitialization.
#define IREE_TRACE_APP_EXIT(exit_code)

// Sets an application-specific payload that will be stored in the trace.
// This can be used to fingerprint traces to particular versions and denote
// compilation options or configuration. The given string value will be copied.
#define IREE_TRACE_SET_APP_INFO(value, value_length)

// Sets the current thread name to the given string value.
// This will only set the thread name as it appears in the tracing backend and
// not set the OS thread name as it would appear in a debugger.
// The C-string |name| will be copied and does not need to be a literal.
#define IREE_TRACE_SET_THREAD_NAME(name)

// Enters a fiber context.
// |fiber| must be a unique pointer and remain live for the process lifetime.
#define IREE_TRACE_FIBER_ENTER(fiber)
// Exits a fiber context.
#define IREE_TRACE_FIBER_LEAVE()

// Publishes a source file to the tracing infrastructure.
// The filename and contents are copied and need not live longer than the call.
#define IREE_TRACE_PUBLISH_SOURCE_FILE(filename, filename_length, content, \
                                       content_length)

// Begins a new zone with the parent function name.
#define IREE_TRACE_ZONE_BEGIN(zone_id) \
  iree_zone_id_t zone_id = 0;          \
  (void)zone_id;
// Begins a new zone with the given compile-time |name_literal|.
// The literal must be static const and will be embedded in the trace buffer by
// reference.
#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) \
  IREE_TRACE_ZONE_BEGIN(zone_id)
// Begins a new zone with the given runtime dynamic string |name|.
// The |name| string will be copied into the trace buffer and does not require
// a NUL terminator.
#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length) \
  IREE_TRACE_ZONE_BEGIN(zone_id)
// Begins an externally defined zone with a dynamic source location.
// The |file_name|, |function_name|, and optional |name| strings will be copied
// into the trace buffer and do not need to persist or have a NUL terminator.
#define IREE_TRACE_ZONE_BEGIN_EXTERNAL(                        \
    zone_id, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)                   \
  IREE_TRACE_ZONE_BEGIN(zone_id)

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_TRACE_ZONE_END(zone_id) (void)(zone_id)

// Ends the current zone before returning on a failure.
// Sugar for IREE_TRACE_ZONE_END + IREE_RETURN_IF_ERROR.
#define IREE_RETURN_AND_END_ZONE_IF_ERROR(zone_id, ...) \
  IREE_RETURN_IF_ERROR(__VA_ARGS__)

// Sets the dynamic color of the zone to an XXBBGGRR value.
#define IREE_TRACE_ZONE_SET_COLOR(zone_id, color_xrgb)

// Appends an int64_t value to the parent zone. May be called multiple times.
#define IREE_TRACE_ZONE_APPEND_VALUE_I64(zone_id, value)

// Appends a string value to the parent zone. May be called multiple times.
// The provided NUL-terminated C string or string view will be copied into the
// trace buffer.
#define IREE_TRACE_ZONE_APPEND_TEXT(zone_id, ...)

// Configures the named plot with an IREE_TRACING_PLOT_TYPE_* representation.
#define IREE_TRACE_SET_PLOT_TYPE(name_literal, plot_type, step, fill, color)
// Plots a value in the named plot group as an int64_t.
#define IREE_TRACE_PLOT_VALUE_I64(name_literal, value)
// Plots a value in the named plot group as a single-precision float.
#define IREE_TRACE_PLOT_VALUE_F32(name_literal, value)
// Plots a value in the named plot group as a double-precision float.
#define IREE_TRACE_PLOT_VALUE_F64(name_literal, value)

// Demarcates an advancement of the top-level unnamed frame group.
#define IREE_TRACE_FRAME_MARK()
// Demarcates an advancement of a named frame group.
#define IREE_TRACE_FRAME_MARK_NAMED(name_literal)
// Begins a discontinuous frame in a named frame group.
// Must be properly matched with a IREE_TRACE_FRAME_MARK_NAMED_END.
#define IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal)
// Ends a discontinuous frame in a named frame group.
#define IREE_TRACE_FRAME_MARK_END_NAMED(name_literal)

// Logs a message at the given logging level to the trace.
// The message text must be a compile-time string literal.
#define IREE_TRACE_MESSAGE(level, value_literal)
// Logs a message with the given color to the trace.
// Standard colors are defined as IREE_TRACING_MESSAGE_LEVEL_* values.
// The message text must be a compile-time string literal.
#define IREE_TRACE_MESSAGE_COLORED(color, value_literal)
// Logs a dynamically-allocated message at the given logging level to the trace.
// The string |value| will be copied into the trace buffer.
#define IREE_TRACE_MESSAGE_DYNAMIC(level, value, value_length)
// Logs a dynamically-allocated message with the given color to the trace.
// Standard colors are defined as IREE_TRACING_MESSAGE_LEVEL_* values.
// The string |value| will be copied into the trace buffer.
#define IREE_TRACE_MESSAGE_DYNAMIC_COLORED(color, value, value_length)

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// Allocation tracking macros (C/C++)
//===----------------------------------------------------------------------===//

#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING)

static void* iree_tracing_obscure_ptr(void* ptr) { return ptr; }

// Traces a new memory allocation with host |ptr| and the given |size|.
// A balanced IREE_TRACE_FREE on the same |ptr| is required for proper memory
// tracking. Allocations will be attributed to their parent zone.
// Reallocations must be recorded as an IREE_TRACE_ALLOC/IREE_TRACE_FREE pair.
#define IREE_TRACE_ALLOC(ptr, size)

// Traces a free of an existing allocation traced with IREE_TRACE_ALLOC.
#define IREE_TRACE_FREE(ptr)

// Traces a new memory allocation in a named memory pool.
// The |name_literal| pointer must be static const and be identical to the
// one passed to a balanced IREE_TRACE_FREE_NAMED. This is best accomplished
// with a compilation-unit scoped `static const char* MY_POOL_ID = "foo";` as
// in debug modes compilers may not deduplicate local literals.
// Reallocations must be recorded as an
// IREE_TRACE_ALLOC_NAMED/IREE_TRACE_FREE_NAMED pair.
#define IREE_TRACE_ALLOC_NAMED(name_literal, ptr, size)

// Traces a free of an existing allocation traced with IREE_TRACE_ALLOC_NAMED.
// Note that the |name_literal| must be the same pointer as passed to the alloc.
#define IREE_TRACE_FREE_NAMED(name_literal, ptr)

#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// Instrumentation C++ RAII types, wrappers, and macros
//===----------------------------------------------------------------------===//

#ifdef __cplusplus

#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION)

// Automatically instruments the calling scope as if calling
// IREE_TRACE_ZONE_BEGIN/IREE_TRACE_ZONE_END.
// The scope's zone ID can be retrieved with IREE_TRACE_SCOPE_ID for use with
// other tracing macros.
#define IREE_TRACE_SCOPE()

// Automatically instruments the calling scope as if calling
// IREE_TRACE_ZONE_BEGIN_NAMED/IREE_TRACE_ZONE_END.
// The scope's zone ID can be retrieved with IREE_TRACE_SCOPE_ID for use with
// other tracing macros.
#define IREE_TRACE_SCOPE_NAMED(name_literal)

// The zone ID of the current scope as set by IREE_TRACE_SCOPE.
// This can be passed to other tracing macros like IREE_TRACE_ZONE_APPEND_TEXT.
#define IREE_TRACE_SCOPE_ID

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

#endif  // __cplusplus

#endif  // IREE_BASE_TRACING_H_
