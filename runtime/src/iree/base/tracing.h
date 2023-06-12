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

// Tracks all allocations (we know about) via new/delete/malloc/free.
// This allows fine-grained allocation and usage tracking down to the code that
// performed the allocations. Allocations or frees that are performed outside of
// the IREE API or runtime library will not be tracked and unbalanced usage
// (allocating with IREE's API then freeing with stdlib free, for example) will
// cause Tracy to become very unhappy.
#define IREE_TRACING_FEATURE_ALLOCATION_TRACKING (1 << 2)

// Captures callstacks up to IREE_TRACING_MAX_CALLSTACK_DEPTH at all allocation
// events when allocation tracking is enabled.
#define IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS (1 << 3)

// Tracks fast locks in all cases (both contended and uncontended).
// This may introduce contention where there would otherwise be none as what
// would be a handful of instructions and little memory access may become
// hundreds. To see only locks under contention use
// IREE_TRACING_FEATURE_SLOW_LOCKS.
#define IREE_TRACING_FEATURE_FAST_LOCKS (1 << 4)

// Tracks slow locks that end up going to the OS for waits/wakes in futexes.
// Uncontended locks will not be displayed and only waits will be visible in the
// Tracy UI.
#define IREE_TRACING_FEATURE_SLOW_LOCKS (1 << 5)

// Forwards log messages to traces, which will be visible under "Messages" in
// the Tracy UI.
#define IREE_TRACING_FEATURE_LOG_MESSAGES (1 << 6)

// Enables fiber support in the Tracy UI.
// Comes with a per-event overhead (less efficient queue insertion) but is
// required when running with asynchronous VM invocations.
#define IREE_TRACING_FEATURE_FIBERS (1 << 7)

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

// Set IREE_TRACING_FEATURES based on IREE_TRACING_MODE if the user hasn't
// overridden it with more specific settings.
//
// IREE_TRACING_MODE = 0: tracing disabled
// IREE_TRACING_MODE = 1: instrumentation, log messages, and basic statistics
// IREE_TRACING_MODE = 2: same as 1 with added allocation tracking
// IREE_TRACING_MODE = 3: same as 2 with callstacks for allocations
// IREE_TRACING_MODE = 4: same as 3 with callstacks for all instrumentation
#if !defined(IREE_TRACING_FEATURES)
#if defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 1
#define IREE_TRACING_FEATURES \
  (IREE_TRACING_FEATURE_INSTRUMENTATION | IREE_TRACING_FEATURE_LOG_MESSAGES)
#undef IREE_TRACING_MAX_CALLSTACK_DEPTH
#define IREE_TRACING_MAX_CALLSTACK_DEPTH 0
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 2
#define IREE_TRACING_FEATURES                 \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |     \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING | \
   IREE_TRACING_FEATURE_LOG_MESSAGES)
// TODO(#9627): make tracy fibers faster; too slow for on-by-default!
// | IREE_TRACING_FEATURE_FIBERS)
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 3
#define IREE_TRACING_FEATURES                   \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |       \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING |   \
   IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS | \
   IREE_TRACING_FEATURE_LOG_MESSAGES | IREE_TRACING_FEATURE_FIBERS)
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE >= 4
#define IREE_TRACING_FEATURES                        \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |            \
   IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS | \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING |        \
   IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS |      \
   IREE_TRACING_FEATURE_LOG_MESSAGES | IREE_TRACING_FEATURE_FIBERS)
#else
#define IREE_TRACING_FEATURES 0
#endif  // IREE_TRACING_MODE
#endif  // !IREE_TRACING_FEATURES

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
#define IREE_TRACE_SET_APP_INFO(value, value_length)
#define IREE_TRACE_SET_THREAD_NAME(name)
#define IREE_TRACE(expr)
#define IREE_TRACE_FIBER_ENTER(fiber)
#define IREE_TRACE_FIBER_LEAVE()
#define IREE_TRACE_ZONE_BEGIN(zone_id) \
  iree_zone_id_t zone_id = 0;          \
  (void)zone_id;
#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal) \
  IREE_TRACE_ZONE_BEGIN(zone_id)
#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length) \
  IREE_TRACE_ZONE_BEGIN(zone_id)
#define IREE_TRACE_ZONE_BEGIN_EXTERNAL(                        \
    zone_id, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)                   \
  IREE_TRACE_ZONE_BEGIN(zone_id)
#define IREE_TRACE_ZONE_SET_COLOR(zone_id, color_xrgb)
#define IREE_TRACE_ZONE_APPEND_VALUE(zone_id, value)
#define IREE_TRACE_ZONE_APPEND_TEXT(zone_id, ...)
#define IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(zone_id, value)
#define IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value, value_length)
#define IREE_TRACE_ZONE_END(zone_id)
#define IREE_RETURN_AND_END_ZONE_IF_ERROR(zone_id, ...) \
  IREE_RETURN_IF_ERROR(__VA_ARGS__)
#define IREE_TRACE_SET_PLOT_TYPE(name_literal, plot_type, step, fill, color)
#define IREE_TRACE_PLOT_VALUE_I64(name_literal, value)
#define IREE_TRACE_PLOT_VALUE_F32(name_literal, value)
#define IREE_TRACE_PLOT_VALUE_F64(name_literal, value)
#define IREE_TRACE_FRAME_MARK()
#define IREE_TRACE_FRAME_MARK_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_END_NAMED(name_literal)
#define IREE_TRACE_MESSAGE(level, value_literal)
#define IREE_TRACE_MESSAGE_COLORED(color, value_literal)
#define IREE_TRACE_MESSAGE_DYNAMIC(level, value, value_length)
#define IREE_TRACE_MESSAGE_DYNAMIC_COLORED(color, value, value_length)
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// Allocation tracking macros (C/C++)
//===----------------------------------------------------------------------===//
//
// IREE_TRACE_ALLOC: records an malloc.
// IREE_TRACE_FREE: records a free.
//
// NOTE: realloc must be recorded as a FREE/ALLOC pair.

#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING)
#define IREE_TRACE_ALLOC(ptr, size)
#define IREE_TRACE_FREE(ptr)
#define IREE_TRACE_ALLOC_NAMED(name, ptr, size)
#define IREE_TRACE_FREE_NAMED(name, ptr)
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// Instrumentation C++ RAII types, wrappers, and macros
//===----------------------------------------------------------------------===//

#ifdef __cplusplus

#if !(IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION)
#define IREE_TRACE_SCOPE()
#define IREE_TRACE_SCOPE_NAMED(name_literal)
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Tracing implementation
//===----------------------------------------------------------------------===//

#include "iree/base/tracing/tracy.h"

#endif  // IREE_BASE_TRACING_H_
