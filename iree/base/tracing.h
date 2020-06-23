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

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "absl/base/attributes.h"

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
// IREE_TRACING_MODE = 1: instrumentation and basic statistics
// IREE_TRACING_MODE = 2: same as 1 with added allocation tracking
// IREE_TRACING_MODE = 3: same as 2 with callstacks for allocations
// IREE_TRACING_MODE = 4: same as 3 with callstacks for all instrumentation
#if !defined(IREE_TRACING_FEATURES)
#if defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 1
#define IREE_TRACING_FEATURES (IREE_TRACING_FEATURE_INSTRUMENTATION)
#undef IREE_TRACING_MAX_CALLSTACK_DEPTH
#define IREE_TRACING_MAX_CALLSTACK_DEPTH 0
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 2
#define IREE_TRACING_FEATURES             \
  (IREE_TRACING_FEATURE_INSTRUMENTATION | \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING)
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE == 3
#define IREE_TRACING_FEATURES                 \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |     \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING | \
   IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS)
#elif defined(IREE_TRACING_MODE) && IREE_TRACING_MODE >= 4
#define IREE_TRACING_FEATURES                        \
  (IREE_TRACING_FEATURE_INSTRUMENTATION |            \
   IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS | \
   IREE_TRACING_FEATURE_ALLOCATION_TRACKING |        \
   IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS)
#else
#define IREE_TRACING_FEATURES 0
#endif  // IREE_TRACING_MODE
#endif  // !IREE_TRACING_FEATURES

//===----------------------------------------------------------------------===//
// Tracy configuration
//===----------------------------------------------------------------------===//
// NOTE: order matters here as we are including files that require/define.

// Enable Tracy only when we are using tracing features.
#if IREE_TRACING_FEATURES != 0
#define TRACY_ENABLE 1
#endif  // IREE_TRACING_FEATURES

// Disable zone nesting verification in release builds.
// The verification makes it easy to find unbalanced zones but doubles the cost
// (at least) of each zone recorded. Run in debug builds to verify new
// instrumentation is correct before capturing traces in release builds.
#if defined(NDEBUG)
#define TRACY_NO_VERIFY 1
#endif  // NDEBUG

// Force callstack capture on all zones (even those without the C suffix).
#if (IREE_TRACING_FEATURES &                             \
     IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS) || \
    (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS)
#define TRACY_CALLSTACK 1
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS

// TODO(#1926): upstream a TRACY_NO_FRAME_IMAGE flag to remove the frame
// compression thread and dxt1 compression code.

// Flush the settings we have so far; settings after this point will be
// overriding values set by Tracy itself.
#if defined(TRACY_ENABLE)
#include "third_party/tracy/TracyC.h"  // IWYU pragma: export
#endif

// Disable callstack capture if our depth is 0; this allows us to avoid any
// expensive capture (and all the associated dependencies) if we aren't going to
// use it. Note that this means that unless code is instrumented we won't be
// able to tell what's happening in the Tracy UI.
#if IREE_TRACING_MAX_CALLSTACK_DEPTH == 0
#undef TRACY_HAS_CALLSTACK
#endif  // IREE_TRACING_MAX_CALLSTACK_DEPTH

//===----------------------------------------------------------------------===//
// C API used for Tracy control
//===----------------------------------------------------------------------===//
// These functions are implementation details and should not be called directly.
// Always use the macros (or C++ RAII types).

// Local zone ID used for the C IREE_TRACE_ZONE_* macros.
typedef uint32_t iree_zone_id_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if IREE_TRACING_FEATURES

void iree_tracing_set_thread_name_impl(const char* name);

ABSL_MUST_USE_RESULT iree_zone_id_t iree_tracing_zone_begin_impl(
    const struct ___tracy_source_location_data* src_loc, const char* name,
    size_t name_length);
ABSL_MUST_USE_RESULT iree_zone_id_t iree_tracing_zone_begin_external_impl(
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length);

void iree_tracing_set_plot_type_impl(const char* name_literal,
                                     uint8_t plot_type);
void iree_tracing_plot_value_i64_impl(const char* name_literal, int64_t value);
void iree_tracing_plot_value_f32_impl(const char* name_literal, float value);
void iree_tracing_plot_value_f64_impl(const char* name_literal, double value);

#endif  // IREE_TRACING_FEATURES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

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

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Sets an application-specific payload that will be stored in the trace.
// This can be used to fingerprint traces to particular versions and denote
// compilation options or configuration. The given string value will be copied.
#define IREE_TRACE_SET_APP_INFO(value, value_length) \
  ___tracy_emit_message_appinfo(value, value_length)

// Sets the current thread name to the given string value.
// This will only set the thread name as it appears in the tracing backend and
// not set the OS thread name as it would appear in a debugger.
// The C-string |name| will be copied and does not need to be a literal.
#define IREE_TRACE_SET_THREAD_NAME(name) iree_tracing_set_thread_name_impl(name)

// Begins a new zone with the parent function name.
#define IREE_TRACE_ZONE_BEGIN(zone_id) \
  IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, NULL)

// Begins a new zone with the given compile-time literal name.
#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal)                    \
  static const struct ___tracy_source_location_data TracyConcat(              \
      __tracy_source_location, __LINE__) = {name_literal, __FUNCTION__,       \
                                            __FILE__, (uint32_t)__LINE__, 0}; \
  iree_zone_id_t zone_id = iree_tracing_zone_begin_impl(                      \
      &TracyConcat(__tracy_source_location, __LINE__), NULL, 0);

// Begins a new zone with the given runtime dynamic string name.
// The |value| string will be copied into the trace buffer.
#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length)   \
  static const struct ___tracy_source_location_data TracyConcat(          \
      __tracy_source_location, __LINE__) = {NULL, __FUNCTION__, __FILE__, \
                                            (uint32_t)__LINE__, 0};       \
  iree_zone_id_t zone_id = iree_tracing_zone_begin_impl(                  \
      &TracyConcat(__tracy_source_location, __LINE__), name, name_length);

// Begins an externally defined zone with a dynamic source location.
// The |file_name|, |function_name|, and optional |name| strings will be copied
// into the trace buffer and do not need to persist.
#define IREE_TRACE_ZONE_BEGIN_EXTERNAL(                                       \
    zone_id, file_name, file_name_length, line, function_name,                \
    function_name_length, name, name_length)                                  \
  iree_zone_id_t zone_id = iree_tracing_zone_begin_external_impl(             \
      file_name, file_name_length, line, function_name, function_name_length, \
      name, name_length)

// Appends a string value to the parent zone. May be called multiple times.
// The |value| string will be copied into the trace buffer.
#define IREE_TRACE_ZONE_APPEND_TEXT(...)                                  \
  IREE_TRACE_IMPL_GET_VARIADIC_((__VA_ARGS__,                             \
                                 IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW, \
                                 IREE_TRACE_ZONE_APPEND_TEXT_CSTRING))    \
  (__VA_ARGS__)
#define IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(zone_id, value) \
  IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value, strlen(value))
#define IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value, value_length)  \
  ___tracy_emit_zone_text((struct ___tracy_c_zone_context){zone_id, 1}, value, \
                          value_length)

// Ends the current zone. Must be passed the |zone_id| from the _BEGIN.
#define IREE_TRACE_ZONE_END(zone_id) \
  ___tracy_emit_zone_end((struct ___tracy_c_zone_context){zone_id, 1})

// Configures the named plot with an IREE_TRACING_PLOT_TYPE_* representation.
#define IREE_TRACE_SET_PLOT_TYPE(name_literal, plot_type) \
  iree_tracing_set_plot_type_impl(name_literal, plot_type)
// Plots a value in the named plot group as an integer.
#define IREE_TRACE_PLOT_VALUE_I64(name_literal, value) \
  iree_tracing_plot_value_i64_impl(name_literal, value)
// Plots a value in the named plot group as a single-precision float.
#define IREE_TRACE_PLOT_VALUE_F32(name_literal, value) \
  iree_tracing_plot_value_f32_impl(name_literal, value)
// Plots a value in the named plot group as a double-precision float.
#define IREE_TRACE_PLOT_VALUE_F64(name_literal, value) \
  iree_tracing_plot_value_f64_impl(name_literal, value)

// Demarcates an advancement of the top-level unnamed frame group.
#define IREE_TRACE_FRAME_MARK() ___tracy_emit_frame_mark(NULL)
// Demarcates an advancement of a named frame group.
#define IREE_TRACE_FRAME_MARK_NAMED(name_literal) \
  ___tracy_emit_frame_mark(name_literal)
// Begins a discontinuous frame in a named frame group.
// Must be properly matched with a IREE_TRACE_FRAME_MARK_NAMED_END.
#define IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal) \
  ___tracy_emit_frame_mark_start(name_literal)
// Ends a discontinuous frame in a named frame group.
#define IREE_TRACE_FRAME_MARK_END_NAMED(name_literal) \
  ___tracy_emit_frame_mark_end(name_literal)

// Logs a message at the given logging level to the trace.
// The message text must be a compile-time string literal.
#define IREE_TRACE_MESSAGE(level, value_literal) \
  ___tracy_emit_messageLC(value_literal, IREE_TRACING_MESSAGE_LEVEL_##level, 0)
// Logs a dynamically-allocated message at the given logging level to the trace.
// The string |value| will be copied into the trace buffer.
#define IREE_TRACE_MESSAGE_DYNAMIC(level, value, value_length) \
  ___tracy_emit_messageC(value, value_length,                  \
                         IREE_TRACING_MESSAGE_LEVEL_##level, 0)

// Utilities:
#define IREE_TRACE_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME
#define IREE_TRACE_IMPL_GET_VARIADIC_(args) \
  IREE_TRACE_IMPL_GET_VARIADIC_HELPER_ args

#else
#define IREE_TRACE_SET_APP_INFO(value, value_length)
#define IREE_TRACE_SET_THREAD_NAME(name)
#define IREE_TRACE_ZONE_BEGIN(zone_id)
#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal)
#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length)
#define IREE_TRACE_ZONE_BEGIN_EXTERNAL(                        \
    zone_id, file_name, file_name_length, line, function_name, \
    function_name_length, name, name_length)
#define IREE_TRACE_ZONE_APPEND_TEXT(zone_id, value, value_length)
#define IREE_TRACE_ZONE_END(zone_id)
#define IREE_TRACE_SET_PLOT_TYPE(name_literal, plot_type)
#define IREE_TRACE_PLOT_VALUE_I64(name_literal, value)
#define IREE_TRACE_PLOT_VALUE_F32(name_literal, value)
#define IREE_TRACE_PLOT_VALUE_F64(name_literal, value)
#define IREE_TRACE_FRAME_MARK()
#define IREE_TRACE_FRAME_MARK_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_END_NAMED(name_literal)
#define IREE_TRACE_MESSAGE(level, value_literal)
#define IREE_TRACE_MESSAGE_DYNAMIC(level, value, value_length)
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// Allocation tracking macros (C/C++)
//===----------------------------------------------------------------------===//
//
// IREE_TRACE_ALLOC: records an malloc.
// IREE_TRACE_FREE: records a free.
//
// NOTE: realloc must be recorded as a FREE/ALLOC pair.

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS

#define IREE_TRACE_ALLOC(ptr, size)               \
  ___tracy_emit_memory_alloc_callstack(ptr, size, \
                                       IREE_TRACING_MAX_CALLSTACK_DEPTH)
#define IREE_TRACE_FREE(ptr) \
  ___tracy_emit_memory_free_callstack(ptr, IREE_TRACING_MAX_CALLSTACK_DEPTH)

#else

#define IREE_TRACE_ALLOC(ptr, size) ___tracy_emit_memory_alloc(ptr, size)
#define IREE_TRACE_FREE(ptr) ___tracy_emit_memory_free(ptr)

#endif  // IREE_TRACING_FEATURE_ALLOCATION_CALLSTACKS

#else
#define IREE_TRACE_ALLOC(ptr, size)
#define IREE_TRACE_FREE(ptr)
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

#ifdef __cplusplus

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING

inline void* operator new(size_t count) {
  auto ptr = malloc(count);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}

inline void operator delete(void* ptr) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}

#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Instrumentation C++ RAII types, wrappers, and macros
//===----------------------------------------------------------------------===//

#ifdef __cplusplus

#if defined(TRACY_ENABLE)
#include "third_party/tracy/Tracy.hpp"  // IWYU pragma: export
#endif

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// TODO(#1886): update these to tracy and drop the 0.
#define IREE_TRACE_SCOPE0(name_spec) ZoneScopedNS(name_spec, 13)
#define IREE_TRACE_SCOPE(name_spec, ...)
#define IREE_TRACE_EVENT0
#define IREE_TRACE_EVENT

#else
#define IREE_TRACE_THREAD_ENABLE(name)
#define IREE_TRACE_SCOPE0(name_spec)
#define IREE_TRACE_SCOPE(name_spec, ...) (void)
#define IREE_TRACE_EVENT0
#define IREE_TRACE_EVENT(void)
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

// TODO(benvanik): macros for LockableCtx / Lockable mutex tracking.

#endif  // __cplusplus

#endif  // IREE_BASE_TRACING_H_
