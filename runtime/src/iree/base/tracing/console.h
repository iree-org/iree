// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"

#ifndef IREE_BASE_TRACING_CONSOLE_H_
#define IREE_BASE_TRACING_CONSOLE_H_

//===----------------------------------------------------------------------===//
// Console tracing configuration
//===----------------------------------------------------------------------===//

// Filter to only supported features.
#if !defined(IREE_TRACING_FEATURES)
#define IREE_TRACING_FEATURES          \
  ((IREE_TRACING_FEATURES_REQUESTED) & \
   (IREE_TRACING_FEATURE_INSTRUMENTATION | IREE_TRACING_FEATURE_LOG_MESSAGES))
#endif  // !IREE_TRACING_FEATURES

// File handle that tracing information will be dumped to.
// We could make this an extern variable to allow for programmatic overriding.
#if !defined(IREE_TRACING_CONSOLE_FILE)
#define IREE_TRACING_CONSOLE_FILE stderr
#endif  // !IREE_TRACING_CONSOLE_FILE

// Flush after every log. This can be useful when debugging crashes but has a
// significant performance overhead and should be left off by default.
#if !defined(IREE_TRACING_CONSOLE_FLUSH)
#define IREE_TRACING_CONSOLE_FLUSH 0
#endif  // !IREE_TRACING_CONSOLE_FLUSH

// Whether to enable printing of end zones and the total zone timing.
#if !defined(IREE_TRACING_CONSOLE_TIMING)
#define IREE_TRACING_CONSOLE_TIMING 1
#endif  // !IREE_TRACING_CONSOLE_TIMING

//===----------------------------------------------------------------------===//
// C API used for tracing control
//===----------------------------------------------------------------------===//
// These functions are implementation details and should not be called directly.
// Always use the macros (or C++ RAII types).

// Local zone ID used for the C IREE_TRACE_ZONE_* macros.
typedef uint32_t iree_zone_id_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if IREE_TRACING_FEATURES

#define IREE_TRACE_IMPL_CONCAT(x, y) IREE_TRACE_IMPL_CONCAT2(x, y)
#define IREE_TRACE_IMPL_CONCAT2(x, y) x##y

#define IREE_TRACE_STRLEN(literal) (sizeof(literal) - 1)

typedef struct iree_tracing_location_t {
  const char* name;
  size_t name_length;
  const char* function_name;
  size_t function_name_length;
  const char* file_name;
  size_t file_name_length;
  uint32_t line;
  uint32_t color;
} iree_tracing_location_t;

#define iree_tracing_make_zone_ctx(zone_id) (zone_id)

void iree_tracing_console_initialize();
void iree_tracing_console_deinitialize();

void iree_tracing_set_thread_name(const char* name);

IREE_MUST_USE_RESULT iree_zone_id_t
iree_tracing_zone_begin_impl(const iree_tracing_location_t* src_loc,
                             const char* name, size_t name_length);
IREE_MUST_USE_RESULT iree_zone_id_t iree_tracing_zone_begin_external_impl(
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length);
void iree_tracing_zone_end(iree_zone_id_t zone_id);

void iree_tracing_message_cstring(const char* value, const char* symbol,
                                  uint32_t color);
void iree_tracing_message_string_view(const char* value, size_t value_length,
                                      const char* symbol, uint32_t color);

#endif  // IREE_TRACING_FEATURES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Instrumentation macros (C)
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#define IREE_TRACING_MESSAGE_LEVEL_ERROR_SYMBOL "!"
#define IREE_TRACING_MESSAGE_LEVEL_WARNING_SYMBOL "w"
#define IREE_TRACING_MESSAGE_LEVEL_INFO_SYMBOL "i"
#define IREE_TRACING_MESSAGE_LEVEL_VERBOSE_SYMBOL "v"
#define IREE_TRACING_MESSAGE_LEVEL_DEBUG_SYMBOL "d"

#define IREE_TRACE(expr) expr

#define IREE_TRACE_APP_ENTER() iree_tracing_console_initialize()
#define IREE_TRACE_APP_EXIT(exit_code) iree_tracing_console_deinitialize()
#define IREE_TRACE_SET_APP_INFO(value, value_length)
#define IREE_TRACE_SET_THREAD_NAME(name) iree_tracing_set_thread_name(name)

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FIBERS
// TODO(benvanik): console tracing fiber markers.
#define IREE_TRACE_FIBER_ENTER(fiber)
#define IREE_TRACE_FIBER_LEAVE()
#else
#define IREE_TRACE_FIBER_ENTER(fiber)
#define IREE_TRACE_FIBER_LEAVE()
#endif  // IREE_TRACING_FEATURE_FIBERS

#define IREE_TRACE_ZONE_BEGIN(zone_id) \
  IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, NULL)

#define IREE_TRACE_ZONE_BEGIN_NAMED(zone_id, name_literal)                     \
  static const iree_tracing_location_t IREE_TRACE_IMPL_CONCAT(                 \
      __iree_tracing_source_location, __LINE__) = {                            \
      name_literal,       IREE_TRACE_STRLEN(name_literal),                     \
      __FUNCTION__,       IREE_TRACE_STRLEN(__FUNCTION__),                     \
      __FILE__,           IREE_TRACE_STRLEN(__FILE__),                         \
      (uint32_t)__LINE__, 0};                                                  \
  iree_zone_id_t zone_id = iree_tracing_zone_begin_impl(                       \
      &IREE_TRACE_IMPL_CONCAT(__iree_tracing_source_location, __LINE__), NULL, \
      0)

#define IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(zone_id, name, name_length)  \
  static const iree_tracing_location_t IREE_TRACE_IMPL_CONCAT(           \
      __iree_tracing_source_location, __LINE__) = {                      \
      NULL,                                                              \
      0,                                                                 \
      __FUNCTION__,                                                      \
      IREE_TRACE_STRLEN(__FUNCTION__),                                   \
      __FILE__,                                                          \
      IREE_TRACE_STRLEN(__FILE__),                                       \
      (uint32_t)__LINE__,                                                \
      0};                                                                \
  iree_zone_id_t zone_id = iree_tracing_zone_begin_impl(                 \
      &IREE_TRACE_IMPL_CONCAT(__iree_tracing_source_location, __LINE__), \
      (name), (name_length))

#define IREE_TRACE_ZONE_BEGIN_EXTERNAL(                                       \
    zone_id, file_name, file_name_length, line, function_name,                \
    function_name_length, name, name_length)                                  \
  iree_zone_id_t zone_id = iree_tracing_zone_begin_external_impl(             \
      file_name, file_name_length, line, function_name, function_name_length, \
      name, name_length)

#define IREE_TRACE_ZONE_END(zone_id) iree_tracing_zone_end(zone_id)

#define IREE_RETURN_AND_END_ZONE_IF_ERROR(zone_id, ...) \
  IREE_RETURN_AND_EVAL_IF_ERROR(IREE_TRACE_ZONE_END(zone_id), __VA_ARGS__)

// TODO(benvanik): move zone color to BEGIN macro so that we can print the line
// with it initially.
#define IREE_TRACE_ZONE_SET_COLOR(zone_id, color_xbgr)

// TODO(benvanik): console tracing zone append value/text. This currently is
// tricky as we don't buffer and interleaving multi-threaded tracing would cause
// issues. If we buffered until a matching zone end we may be able to catch
// most of these.
#define IREE_TRACE_ZONE_APPEND_VALUE_I64(zone_id, value) (void)(value)
#define IREE_TRACE_ZONE_APPEND_TEXT(...)                                  \
  IREE_TRACE_IMPL_GET_VARIADIC_((__VA_ARGS__,                             \
                                 IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW, \
                                 IREE_TRACE_ZONE_APPEND_TEXT_CSTRING))    \
  (__VA_ARGS__)
#define IREE_TRACE_ZONE_APPEND_TEXT_CSTRING(zone_id, value) (void)(value)
#define IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(zone_id, value, value_length) \
  (void)(value), (void)(value_length)

// TODO(benvanik): console tracing plot support. We could track the plots and
// dump the statistics in IREE_TRACE_APP_EXIT.
#define IREE_TRACE_SET_PLOT_TYPE(name_literal, plot_type, step, fill, color) \
  (void)(name_literal), (void)(plot_type), (void)(step), (void)(fill),       \
      (void)(color)
#define IREE_TRACE_PLOT_VALUE_I64(name_literal, value) \
  (void)(name_literal), (void)(value)
#define IREE_TRACE_PLOT_VALUE_F32(name_literal, value) \
  (void)(name_literal), (void)(value)
#define IREE_TRACE_PLOT_VALUE_F64(name_literal, value) \
  (void)(name_literal), (void)(value)

// TODO(benvanik): console tracing frame support. We could use these to bound
// statistics or update progress indicators or something.
#define IREE_TRACE_FRAME_MARK()
#define IREE_TRACE_FRAME_MARK_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name_literal)
#define IREE_TRACE_FRAME_MARK_END_NAMED(name_literal)

#define IREE_TRACE_MESSAGE(level, value_literal)                            \
  iree_tracing_message_cstring(value_literal,                               \
                               IREE_TRACING_MESSAGE_LEVEL_##level##_SYMBOL, \
                               IREE_TRACING_MESSAGE_LEVEL_##level)
#define IREE_TRACE_MESSAGE_COLORED(color, value_literal) \
  iree_tracing_message_cstring(value_literal, " ", color)
#define IREE_TRACE_MESSAGE_DYNAMIC(level, value, value_length)          \
  iree_tracing_message_string_view(                                     \
      value, value_length, IREE_TRACING_MESSAGE_LEVEL_##level##_SYMBOL, \
      IREE_TRACING_MESSAGE_LEVEL_##level)
#define IREE_TRACE_MESSAGE_DYNAMIC_COLORED(color, value, value_length) \
  iree_tracing_message_string_view(value, value_length, " ", color)

// Utilities:
#define IREE_TRACE_IMPL_GET_VARIADIC_HELPER_(_1, _2, _3, NAME, ...) NAME
#define IREE_TRACE_IMPL_GET_VARIADIC_(args) \
  IREE_TRACE_IMPL_GET_VARIADIC_HELPER_ args

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// Allocation tracking macros (C/C++)
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING

// TODO(benvanik): console tracing allocation support. We could just print out
// pointers or keep track of statistics and dump them in IREE_TRACE_APP_EXIT.
#define IREE_TRACE_ALLOC(ptr, size)
#define IREE_TRACE_FREE(ptr)
#define IREE_TRACE_ALLOC_NAMED(name_literal, ptr, size)
#define IREE_TRACE_FREE_NAMED(name_literal, ptr)

#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// Instrumentation C++ RAII types, wrappers, and macros
//===----------------------------------------------------------------------===//

#ifdef __cplusplus

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

namespace iree {

class ScopedZone {
 public:
  ScopedZone(const ScopedZone&) = delete;
  ScopedZone(ScopedZone&&) = delete;
  ScopedZone& operator=(const ScopedZone&) = delete;
  ScopedZone& operator=(ScopedZone&&) = delete;

  IREE_ATTRIBUTE_ALWAYS_INLINE ScopedZone(
      const iree_tracing_location_t* src_loc) {
    zone_id_ = iree_tracing_zone_begin_impl(src_loc, NULL, 0);
  }
  IREE_ATTRIBUTE_ALWAYS_INLINE ~ScopedZone() { IREE_TRACE_ZONE_END(zone_id_); }

  operator iree_zone_id_t() const noexcept { return zone_id_; }

 private:
  iree_zone_id_t zone_id_;
};

}  // namespace iree

#define IREE_TRACE_SCOPE()                                         \
  static constexpr iree_tracing_location_t IREE_TRACE_IMPL_CONCAT( \
      __iree_tracing_source_location, __LINE__){                   \
      nullptr,                                                     \
      0,                                                           \
      __FUNCTION__,                                                \
      IREE_TRACE_STRLEN(__FUNCTION__),                             \
      __FILE__,                                                    \
      IREE_TRACE_STRLEN(__FILE__),                                 \
      (uint32_t)__LINE__,                                          \
      0};                                                          \
  ::iree::ScopedZone ___iree_tracing_scoped_zone(                  \
      &IREE_TRACE_IMPL_CONCAT(__iree_tracing_source_location, __LINE__))
#define IREE_TRACE_SCOPE_NAMED(name_literal)                       \
  static constexpr iree_tracing_location_t IREE_TRACE_IMPL_CONCAT( \
      __iree_tracing_source_location, __LINE__){                   \
      name_literal,       IREE_TRACE_STRLEN(name_literal),         \
      __FUNCTION__,       IREE_TRACE_STRLEN(__FUNCTION__),         \
      __FILE__,           IREE_TRACE_STRLEN(__FILE__),             \
      (uint32_t)__LINE__, 0};                                      \
  ::iree::ScopedZone ___iree_tracing_scoped_zone(                  \
      &IREE_TRACE_IMPL_CONCAT(__iree_tracing_source_location, __LINE__))
#define IREE_TRACE_SCOPE_ID ___iree_tracing_scoped_zone

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION

#endif  // __cplusplus

#endif  // IREE_BASE_TRACING_CONSOLE_H_
