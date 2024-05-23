// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/time.h"
#include "iree/base/tracing.h"

// NOTE: threading support is optional.
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#define iree_thread_local static
#define iree_thread_id() 0

#else

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201102L) && \
    !__STDC_NO_THREADS__
#define iree_thread_local _Thread_local
#elif defined(IREE_COMPILER_MSVC)
#define iree_thread_local __declspec(thread)
#else
#define iree_thread_local
#endif  // __STDC_NO_THREADS__

#if defined(IREE_PLATFORM_ANDROID)
#include <unistd.h>
#define iree_thread_id() ((uint64_t)gettid())
#elif defined(IREE_PLATFORM_APPLE)
#include <pthread.h>
#define iree_thread_id() ((uint64_t)pthread_mach_thread_np(pthread_self()))
#elif defined(IREE_PLATFORM_LINUX)
#include <sys/syscall.h>
#include <unistd.h>
#define iree_thread_id() ((uint64_t)syscall(__NR_gettid))
#elif defined(IREE_PLATFORM_WINDOWS)
#define iree_thread_id() ((uint64_t)GetCurrentThreadId())
#else
#define iree_thread_id() 0
#endif  // IREE_PLATFORM_*

#endif  // IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#if IREE_TRACING_FEATURES

typedef struct iree_tracing_console_t {
  // The file that all tracing output is routed to.
  FILE* file;

  // TODO(benvanik): storage for a fixed number of plots.
  // TODO(benvanik): storage for memory statistics (global and named pools).
} iree_tracing_console_t;

// Global shared console tracing context.
// We could potentially allow for multiple to exist but that will require apps
// to manage the lifetime. Most command line apps would have an easy time but
// garbage collection and long-lived processes make it really tricky without
// also capturing the tracing context and ref counting it everywhere.
static iree_tracing_console_t _console = {0};

void iree_tracing_console_initialize() {
  if (_console.file) return;
  _console.file = IREE_TRACING_CONSOLE_FILE;
}

void iree_tracing_console_deinitialize() {
  if (!_console.file) return;
  fflush(_console.file);
}

#define IREE_TRACING_MAX_THREAD_LENGTH 16

typedef struct iree_trace_zone_t {
  uint64_t start_timestamp_ns;
  char name[64];
  int16_t name_length;
} iree_trace_zone_t;

typedef struct iree_trace_thread_t {
  char name[IREE_TRACING_MAX_THREAD_LENGTH];
  int16_t name_length;
  uint32_t depth;  // 0 = no zone open
#if IREE_TRACING_CONSOLE_TIMING
  iree_trace_zone_t stack[128];
#endif  // IREE_TRACING_CONSOLE_TIMING
} iree_trace_thread_t;
static iree_thread_local iree_trace_thread_t _thread = {0};

void iree_tracing_set_thread_name(const char* name) {
  _thread.name_length = iree_min(strlen(name), IREE_ARRAYSIZE(_thread.name));
  memcpy(_thread.name, name, _thread.name_length);
}

typedef struct {
  const char* value;
  int length;
} iree_trace_file_name_t;
static iree_trace_file_name_t iree_tracing_trim_file_path(
    const char* file_name, size_t file_name_length) {
  for (int i = (int)file_name_length - 1; i >= 0; --i) {
    char c = file_name[i];
    if (c == '/' || c == '\\') {
      return (iree_trace_file_name_t){file_name + i + 1,
                                      file_name_length - i - 1};
    }
  }
  return (iree_trace_file_name_t){file_name, file_name_length};
}

static iree_zone_id_t iree_tracing_zone_begin(
    iree_trace_file_name_t file_name, uint32_t line, const char* function_name,
    size_t function_name_length, const char* name, size_t name_length) {
  // Push the zone onto the zone stack.
  assert(_thread.depth + 1 < IREE_ARRAYSIZE(_thread.stack));
  iree_zone_id_t zone_id = ++_thread.depth;

  // Use the function name for display if no override was provided.
  if (!name) {
    name = function_name;
    name_length = function_name_length;
  }

  // Log the line immediately.
  fprintf(_console.file, "[%08" PRIX64 "][%.*s%*s] %*s► %.*s (%.*s:%u)\n",
          iree_thread_id(), (int)_thread.name_length, _thread.name,
          (int)(IREE_TRACING_MAX_THREAD_LENGTH - _thread.name_length), "",
          (zone_id - 1), "", (int)name_length, name, file_name.length,
          file_name.value, line);
#if IREE_TRACING_CONSOLE_FLUSH
  fflush(_console.file);
#endif  // IREE_TRACING_CONSOLE_FLUSH

#if IREE_TRACING_CONSOLE_TIMING
  // When timing copy the name so we can print it later and capture the start
  // time.
  iree_trace_zone_t* zone = &_thread.stack[zone_id];
  zone->name_length = iree_min(name_length, IREE_ARRAYSIZE(zone->name));
  memcpy(zone->name, name, name_length);
  zone->start_timestamp_ns = iree_time_now_ns();
#endif  // IREE_TRACING_CONSOLE_TIMING

  return zone_id;
}

IREE_MUST_USE_RESULT iree_zone_id_t
iree_tracing_zone_begin_impl(const iree_tracing_location_t* src_loc,
                             const char* name, size_t name_length) {
  return iree_tracing_zone_begin(
      iree_tracing_trim_file_path(src_loc->file_name,
                                  src_loc->file_name_length),
      src_loc->line, src_loc->function_name, src_loc->function_name_length,
      name ? name : src_loc->name,
      name_length ? name_length : src_loc->name_length);
}

IREE_MUST_USE_RESULT iree_zone_id_t iree_tracing_zone_begin_external_impl(
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  return iree_tracing_zone_begin(
      iree_tracing_trim_file_path(file_name, file_name_length), line,
      function_name, function_name_length, name, name_length);
}

void iree_tracing_zone_end(iree_zone_id_t zone_id) {
  if (!zone_id) return;

  assert(_thread.depth > 0);

#if IREE_TRACING_CONSOLE_TIMING
  // Capture timestamp first so that we don't measure too much of ourselves.
  uint64_t end_timestamp_ns = iree_time_now_ns();
  iree_trace_zone_t* zone = &_thread.stack[zone_id];
  uint64_t duration_ns = end_timestamp_ns - zone->start_timestamp_ns;
  fprintf(_console.file,
          "[%08" PRIX64 "][%.*s%*s] %*s◄ %.*s t = %" PRIu64 "ns / %" PRIu64
          "us / %gms\n",
          iree_thread_id(), (int)_thread.name_length, _thread.name,
          (int)(IREE_TRACING_MAX_THREAD_LENGTH - _thread.name_length), "",
          (zone_id - 1), "", (int)zone->name_length, zone->name, duration_ns,
          duration_ns / 1000, duration_ns / 1000000.0f);
#if IREE_TRACING_CONSOLE_FLUSH
  fflush(_console.file);
#endif  // IREE_TRACING_CONSOLE_FLUSH
#endif  // IREE_TRACING_CONSOLE_TIMING

  --_thread.depth;
}

void iree_tracing_message_cstring(const char* value, const char* symbol,
                                  uint32_t color) {
  iree_tracing_message_string_view(value, strlen(value), symbol, color);
}

void iree_tracing_message_string_view(const char* value, size_t value_length,
                                      const char* symbol, uint32_t color) {
  fprintf(_console.file, "[%08" PRIX64 "][%.*s%*s] %*s %s %.*s\n",
          iree_thread_id(), (int)_thread.name_length, _thread.name,
          (int)(IREE_TRACING_MAX_THREAD_LENGTH - _thread.name_length), "",
          _thread.depth, "", symbol, (int)value_length, value);
#if IREE_TRACING_CONSOLE_FLUSH
  fflush(_console.file);
#endif  // IREE_TRACING_CONSOLE_FLUSH
}

void iree_tracing_memory_alloc(const char* name, size_t name_length, void* ptr,
                               size_t size) {
  fprintf(_console.file, "[%08" PRIX64 "][%.*s%*s] %*s● %.*s alloc %p (%zu)\n",
          iree_thread_id(), (int)_thread.name_length, _thread.name,
          (int)(IREE_TRACING_MAX_THREAD_LENGTH - _thread.name_length), "",
          _thread.depth, "", (int)name_length, name, ptr, size);
#if IREE_TRACING_CONSOLE_FLUSH
  fflush(_console.file);
#endif  // IREE_TRACING_CONSOLE_FLUSH
}

void iree_tracing_memory_free(const char* name, size_t name_length, void* ptr) {
  fprintf(_console.file, "[%08" PRIX64 "][%.*s%*s] %*s◌ %.*s free %p\n",
          iree_thread_id(), (int)_thread.name_length, _thread.name,
          (int)(IREE_TRACING_MAX_THREAD_LENGTH - _thread.name_length), "",
          _thread.depth, "", (int)name_length, name, ptr);
#if IREE_TRACING_CONSOLE_FLUSH
  fflush(_console.file);
#endif  // IREE_TRACING_CONSOLE_FLUSH
}

#endif  // IREE_TRACING_FEATURES
