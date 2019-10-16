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

// Utilities for profiling and tracing.
// These attempt to support the various tools we use in a way that scales better
// than one annotation per tool per site and ensures things stay consistent and
// easy to correlate across tools.
//
// Tracing with WTF:
// - build with --define=GLOBAL_WTF_ENABLE=1
// - pass --iree_trace_file=/tmp/foo.wtf-trace when running
// - view trace in WTF UI
//
// If GLOBAL_WTF_ENABLE=1 is specified WTF will automatically be initialized on
// startup and flushed on exit.

#ifndef IREE_BASE_TRACING_H_
#define IREE_BASE_TRACING_H_

#if defined(WTF_ENABLE) || defined(IREE_CONFIG_GOOGLE_INTERNAL)

#include "wtf/event.h"   // IWYU pragma: export
#include "wtf/macros.h"  // IWYU pragma: export

namespace iree {

// Initializes tracing if it is built into the binary.
// Does nothing if already initialized.
void InitializeTracing();

// Stops tracing and flushes any pending data.
void StopTracing();

// Flushes pending trace data to disk, if enabled.
void FlushTrace();

// Enables the current thread for WTF profiling/tracing.
#define IREE_TRACE_THREAD_ENABLE(name) WTF_THREAD_ENABLE(name);

// Tracing scope that emits WTF tracing scopes depending on whether
// profiling/tracing are enabled.
// See WTF_SCOPE0 for more information.
#define IREE_TRACE_SCOPE0(name_spec) WTF_SCOPE0(name_spec);

// Tracing scope that emits WTF tracing scopes with additional
// arguments depending on whether profiling/tracing is enabled.
// See WTF_SCOPE for more information.
#define IREE_TRACE_SCOPE(name_spec, ...) WTF_SCOPE(name_spec, __VA_ARGS__)

// Tracing event that emits a WTF event.
// See WTF_EVENT0 for more information.
#define IREE_TRACE_EVENT0 WTF_EVENT0

// Tracing event that emits a WTF event with additional arguments.
// See WTF_EVENT for more information.
#define IREE_TRACE_EVENT WTF_EVENT

}  // namespace iree

#else

namespace iree {

inline void InitializeTracing() {}
inline void StopTracing() {}
inline void FlushTrace() {}

#define IREE_TRACE_THREAD_ENABLE(name)
#define IREE_TRACE_SCOPE0(name_spec)
#define IREE_TRACE_SCOPE(name_spec, ...) (void)
#define IREE_TRACE_EVENT0
#define IREE_TRACE_EVENT (void)

}  // namespace iree

#endif  // GLOBAL_WTF_ENABLE

#endif  // IREE_BASE_TRACING_H_
