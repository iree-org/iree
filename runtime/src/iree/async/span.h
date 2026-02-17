// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Value-type subrange of a registered memory region.
//
// An iree_async_span_t identifies a contiguous byte range within an
// iree_async_region_t. Spans are non-owning (like iree_string_view_t): the
// caller ensures the referenced region remains valid for the span's lifetime.
//
// Operations that transfer data (recv, send, read, write) take spans to
// describe their buffers. The proactor uses the span's region to access
// backend-specific handles for zero-copy I/O.
//
// ## Region lifetime during I/O
//
// When a span is embedded in an operation and submitted to a proactor, the
// proactor retains the span's region for the duration of the operation. This
// ensures the region (and its backend handles) remain valid even if the caller
// releases the buffer registration before the operation completes.
//
// Proactor implementations call iree_async_span_retain_region() at submit time
// and iree_async_span_release_region() after the final completion callback
// fires. This is transparent to callers: they construct spans, submit
// operations, and the proactor manages the lifetime window.
//
// For spans with region == NULL (unregistered memory), no retain/release
// occurs. The caller must ensure the memory remains valid for the operation's
// lifetime.

#ifndef IREE_ASYNC_SPAN_H_
#define IREE_ASYNC_SPAN_H_

#include "iree/async/region.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Span
//===----------------------------------------------------------------------===//

// A non-owning reference to a contiguous byte range.
//
// When region is non-NULL, the span references a subrange of a registered
// memory region (offset is relative to region->base_ptr). The proactor can
// use the region's backend handles for zero-copy I/O.
//
// When region is NULL, the span references unregistered memory (offset holds
// the raw pointer cast to iree_host_size_t). The proactor falls back to
// copy-based I/O. The caller must ensure the memory remains valid for the
// span's lifetime.
typedef struct iree_async_span_t {
  // The region this span references. NULL for unregistered (raw pointer) spans.
  iree_async_region_t* region;

  // When region is non-NULL: byte offset from region->base_ptr.
  // When region is NULL: the raw pointer cast to iree_host_size_t.
  iree_host_size_t offset;

  // Byte length of the span.
  iree_host_size_t length;
} iree_async_span_t;

// Creates a span referencing a subrange of a region.
static inline iree_async_span_t iree_async_span_make(
    iree_async_region_t* region, iree_host_size_t offset,
    iree_host_size_t length) {
  iree_async_span_t span;
  span.region = region;
  span.offset = offset;
  span.length = length;
  return span;
}

// Creates a span covering an entire region.
static inline iree_async_span_t iree_async_span_from_region(
    iree_async_region_t* region, iree_host_size_t region_length) {
  iree_async_span_t span;
  span.region = region;
  span.offset = 0;
  span.length = region_length;
  return span;
}

// Creates a span from a raw (unregistered) pointer.
// The caller must ensure the memory remains valid for the span's lifetime.
// Proactors use copy-based I/O for raw pointer spans (no zero-copy).
static inline iree_async_span_t iree_async_span_from_ptr(
    void* ptr, iree_host_size_t length) {
  iree_async_span_t span;
  span.region = NULL;
  span.offset = (iree_host_size_t)(uintptr_t)ptr;
  span.length = length;
  return span;
}

// Returns an empty span.
static inline iree_async_span_t iree_async_span_empty(void) {
  iree_async_span_t span = {0};
  return span;
}

// Returns true if the span has zero length.
static inline bool iree_async_span_is_empty(iree_async_span_t span) {
  return span.length == 0;
}

// Returns true if the span references CPU-accessible memory.
// For raw pointer spans (region == NULL), returns true (caller provided ptr).
// For region spans, returns true if region->base_ptr is non-NULL.
//
// CPU-inaccessible spans (device-only dma-buf) can only be used with carriers
// that support DEVICE_MEMORY_TX/RX, and cannot be used with codecs that
// require content inspection (compression, encryption).
static inline bool iree_async_span_is_cpu_accessible(iree_async_span_t span) {
  // Raw pointer spans are always CPU-accessible (caller gave us a pointer).
  if (!span.region) return true;
  // Region spans are CPU-accessible if base_ptr is non-NULL.
  return span.region->base_ptr != NULL;
}

// Returns the host pointer for the start of the span.
// For registered spans, returns region->base_ptr + offset.
// For raw pointer spans (region == NULL), returns the pointer stored in offset.
static inline uint8_t* iree_async_span_ptr(iree_async_span_t span) {
  if (!span.region) return (uint8_t*)(uintptr_t)span.offset;
  return (uint8_t*)span.region->base_ptr + span.offset;
}

// Returns the span's memory as an iree_byte_span_t (mutable).
static inline iree_byte_span_t iree_async_span_data(iree_async_span_t span) {
  return iree_make_byte_span(iree_async_span_ptr(span), span.length);
}

// Returns the span's memory as an iree_const_byte_span_t (read-only).
static inline iree_const_byte_span_t iree_async_span_const_data(
    iree_async_span_t span) {
  return iree_make_const_byte_span(iree_async_span_ptr(span), span.length);
}

//===----------------------------------------------------------------------===//
// Span list
//===----------------------------------------------------------------------===//

// A scatter-gather list of spans for vectored I/O operations.
typedef struct iree_async_span_list_t {
  iree_async_span_t* values;
  iree_host_size_t count;
} iree_async_span_list_t;

// Creates a span list from a pointer to an array of spans and a count.
// The |values| array must remain valid for the span list's lifetime.
static inline iree_async_span_list_t iree_async_span_list_make(
    iree_async_span_t* values, iree_host_size_t count) {
  iree_async_span_list_t list;
  list.values = values;
  list.count = count;
  return list;
}

// Returns an empty span list with zero entries.
static inline iree_async_span_list_t iree_async_span_list_empty(void) {
  iree_async_span_list_t list = {0};
  return list;
}

// Returns true if the span list contains no entries.
static inline bool iree_async_span_list_is_empty(iree_async_span_list_t list) {
  return list.count == 0;
}

//===----------------------------------------------------------------------===//
// Region lifetime helpers (proactor-internal)
//===----------------------------------------------------------------------===//

// Retains the region referenced by a span (if non-NULL).
// Called by proactor implementations at submit time to ensure the region
// remains valid for the duration of the operation.
static inline void iree_async_span_retain_region(iree_async_span_t span) {
  if (span.region) iree_async_region_retain(span.region);
}

// Releases the region referenced by a span. No-op if the region is NULL.
// Called by proactor implementations after the final completion callback.
static inline void iree_async_span_release_region(iree_async_span_t span) {
  iree_async_region_release(span.region);
}

// Retains all regions in a span list.
static inline void iree_async_span_list_retain_regions(
    iree_async_span_list_t list) {
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    iree_async_span_retain_region(list.values[i]);
  }
}

// Releases all regions in a span list.
static inline void iree_async_span_list_release_regions(
    iree_async_span_list_t list) {
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    iree_async_span_release_region(list.values[i]);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_SPAN_H_
