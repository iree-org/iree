// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_cache.h"

#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

//===----------------------------------------------------------------------===//
// Freelist entry
//===----------------------------------------------------------------------===//

// A cached slab in the freelist, ready for immediate acquisition.
typedef struct iree_hal_slab_cache_entry_t {
  // Slab returned from the inner provider.
  iree_hal_slab_t slab;

  // Timestamp (nanoseconds) when this entry was pushed to the freelist.
  // Used for adaptive retention: entries idle too long are released.
  iree_time_t return_time;

  // Intrusive linkage for the atomic slist.
  iree_atomic_slist_intrusive_ptr_t slist_next;
} iree_hal_slab_cache_entry_t;

IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_slab_cache_entry,
                                iree_hal_slab_cache_entry_t,
                                offsetof(iree_hal_slab_cache_entry_t,
                                         slist_next));

//===----------------------------------------------------------------------===//
// Cache state
//===----------------------------------------------------------------------===//

typedef struct iree_hal_slab_cache_t {
  // Base slab-provider interface.
  iree_hal_slab_provider_t base;

  // Retained provider that performs real slab allocation and wrapping.
  iree_hal_slab_provider_t* inner_provider;

  // Host allocator used for cache bookkeeping.
  iree_allocator_t host_allocator;

  // Uniform size of slabs prepared and retained by the cache.
  iree_device_size_t slab_size;

  // Number of ready slabs the background thread tries to maintain.
  uint32_t target_count;

  // Maximum number of ready slabs retained by release_slab().
  uint32_t max_count;

  // Lock-free freelist of pre-acquired, pre-faulted slabs.
  iree_hal_slab_cache_entry_slist_t freelist;

  // Current number of entries in the freelist. Relaxed atomic; approximate
  // count for the background thread's refill decisions.
  iree_atomic_int32_t ready_count;

  // Signaled when ready_count drops below target_count or on shutdown.
  // The background thread waits on this.
  iree_notification_t wake_notification;

  // Signaled when a new slab is added to the freelist (by the background
  // thread) or when the thread encounters an error. Threads blocked in
  // acquire_slab wait on this.
  iree_notification_t ready_notification;

  // Set to true to request the background thread to exit.
  iree_atomic_int32_t shutdown;

  // Error from the last failed refill attempt, stored as an atomic intptr_t
  // (iree_status_t is a pointer). The background thread stores errors here
  // via exchange and stops refilling until the error is consumed by an
  // acquire_slab caller (also via exchange). This propagates the inner
  // provider's backtrace to the caller that needs it; the user decides
  // whether to retry (unload a model, shrink pools) or give up.
  iree_atomic_intptr_t pending_error;

  // The background thread that pre-acquires and pre-faults slabs.
  // NULL if thread creation failed during _create (partial init state).
  iree_thread_t* thread;

  // Number of acquire_slab() calls satisfied from the freelist.
  iree_atomic_int64_t cache_hit_count;

  // Number of acquire_slab() calls that had to wait for the refill thread.
  iree_atomic_int64_t cache_miss_count;

  // Cumulative nanoseconds spent prefaulting slabs on the refill thread.
  iree_atomic_int64_t prefault_time_nanoseconds;

  // Exponential moving average of slab reuse interval in nanoseconds.
  iree_atomic_int64_t ema_reuse_interval_nanoseconds;
} iree_hal_slab_cache_t;

static const iree_hal_slab_provider_vtable_t iree_hal_slab_cache_vtable;

//===----------------------------------------------------------------------===//
// Background thread
//===----------------------------------------------------------------------===//

// Acquires and prefaults one slab, pushes it to the freelist.
// Returns the status from the inner provider on failure.
static iree_status_t iree_hal_slab_cache_refill_one(
    iree_hal_slab_cache_t* cache) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "refill_one");

  iree_hal_slab_cache_entry_t* entry = NULL;
  iree_status_t status = iree_allocator_malloc(cache->host_allocator,
                                               sizeof(*entry), (void**)&entry);
  if (iree_status_is_ok(status)) {
    memset(entry, 0, sizeof(*entry));
    status = iree_hal_slab_provider_acquire_slab(
        cache->inner_provider, cache->slab_size, &entry->slab);
  }
  if (iree_status_is_ok(status)) {
    iree_time_t prefault_start = iree_time_now();
    iree_hal_slab_provider_prefault(cache->inner_provider, &entry->slab);
    iree_atomic_fetch_add(&cache->prefault_time_nanoseconds,
                          (int64_t)(iree_time_now() - prefault_start),
                          iree_memory_order_relaxed);

    entry->return_time = iree_time_now();
    iree_hal_slab_cache_entry_slist_push(&cache->freelist, entry);
    iree_atomic_fetch_add(&cache->ready_count, 1, iree_memory_order_release);
    iree_notification_post(&cache->ready_notification, IREE_ALL_WAITERS);
  } else {
    // Entry allocation succeeded but slab acquisition failed; free entry.
    // If entry allocation itself failed, entry is NULL and this is safe.
    iree_allocator_free(cache->host_allocator, entry);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Returns true if the thread has a pending error that hasn't been consumed.
static bool iree_hal_slab_cache_has_pending_error(
    iree_hal_slab_cache_t* cache) {
  return iree_atomic_load(&cache->pending_error, iree_memory_order_acquire) !=
         0;
}

// Publishes |status| as the pending cache error if no error is already pending.
// Takes ownership of |status| in all cases. First error wins: if a different
// caller already published an error, this releases the later status.
static void iree_hal_slab_cache_publish_error(iree_hal_slab_cache_t* cache,
                                              iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status));
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &cache->pending_error, &expected, (intptr_t)status,
          iree_memory_order_release, iree_memory_order_relaxed)) {
    iree_status_free(status);
    return;
  }

  // Wake waiters so they can consume the error instead of blocking.
  iree_notification_post(&cache->ready_notification, IREE_ALL_WAITERS);
}

// Predicate for iree_notification_await: returns true when the thread should
// wake (ready_count below target, no pending error, or shutdown requested).
static bool iree_hal_slab_cache_should_wake(void* arg) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)arg;
  if (iree_atomic_load(&cache->shutdown, iree_memory_order_acquire)) {
    return true;
  }
  if (iree_hal_slab_cache_has_pending_error(cache)) {
    return false;  // Don't wake while an error is pending.
  }
  return iree_atomic_load(&cache->ready_count, iree_memory_order_relaxed) <
         (int32_t)cache->target_count;
}

static int iree_hal_slab_cache_thread_main(void* arg) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)arg;

  while (!iree_atomic_load(&cache->shutdown, iree_memory_order_acquire)) {
    iree_notification_await(&cache->wake_notification,
                            iree_hal_slab_cache_should_wake, cache,
                            iree_infinite_timeout());
    if (iree_atomic_load(&cache->shutdown, iree_memory_order_acquire)) {
      break;
    }

    // Refill the freelist up to target_count.
    while (!iree_atomic_load(&cache->shutdown, iree_memory_order_acquire) &&
           !iree_hal_slab_cache_has_pending_error(cache) &&
           iree_atomic_load(&cache->ready_count, iree_memory_order_relaxed) <
               (int32_t)cache->target_count) {
      iree_status_t status = iree_hal_slab_cache_refill_one(cache);
      if (!iree_status_is_ok(status)) {
        // Store the error for the next acquire_slab caller to consume. The
        // thread stops refilling until the error is consumed.
        iree_hal_slab_cache_publish_error(cache, status);
        break;
      }
    }
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

static void iree_hal_slab_cache_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_provider;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal the background thread to exit and wait for it.
  if (cache->thread) {
    iree_atomic_store(&cache->shutdown, 1, iree_memory_order_release);
    iree_notification_post(&cache->wake_notification, IREE_ALL_WAITERS);
    iree_thread_release(cache->thread);
  }

  // Flush all cached slabs back to the inner provider.
  iree_hal_slab_cache_entry_t* entry = NULL;
  while ((entry = iree_hal_slab_cache_entry_slist_pop(&cache->freelist)) !=
         NULL) {
    iree_hal_slab_provider_release_slab(cache->inner_provider, &entry->slab);
    iree_allocator_free(cache->host_allocator, entry);
  }

  // Release any pending error that was never picked up by a caller. The cache
  // is being torn down and there is no caller left to receive it.
  iree_status_t pending_error = (iree_status_t)iree_atomic_exchange(
      &cache->pending_error, 0, iree_memory_order_acquire);
  iree_status_free(pending_error);

  iree_hal_slab_cache_entry_slist_deinitialize(&cache->freelist);
  iree_notification_deinitialize(&cache->ready_notification);
  iree_notification_deinitialize(&cache->wake_notification);
  iree_hal_slab_provider_release(cache->inner_provider);

  iree_allocator_t host_allocator = cache->host_allocator;
  iree_allocator_free(host_allocator, cache);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_slab_cache_create(
    iree_hal_slab_cache_options_t options,
    iree_hal_slab_provider_t* inner_provider,
    iree_thread_affinity_t thread_affinity, iree_allocator_t host_allocator,
    iree_hal_slab_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(inner_provider);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;

  if (options.slab_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "slab_size must be > 0");
  }
  if (options.target_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target_count must be > 0");
  }
  if (options.max_count < options.target_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "max_count (%" PRIu32
                            ") must be >= target_count (%" PRIu32 ")",
                            options.max_count, options.target_count);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_slab_cache_t* cache = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache);
  if (iree_status_is_ok(status)) {
    memset(cache, 0, sizeof(*cache));
    iree_hal_slab_provider_initialize(&iree_hal_slab_cache_vtable,
                                      &cache->base);
    iree_hal_slab_provider_retain(inner_provider);
    cache->inner_provider = inner_provider;
    cache->host_allocator = host_allocator;
    cache->slab_size = options.slab_size;
    cache->target_count = options.target_count;
    cache->max_count = options.max_count;
    iree_atomic_store(&cache->pending_error, 0, iree_memory_order_relaxed);

    iree_hal_slab_cache_entry_slist_initialize(&cache->freelist);
    iree_notification_initialize(&cache->wake_notification);
    iree_notification_initialize(&cache->ready_notification);
  }
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t thread_params = {
        .name = IREE_SV("slab_cache"),
        .priority_class = IREE_THREAD_PRIORITY_CLASS_LOW,
        .initial_affinity = thread_affinity,
    };
    status = iree_thread_create(iree_hal_slab_cache_thread_main, cache,
                                thread_params, host_allocator, &cache->thread);
  }
  if (iree_status_is_ok(status)) {
    *out_provider = &cache->base;
  } else if (cache) {
    iree_hal_slab_cache_destroy(&cache->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Predicate for acquire_slab's await: returns true when there's something to
// do (a slab is available or an error is pending).
static bool iree_hal_slab_cache_acquire_ready(void* arg) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)arg;
  return iree_atomic_load(&cache->ready_count, iree_memory_order_relaxed) > 0 ||
         iree_hal_slab_cache_has_pending_error(cache);
}

// Pops an entry from the freelist, transfers the slab to |out_slab|, frees
// the entry struct, and updates counters. Returns true if an entry was
// available.
static bool iree_hal_slab_cache_pop_entry(iree_hal_slab_cache_t* cache,
                                          iree_hal_slab_t* out_slab) {
  iree_hal_slab_cache_entry_t* entry =
      iree_hal_slab_cache_entry_slist_pop(&cache->freelist);
  if (!entry) {
    return false;
  }
  *out_slab = entry->slab;

  // Update adaptive retention EMA: track how quickly slabs are reused.
  // EMA with alpha = 1/8: new = old * 7/8 + sample * 1/8.
  iree_time_t reuse_interval = iree_time_now() - entry->return_time;
  int64_t old_ema = iree_atomic_load(&cache->ema_reuse_interval_nanoseconds,
                                     iree_memory_order_relaxed);
  int64_t new_ema = (old_ema * 7 + (int64_t)reuse_interval) / 8;
  iree_atomic_store(&cache->ema_reuse_interval_nanoseconds, new_ema,
                    iree_memory_order_relaxed);

  iree_allocator_free(cache->host_allocator, entry);
  iree_atomic_fetch_add(&cache->ready_count, -1, iree_memory_order_relaxed);
  return true;
}

// Takes the pending error from the cache, clearing the slot so the
// background thread can resume refilling. Returns iree_ok_status() if no
// error was pending.
static iree_status_t iree_hal_slab_cache_consume_error(
    iree_hal_slab_cache_t* cache) {
  return (iree_status_t)iree_atomic_exchange(&cache->pending_error, 0,
                                             iree_memory_order_acq_rel);
}

//===----------------------------------------------------------------------===//
// Slab provider vtable
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_slab_cache_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_provider;
  memset(out_slab, 0, sizeof(*out_slab));

  // Oversized requests bypass the cache entirely.
  if (min_length > cache->slab_size) {
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "oversized_bypass");
    iree_status_t status = iree_hal_slab_provider_acquire_slab(
        cache->inner_provider, min_length, out_slab);
    if (iree_status_is_ok(status)) {
      iree_hal_slab_provider_prefault(cache->inner_provider, out_slab);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Deliver refill errors before cached slabs. Pending errors stop the refill
  // thread; consuming them promptly lets the next caller retry instead of
  // silently draining the remaining cache and leaving refill parked.
  iree_status_t status = iree_hal_slab_cache_consume_error(cache);
  if (!iree_status_is_ok(status)) return status;

  // Fast path: pop from the freelist.
  if (iree_hal_slab_cache_pop_entry(cache, out_slab)) {
    iree_atomic_fetch_add(&cache->cache_hit_count, 1,
                          iree_memory_order_relaxed);
    // Signal the background thread to refill if below target.
    if (iree_atomic_load(&cache->ready_count, iree_memory_order_relaxed) <
        (int32_t)cache->target_count) {
      iree_notification_post(&cache->wake_notification, IREE_ALL_WAITERS);
    }
    return iree_ok_status();
  }

  iree_atomic_fetch_add(&cache->cache_miss_count, 1, iree_memory_order_relaxed);
  for (;;) {
    // Slow path: freelist empty. Signal the background thread and wait for
    // either a slab to appear or an error to be posted. The condition predicate
    // ensures we only wake when there's something actionable, but another
    // acquire_slab() caller may consume that slab/error before this thread
    // resumes. Keep looping until this caller owns a slab or a real error.
    iree_notification_post(&cache->wake_notification, IREE_ALL_WAITERS);
    iree_notification_await(&cache->ready_notification,
                            iree_hal_slab_cache_acquire_ready, cache,
                            iree_infinite_timeout());

    status = iree_hal_slab_cache_consume_error(cache);
    if (!iree_status_is_ok(status)) return status;

    if (iree_hal_slab_cache_pop_entry(cache, out_slab)) {
      return iree_ok_status();
    }
  }
}

static void iree_hal_slab_cache_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_provider;

  // Oversized slabs were not cached; return directly to inner provider.
  if (slab->length != cache->slab_size) {
    iree_hal_slab_provider_release_slab(cache->inner_provider, slab);
    return;
  }

  // Over max capacity; release to inner provider.
  if (iree_atomic_load(&cache->ready_count, iree_memory_order_relaxed) >=
      (int32_t)cache->max_count) {
    iree_hal_slab_provider_release_slab(cache->inner_provider, slab);
    return;
  }

  // Return to the freelist for reuse.
  iree_hal_slab_cache_entry_t* entry = NULL;
  iree_status_t status = iree_allocator_malloc(cache->host_allocator,
                                               sizeof(*entry), (void**)&entry);
  if (!iree_status_is_ok(status)) {
    // Entry allocation failed; store the error for the next acquire. If an
    // error is already pending, first error wins and this status is released.
    iree_hal_slab_cache_publish_error(cache, status);
    iree_hal_slab_provider_release_slab(cache->inner_provider, slab);
    return;
  }
  memset(entry, 0, sizeof(*entry));
  entry->slab = *slab;
  entry->return_time = iree_time_now();
  iree_hal_slab_cache_entry_slist_push(&cache->freelist, entry);
  iree_atomic_fetch_add(&cache->ready_count, 1, iree_memory_order_relaxed);
  // Wake threads blocked in acquire_slab's slow path.
  iree_notification_post(&cache->ready_notification, IREE_ALL_WAITERS);
}

static iree_status_t iree_hal_slab_cache_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_provider;
  return iree_hal_slab_provider_wrap_buffer(
      cache->inner_provider, slab, slab_offset, allocation_size, params,
      release_callback, out_buffer);
}

static void iree_hal_slab_cache_prefault(
    iree_hal_slab_provider_t* base_provider, iree_hal_slab_t* slab) {
  // Slabs from the cache are already pre-faulted by the background thread.
  // Oversized slabs that bypassed the cache were pre-faulted in acquire_slab.
}

static void iree_hal_slab_cache_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  const iree_hal_slab_cache_t* cache =
      (const iree_hal_slab_cache_t*)base_provider;
  iree_hal_slab_provider_query_properties(cache->inner_provider,
                                          out_memory_type, out_supported_usage);
}

static void iree_hal_slab_cache_trim(
    iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_trim_flags_t flags) {
  iree_hal_slab_cache_t* cache = (iree_hal_slab_cache_t*)base_provider;
  IREE_TRACE_ZONE_BEGIN(z0);

  int32_t target = (flags & IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_ALL)
                       ? 0
                       : (int32_t)cache->target_count;

  iree_hal_slab_cache_entry_t* entry = NULL;
  while (iree_atomic_load(&cache->ready_count, iree_memory_order_relaxed) >
         target) {
    entry = iree_hal_slab_cache_entry_slist_pop(&cache->freelist);
    if (!entry) {
      break;
    }
    iree_hal_slab_provider_release_slab(cache->inner_provider, &entry->slab);
    iree_allocator_free(cache->host_allocator, entry);
    iree_atomic_fetch_add(&cache->ready_count, -1, iree_memory_order_relaxed);
  }

  // Trim the inner provider as well.
  iree_hal_slab_provider_trim(cache->inner_provider, flags);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_slab_cache_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  const iree_hal_slab_cache_t* cache =
      (const iree_hal_slab_cache_t*)base_provider;
  if (iree_hal_slab_provider_visited(visited, base_provider)) {
    return;
  }

  // Recurse into the inner provider first.
  iree_hal_slab_provider_query_stats(cache->inner_provider, visited, out_stats);

  // Add cache-layer contributions.
  out_stats->cache.count = (uint32_t)iree_atomic_load(
      &cache->ready_count, iree_memory_order_relaxed);
  out_stats->cache.hit_count += (uint64_t)iree_atomic_load(
      &cache->cache_hit_count, iree_memory_order_relaxed);
  out_stats->cache.miss_count += (uint64_t)iree_atomic_load(
      &cache->cache_miss_count, iree_memory_order_relaxed);
  out_stats->cache.ema_reuse_interval_nanoseconds = (uint64_t)iree_atomic_load(
      &cache->ema_reuse_interval_nanoseconds, iree_memory_order_relaxed);
  out_stats->cache.prefault_time_nanoseconds += (uint64_t)iree_atomic_load(
      &cache->prefault_time_nanoseconds, iree_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_slab_provider_vtable_t iree_hal_slab_cache_vtable = {
    .destroy = iree_hal_slab_cache_destroy,
    .acquire_slab = iree_hal_slab_cache_acquire_slab,
    .release_slab = iree_hal_slab_cache_release_slab,
    .wrap_buffer = iree_hal_slab_cache_wrap_buffer,
    .prefault = iree_hal_slab_cache_prefault,
    .trim = iree_hal_slab_cache_trim,
    .query_stats = iree_hal_slab_cache_query_stats,
    .query_properties = iree_hal_slab_cache_query_properties,
};
