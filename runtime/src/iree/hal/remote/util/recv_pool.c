// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/util/recv_pool.h"

#include "iree/async/affinity.h"
#include "iree/async/proactor.h"
#include "iree/async/slab.h"

struct iree_hal_remote_recv_pool_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  iree_async_proactor_pool_t* proactor_pool;  // retained (keeps thread alive)
  iree_async_proactor_t* proactor;            // retained
  iree_async_slab_t* slab;
  iree_async_region_t* region;
  iree_async_buffer_pool_t* buffer_pool;
};

static void iree_hal_remote_recv_pool_destroy(
    iree_hal_remote_recv_pool_t* recv_pool) {
  iree_allocator_t host_allocator = recv_pool->host_allocator;
  iree_async_buffer_pool_free(recv_pool->buffer_pool);
  iree_async_region_release(recv_pool->region);
  iree_async_slab_release(recv_pool->slab);
  iree_async_proactor_release(recv_pool->proactor);
  iree_async_proactor_pool_release(recv_pool->proactor_pool);
  iree_allocator_free(host_allocator, recv_pool);
}

iree_status_t iree_hal_remote_recv_pool_create(
    iree_async_proactor_pool_t* proactor_pool, uint32_t numa_node_id,
    iree_allocator_t host_allocator,
    iree_hal_remote_recv_pool_t** out_recv_pool) {
  IREE_ASSERT_ARGUMENT(proactor_pool);
  IREE_ASSERT_ARGUMENT(out_recv_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_recv_pool = NULL;

  iree_hal_remote_recv_pool_t* recv_pool = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*recv_pool), (void**)&recv_pool);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_atomic_ref_count_init(&recv_pool->ref_count);
  recv_pool->host_allocator = host_allocator;

  // Retain the proactor pool to keep its threads alive. The pool's proactor
  // thread polls the io_uring/epoll ring — without it, no I/O completes.
  recv_pool->proactor_pool = proactor_pool;
  iree_async_proactor_pool_retain(proactor_pool);

  // Select the proactor for this NUMA node and retain it.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_get_for_node(proactor_pool, numa_node_id,
                                                   &recv_pool->proactor);
    if (iree_status_is_ok(status)) {
      iree_async_proactor_retain(recv_pool->proactor);
    }
  }

  // Allocate the slab with NUMA affinity.
  if (iree_status_is_ok(status)) {
    iree_async_affinity_t affinity = iree_async_affinity_any();
    affinity.numa_node = numa_node_id;
    iree_async_slab_options_t slab_options = {
        .buffer_size = 4096,
        .buffer_count = 32,
        .affinity = &affinity,
    };
    status =
        iree_async_slab_create(slab_options, host_allocator, &recv_pool->slab);
  }

  // Try WRITE access first (enables kernel-managed recv buffer selection).
  // Falls back to NONE if WRITE registration fails (e.g., io_uring PBUF_RING
  // races with the poll thread on some configurations). The io_uring
  // SOCKET_RECV_POOL path has a userspace fallback for NONE-registered pools.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_slab(
        recv_pool->proactor, recv_pool->slab,
        IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &recv_pool->region);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      status = iree_async_proactor_register_slab(
          recv_pool->proactor, recv_pool->slab,
          IREE_ASYNC_BUFFER_ACCESS_FLAG_NONE, &recv_pool->region);
    }
  }

  // Create the lock-free buffer pool over the registered region.
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_allocate(recv_pool->region, host_allocator,
                                             &recv_pool->buffer_pool);
  }

  if (iree_status_is_ok(status)) {
    *out_recv_pool = recv_pool;
  } else {
    iree_hal_remote_recv_pool_destroy(recv_pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_recv_pool_wrap(
    iree_async_proactor_t* proactor, iree_async_slab_t* slab,
    iree_async_region_t* region, iree_async_buffer_pool_t* buffer_pool,
    iree_allocator_t host_allocator,
    iree_hal_remote_recv_pool_t** out_recv_pool) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(buffer_pool);
  IREE_ASSERT_ARGUMENT(out_recv_pool);
  *out_recv_pool = NULL;

  iree_hal_remote_recv_pool_t* recv_pool = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*recv_pool),
                                             (void**)&recv_pool));
  iree_atomic_ref_count_init(&recv_pool->ref_count);
  recv_pool->host_allocator = host_allocator;
  recv_pool->proactor = proactor;
  iree_async_proactor_retain(proactor);
  recv_pool->slab = slab;
  iree_async_slab_retain(slab);
  recv_pool->region = region;
  iree_async_region_retain(region);
  recv_pool->buffer_pool = buffer_pool;

  *out_recv_pool = recv_pool;
  return iree_ok_status();
}

void iree_hal_remote_recv_pool_retain(iree_hal_remote_recv_pool_t* recv_pool) {
  if (recv_pool) {
    iree_atomic_ref_count_inc(&recv_pool->ref_count);
  }
}

void iree_hal_remote_recv_pool_release(iree_hal_remote_recv_pool_t* recv_pool) {
  if (recv_pool && iree_atomic_ref_count_dec(&recv_pool->ref_count) == 1) {
    iree_hal_remote_recv_pool_destroy(recv_pool);
  }
}

iree_async_proactor_t* iree_hal_remote_recv_pool_proactor(
    iree_hal_remote_recv_pool_t* recv_pool) {
  return recv_pool->proactor;
}

iree_async_buffer_pool_t* iree_hal_remote_recv_pool_buffer_pool(
    iree_hal_remote_recv_pool_t* recv_pool) {
  return recv_pool->buffer_pool;
}
