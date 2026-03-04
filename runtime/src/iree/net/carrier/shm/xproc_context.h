// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Release context for cross-process SHM carriers.
//
// Each cross-process carrier owns one of these. It holds the resources
// specific to the cross-process setup: our mapping of the carrier SHM region,
// our mapping of the peer's wake epoch SHM, the peer notification proxy, and
// the owned copy of the peer's signal primitive. All are cleaned up when the
// carrier is destroyed (last reference released).
//
// The ref count supports the case where both carrier endpoints are in the same
// process (e.g., during testing). In the normal cross-process case, each side
// has its own independent xproc_context with ref_count = 1.

#ifndef IREE_NET_CARRIER_SHM_XPROC_CONTEXT_H_
#define IREE_NET_CARRIER_SHM_XPROC_CONTEXT_H_

#include "iree/async/notification.h"
#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/shm.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cross-process carrier release context. Passed as release_context to
// iree_net_shm_carrier_create().
typedef struct iree_net_shm_xproc_context_t {
  iree_atomic_ref_count_t ref_count;

  // Our mapping of the carrier SHM region (rings + armed flags + epochs).
  iree_shm_mapping_t shm_mapping;

  // Our mapping of the peer's shared_wake epoch SHM page.
  // This is the same physical page the peer's shared_wake notification uses
  // as its epoch counter. Our peer_notification's epoch_ptr points here.
  iree_shm_mapping_t peer_wake_epoch_mapping;

  // Proxy notification for waking the peer's proactor. Created locally using
  // iree_async_notification_create_shared() with the peer's epoch SHM and
  // signal primitive. Incrementing the epoch is visible to the peer (same
  // physical page); writing to the signal primitive wakes their proactor.
  iree_async_notification_t* peer_notification;

  // Owned duplicate of the peer's signal primitive (received via handshake).
  // The peer_notification references but does not own this (SHARED flag).
  // We close it here on destroy.
  iree_async_primitive_t peer_signal_primitive;

  iree_allocator_t allocator;
} iree_net_shm_xproc_context_t;

// Creates an xproc context. All fields except ref_count and allocator must
// be populated by the caller after creation.
static inline iree_status_t iree_net_shm_xproc_context_create(
    iree_allocator_t allocator, iree_net_shm_xproc_context_t** out_context) {
  iree_net_shm_xproc_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*context), (void**)&context));
  memset(context, 0, sizeof(*context));
  iree_atomic_ref_count_init(&context->ref_count);
  context->allocator = allocator;
  *out_context = context;
  return iree_ok_status();
}

// Retains a reference to the xproc context.
static inline void iree_net_shm_xproc_context_retain(
    iree_net_shm_xproc_context_t* context) {
  iree_atomic_ref_count_inc(&context->ref_count);
}

// Releases a reference. Destroys when the reference count reaches zero:
// unmaps both SHM regions, releases the proxy notification, and closes the
// owned peer signal primitive.
static inline void iree_net_shm_xproc_context_release(void* opaque) {
  iree_net_shm_xproc_context_t* context = (iree_net_shm_xproc_context_t*)opaque;
  if (iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    iree_async_notification_release(context->peer_notification);
    iree_async_primitive_close(&context->peer_signal_primitive);
    iree_shm_close(&context->peer_wake_epoch_mapping);
    iree_shm_close(&context->shm_mapping);
    iree_allocator_t allocator = context->allocator;
    iree_allocator_free(allocator, context);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_XPROC_CONTEXT_H_
