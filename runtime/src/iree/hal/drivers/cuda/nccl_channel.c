// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/nccl_channel.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

// Returns the same value as NCCL's init.cc hashUniqueId.
// These magic constants were chosen by their implementation and unlikely to
// be stable as it's not part of their public API. Only to be used for
// correlating debug logging/traces. We keep it internal here too so that we
// aren't tempted to use it either.
static uint64_t iree_hal_cuda_nccl_hash_id(const iree_hal_cuda_nccl_id_t* id) {
  uint64_t hash = 0xDEADBEEF;
  for (iree_host_size_t i = 0; i < sizeof(*id); i++) {
    hash ^= hash >> 32;
    hash *= 0x8DB3DB47FA2994ADull;
    hash += id->data[i];
  }
  return hash;
}

typedef struct iree_hal_cuda_nccl_channel_t {
  iree_hal_resource_t resource;
  iree_hal_cuda_context_wrapper_t* context_wrapper;

  // Hash of the unique ID used to create the communicator.
  // This is consistent with the hashes NCCL itself uses for logging but is not
  // guaranteed to be unique - only use for informational purposes.
  uint64_t id_hash;

  // This participant's rank in the communicator.
  // Equivalent to ncclCommUserRank.
  int rank;
  // Total number of participants in the communicator.
  // Equivalent to ncclCommCount.
  int count;

  // Communicator handle.
  ncclComm_t comm;
} iree_hal_cuda_nccl_channel_t;

static const iree_hal_channel_vtable_t iree_hal_cuda_nccl_channel_vtable;

static iree_hal_cuda_nccl_channel_t* iree_hal_cuda_nccl_channel_cast(
    iree_hal_channel_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_nccl_channel_vtable);
  return (iree_hal_cuda_nccl_channel_t*)base_value;
}

iree_status_t iree_hal_cuda_nccl_channel_create(
    iree_hal_cuda_context_wrapper_t* context_wrapper,
    const iree_hal_cuda_nccl_id_t* id, int rank, int count,
    iree_hal_channel_t** out_channel) {
  IREE_ASSERT_ARGUMENT(context_wrapper);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  const uint64_t id_hash = iree_hal_cuda_nccl_hash_id(id);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, id_hash);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, rank);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, count);

  // TODO(#9580): actually use nccl to create a communicator.
  // Something like:
  //  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  //  config.blocking = 0;
  //  syms->ncclCommInitRankConfig(&comm, count, *id, rank, &config);
  // NOTE: CHECK ERRORS! we can safely return here as we haven't allocated the
  // channel wrapper yet.
  ncclComm_t comm = NULL;
  if (!comm) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "failed to create NCCL communicator for rank=%d count=%d", rank, count);
  }

  iree_hal_cuda_nccl_channel_t* channel = NULL;
  iree_status_t status = iree_allocator_malloc(
      context_wrapper->host_allocator, sizeof(*channel), (void**)&channel);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_nccl_channel_vtable,
                                 &channel->resource);
    channel->context_wrapper = context_wrapper;
    channel->id_hash = id_hash;
    channel->rank = rank;
    channel->count = count;
    channel->comm = comm;
    *out_channel = (iree_hal_channel_t*)channel;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_nccl_channel_destroy(
    iree_hal_channel_t* base_channel) {
  iree_hal_cuda_nccl_channel_t* channel =
      iree_hal_cuda_nccl_channel_cast(base_channel);
  iree_allocator_t host_allocator = channel->context_wrapper->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, channel->id_hash);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, channel->rank);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, channel->count);

  // TODO(#9580): tear down nccl - blocking if needed.
  // We could be smarter about starting finalization of all channels async and
  // then waiting for them to complete but we aren't currently optimizing for
  // lifetime performance. To do that we'd probably want to track each open
  // channel on the device that created them and manage teardown there.
  //
  // Recommended:
  //  syms->ncclCommFinalize(channel->comm);  // non-blocking!
  //  while (ncclCommGetAsyncError == ncclInProgress) sleep(1);
  //  syms->ncclCommDestroy(channel->comm)
  // Should work the same (as we are doing a blocking teardown):
  //  syms->ncclCommDestroy(channel->comm)

  iree_allocator_free(host_allocator, channel);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_cuda_nccl_channel_query_rank_and_count(
    const iree_hal_channel_t* base_channel, int32_t* out_rank,
    int32_t* out_count) {
  IREE_ASSERT_ARGUMENT(base_channel);
  iree_hal_cuda_nccl_channel_t* channel =
      iree_hal_cuda_nccl_channel_cast((iree_hal_channel_t*)base_channel);
  // NOTE: since it's cheap we keep rank/count local - this lets us trace them
  // out without needing to call into NCCL each time.
  *out_rank = channel->rank;
  *out_count = channel->count;
}

ncclComm_t iree_hal_cuda_nccl_channel_comm(iree_hal_channel_t* base_channel) {
  IREE_ASSERT_ARGUMENT(base_channel);
  iree_hal_cuda_nccl_channel_t* channel =
      iree_hal_cuda_nccl_channel_cast(base_channel);
  return channel->comm;
}

iree_status_t iree_hal_cuda_nccl_submit_batch(
    iree_hal_cuda_context_wrapper_t* context,
    const iree_hal_collective_batch_t* batch, CUstream stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(stream);

  // TODO(#9580): issue the operations in the batch. Note that the channel may
  // change between ops and the communicator should be retrieved from each.
  //
  // Something like:
  //  make context->cu_context active (for when using multiple devices)
  //  syms->ncclGroupStart();
  //  for each entry in batch:
  //    ncclComm_t comm = iree_hal_cuda_nccl_channel_comm(entry->channel);
  //    syms->nccl*(comm, ...);
  //  syms->ncclGroupEnd();

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "NCCL submission not yet implemented");
}

static const iree_hal_channel_vtable_t iree_hal_cuda_nccl_channel_vtable = {
    .destroy = iree_hal_cuda_nccl_channel_destroy,
    .query_rank_and_count = iree_hal_cuda_nccl_channel_query_rank_and_count,
};
