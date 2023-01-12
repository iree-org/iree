// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/nccl_channel.h"

#include <iree/base/config.h>
#include <iree/base/status.h>
#include <iree/hal/command_buffer.h>
#include <iree/hal/utils/collective_batch.h>
#if IREE_HAL_CUDA_NCCL_ENABLE
#include <nccl.h>
#endif
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda/cuda_buffer.h"
#include "iree/hal/drivers/cuda/status_util.h"

#if IREE_HAL_CUDA_NCCL_ENABLE

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

  ncclComm_t comm = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;  // FIXME: use async to check a timeout
  iree_status_t status = NCCL_RESULT_TO_STATUS(
      context_wrapper->syms,
      ncclCommInitRankConfig(&comm, count, *((const ncclUniqueId*)id), rank,
                             &config));
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_cuda_nccl_channel_t* channel = NULL;
  status = iree_allocator_malloc(context_wrapper->host_allocator,
                                 sizeof(*channel), (void**)&channel);
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

  // TODO(#9580): support async tear down
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
  NCCL_IGNORE_ERROR(channel->context_wrapper->syms,
                    ncclCommFinalize(channel->comm));
  NCCL_IGNORE_ERROR(channel->context_wrapper->syms,
                    ncclCommDestroy(channel->comm));
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

// Returns the NCCL communicator for the given |channel|, if available.
static ncclComm_t iree_hal_cuda_nccl_channel_comm(
    iree_hal_channel_t* base_channel) {
  IREE_ASSERT_ARGUMENT(base_channel);
  iree_hal_cuda_nccl_channel_t* channel =
      iree_hal_cuda_nccl_channel_cast(base_channel);
  return channel->comm;
}

static iree_status_t get_nccl_data_type(iree_hal_collective_element_type_t in,
                                        ncclDataType_t* out) {
  switch (in) {
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_8:
      *out = ncclInt8;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_8:
      *out = ncclUint8;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_16:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "SINT16 is not supported for collective op");
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_16:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "UINT16 is not supported for collective op");
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_32:
      *out = ncclInt32;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_32:
      *out = ncclUint32;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_64:
      *out = ncclInt64;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_UINT_64:
      *out = ncclUint64;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_16:
      *out = ncclFloat16;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_32:
      *out = ncclFloat32;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_FLOAT_64:
      *out = ncclFloat64;
      break;
    case IREE_HAL_COLLECTIVE_ELEMENT_TYPE_BFLOAT_16:
      *out = ncclFloat64;
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled element type for collective op");
  }
  return iree_ok_status();
}

static iree_status_t get_nccl_red_type(iree_hal_collective_reduction_t in,
                                       ncclRedOp_t* out) {
  switch (in) {
    case IREE_HAL_COLLECTIVE_REDUCTION_SUM:
      *out = ncclSum;
      break;
    case IREE_HAL_COLLECTIVE_REDUCTION_PRODUCT:
      *out = ncclProd;
      break;
    case IREE_HAL_COLLECTIVE_REDUCTION_MINIMUM:
      *out = ncclMin;
      break;
    case IREE_HAL_COLLECTIVE_REDUCTION_MAXIMUM:
      *out = ncclMax;
      break;
    case IREE_HAL_COLLECTIVE_REDUCTION_AVERAGE:
      *out = ncclAvg;
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled reduction type for collective op");
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_nccl_submit_batch_entry(
    const iree_hal_collective_batch_entry_t* entry, CUstream stream) {
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT_ARGUMENT(stream);

  iree_hal_cuda_nccl_channel_t* channel =
      iree_hal_cuda_nccl_channel_cast(entry->channel);
  iree_hal_cuda_dynamic_symbols_t* syms = channel->context_wrapper->syms;
  ncclComm_t comm = iree_hal_cuda_nccl_channel_comm(entry->channel);
  ncclDataType_t datatype;
  IREE_RETURN_IF_ERROR(get_nccl_data_type(entry->op.element_type, &datatype));

  switch (entry->op.kind) {
    case IREE_HAL_COLLECTIVE_KIND_ALL_GATHER: {
      CUdeviceptr sendbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      NCCL_RETURN_IF_ERROR(
          syms,
          ncclAllGather((const void*)sendbuff, (void*)recvbuff,
                        entry->element_count, datatype, comm, stream),
          "ncclAllGather");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_ALL_REDUCE: {
      CUdeviceptr sendbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      ncclRedOp_t redop;
      IREE_RETURN_IF_ERROR(get_nccl_red_type(entry->op.reduction, &redop));
      NCCL_RETURN_IF_ERROR(
          syms,
          ncclAllReduce((const void*)sendbuff, (void*)recvbuff,
                        entry->element_count, datatype, redop, comm, stream),
          "ncclAllReduce");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_BROADCAST: {
      CUdeviceptr sendbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      NCCL_RETURN_IF_ERROR(syms,
                           ncclBroadcast((const void*)sendbuff, (void*)recvbuff,
                                         entry->element_count, datatype,
                                         entry->param, comm, stream),
                           "ncclBroadcast");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_REDUCE: {
      CUdeviceptr sendbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      ncclRedOp_t redop;
      IREE_RETURN_IF_ERROR(get_nccl_red_type(entry->op.reduction, &redop));
      NCCL_RETURN_IF_ERROR(syms,
                           ncclReduce((const void*)sendbuff, (void*)recvbuff,
                                      entry->element_count, datatype, redop,
                                      entry->param, comm, stream),
                           "ncclReduce");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_REDUCE_SCATTER: {
      CUdeviceptr sendbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      ncclRedOp_t redop;
      IREE_RETURN_IF_ERROR(get_nccl_red_type(entry->op.reduction, &redop));
      NCCL_RETURN_IF_ERROR(
          syms,
          ncclReduceScatter((const void*)sendbuff, (void*)recvbuff,
                            entry->element_count, datatype, redop, comm,
                            stream),
          "ncclReduceScatter");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_SEND: {
      CUdeviceptr sendbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      NCCL_RETURN_IF_ERROR(syms,
                           ncclSend((const void*)sendbuff, entry->element_count,
                                    datatype, entry->param, comm, stream),
                           "ncclSend");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_RECV: {
      CUdeviceptr recvbuff =
          iree_hal_cuda_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      NCCL_RETURN_IF_ERROR(syms,
                           ncclRecv((void*)recvbuff, entry->element_count,
                                    datatype, entry->param, comm, stream),
                           "ncclRecv");
      break;
    }
  }  // switch
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_nccl_submit_batch(
    iree_hal_cuda_context_wrapper_t* context,
    const iree_hal_collective_batch_t* batch, CUstream stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(stream);
  NCCL_RETURN_IF_ERROR(context->syms, ncclGroupStart(), "ncclGroupStart");
  for (IREE_HOST_SIZE_T i = 0; i < batch->count; ++i) {
    iree_hal_cuda_nccl_submit_batch_entry(&batch->entries[i], stream);
  }
  return NCCL_RESULT_TO_STATUS(context->syms, ncclGroupEnd(), "ncclGroupEnd");
}

static const iree_hal_channel_vtable_t iree_hal_cuda_nccl_channel_vtable = {
    .destroy = iree_hal_cuda_nccl_channel_destroy,
    .query_rank_and_count = iree_hal_cuda_nccl_channel_query_rank_and_count,
};

#else  // IREE_HAL_CUDA_NCCL_ENABLE

iree_status_t iree_hal_cuda_nccl_channel_create(
    iree_hal_cuda_context_wrapper_t* context_wrapper,
    const iree_hal_cuda_nccl_id_t* id, int rank, int count,
    iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "iree_hal_cuda_nccl_channel_create()");
}

iree_status_t iree_hal_cuda_nccl_submit_batch(
    iree_hal_cuda_context_wrapper_t* context,
    const iree_hal_collective_batch_t* batch, CUstream stream) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "iree_hal_cuda_nccl_submit_batch()");
}

#endif  // IREE_HAL_CUDA_NCCL_ENABLE
