// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda2/nccl_channel.h"

#include <stddef.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda2/cuda_buffer.h"
#include "iree/hal/drivers/cuda2/cuda_status_util.h"
#include "iree/hal/drivers/cuda2/nccl_status_util.h"

typedef struct iree_hal_cuda2_nccl_channel_t {
  iree_hal_resource_t resource;

  const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols;
  const iree_hal_cuda2_nccl_dynamic_symbols_t* nccl_symbols;

  iree_allocator_t host_allocator;

  // Parent channel this was split from, if any.
  // This is only used to keep the parent channel live for as long as there are
  // any split channels live (including transitive splits).
  iree_hal_channel_t* parent_channel;

  // This participant's rank in the communicator.
  // Equivalent to ncclCommUserRank.
  int rank;
  // Total number of participants in the communicator.
  // Equivalent to ncclCommCount.
  int count;

  // Communicator handle.
  ncclComm_t comm;

  // Hash of the unique ID used to create the communicator.
  // This is consistent with the hashes NCCL itself uses for logging but is not
  // guaranteed to be unique - only use for informational purposes.
  IREE_TRACE(uint64_t id_hash;)
} iree_hal_cuda2_nccl_channel_t;

static const iree_hal_channel_vtable_t iree_hal_cuda2_nccl_channel_vtable;

static iree_hal_cuda2_nccl_channel_t* iree_hal_cuda2_nccl_channel_cast(
    iree_hal_channel_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_nccl_channel_vtable);
  return (iree_hal_cuda2_nccl_channel_t*)base_value;
}

static const iree_hal_cuda2_nccl_channel_t*
iree_hal_cuda2_nccl_channel_const_cast(const iree_hal_channel_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_nccl_channel_vtable);
  return (const iree_hal_cuda2_nccl_channel_t*)base_value;
}

// Returns the same value as NCCL's init.cc hashUniqueId.
// These magic constants were chosen by their implementation and unlikely to
// be stable as it's not part of their public API. So they are only meant to be
// used for correlating debug logging/traces. We keep it internal here too so
// that we aren't tempted to use it in other places.
static uint64_t iree_hal_cuda2_nccl_hash_id(
    const iree_hal_cuda2_nccl_id_t* id) {
  uint64_t hash = 0xDEADBEEF;
  for (iree_host_size_t i = 0; i < sizeof(*id); i++) {
    hash ^= hash >> 32;
    hash *= 0x8DB3DB47FA2994ADull;
    hash += id->data[i];
  }
  return hash;
}

iree_status_t iree_hal_cuda2_nccl_get_unique_id(
    const iree_hal_cuda2_nccl_dynamic_symbols_t* symbols,
    iree_hal_cuda2_nccl_id_t* out_id) {
  static_assert(sizeof(*out_id) == sizeof(ncclUniqueId),
                "NCCL ID size mismatch");

  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_id);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_id, 0, sizeof(*out_id));
  iree_status_t status = IREE_NCCL_RESULT_TO_STATUS(
      symbols, ncclGetUniqueId((ncclUniqueId*)out_id), "ncclGetUniqueId");

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_cuda2_nccl_channel_create(
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols,
    const iree_hal_cuda2_nccl_dynamic_symbols_t* nccl_symbols,
    const iree_hal_cuda2_nccl_id_t* id, int rank, int count,
    iree_allocator_t host_allocator, iree_hal_channel_t** out_channel) {
  IREE_ASSERT_ARGUMENT(cuda_symbols);
  IREE_ASSERT_ARGUMENT(nccl_symbols);
  IREE_ASSERT_ARGUMENT(id);
  IREE_ASSERT_ARGUMENT(out_channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_channel = NULL;
  IREE_TRACE(const uint64_t id_hash = iree_hal_cuda2_nccl_hash_id(id));
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, id_hash);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, rank);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);

  ncclComm_t comm = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  // TODO: use async to check a timeout.
  config.blocking = 1;
  IREE_NCCL_RETURN_AND_END_ZONE_IF_ERROR(
      z0, nccl_symbols,
      ncclCommInitRankConfig(&comm, count, *((const ncclUniqueId*)id), rank,
                             &config),
      "ncclCommInitRankConfig");

  iree_hal_cuda2_nccl_channel_t* channel = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*channel),
                                (void**)&channel));

  iree_hal_resource_initialize(&iree_hal_cuda2_nccl_channel_vtable,
                               &channel->resource);
  channel->cuda_symbols = cuda_symbols;
  channel->nccl_symbols = nccl_symbols;
  channel->host_allocator = host_allocator;
  channel->parent_channel = NULL;
  channel->rank = rank;
  channel->count = count;
  channel->comm = comm;
  IREE_TRACE(channel->id_hash = id_hash);
  *out_channel = (iree_hal_channel_t*)channel;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda2_nccl_channel_destroy(
    iree_hal_channel_t* base_channel) {
  iree_hal_cuda2_nccl_channel_t* channel =
      iree_hal_cuda2_nccl_channel_cast(base_channel);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, channel->id_hash);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, channel->rank);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, channel->count);

  iree_allocator_t host_allocator = channel->host_allocator;

  // TODO(#9580): support async tear down
  // We could be smarter about starting finalization of all channels async and
  // then waiting for them to complete but we aren't currently optimizing for
  // lifetime performance. To do that we'd probably want to track each open
  // channel on the device that created them and manage teardown there.
  //
  // Recommended:
  //   ncclCommFinalize(channel->comm);  // non-blocking!
  //   while (ncclCommGetAsyncError == ncclInProgress) sleep(1);
  //   ncclCommDestroy(channel->comm)
  // Should work the same (as we are doing a blocking teardown):
  //   ncclCommDestroy(channel->comm)
  IREE_NCCL_IGNORE_ERROR(channel->nccl_symbols,
                         ncclCommFinalize(channel->comm));

  IREE_NCCL_IGNORE_ERROR(channel->nccl_symbols, ncclCommDestroy(channel->comm));

  iree_hal_channel_release(channel->parent_channel);
  iree_allocator_free(host_allocator, channel);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda2_nccl_channel_split(
    iree_hal_channel_t* base_channel, int32_t color, int32_t key,
    iree_hal_channel_flags_t flags, iree_hal_channel_t** out_split_channel) {
  iree_hal_cuda2_nccl_channel_t* channel =
      iree_hal_cuda2_nccl_channel_cast(base_channel);

  // TODO: see if we need to set the sharing config - we may always want to.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  // TODO: use async to check a timeout.
  config.blocking = 1;

  // Split the communicator.
  ncclComm_t split_comm = NULL;
  IREE_NCCL_RETURN_IF_ERROR(
      channel->nccl_symbols,
      ncclCommSplit(channel->comm, color, key, &split_comm, &config),
      "ncclCommSplit");

  // Query the local rank/count from the split communicator.
  int split_rank = 0;
  int split_count = 0;
  iree_status_t status = IREE_NCCL_RESULT_TO_STATUS(
      channel->nccl_symbols, ncclCommUserRank(split_comm, &split_rank),
      "ncclCommUserRank");
  if (iree_status_is_ok(status)) {
    status = IREE_NCCL_RESULT_TO_STATUS(channel->nccl_symbols,
                                        ncclCommCount(split_comm, &split_count),
                                        "ncclCommCount");
  }

  // Wrap the split communicator in a new channel.
  iree_hal_cuda2_nccl_channel_t* split_channel = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(channel->host_allocator, sizeof(*split_channel),
                              (void**)&split_channel);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda2_nccl_channel_vtable,
                                 &split_channel->resource);
    split_channel->cuda_symbols = channel->cuda_symbols;
    split_channel->nccl_symbols = channel->nccl_symbols;
    split_channel->host_allocator = channel->host_allocator;
    split_channel->parent_channel = base_channel;
    iree_hal_channel_retain(base_channel);
    split_channel->rank = split_rank;
    split_channel->count = split_count;
    split_channel->comm = split_comm;
    *out_split_channel = (iree_hal_channel_t*)split_channel;
  }

  if (!iree_status_is_ok(status)) {
    IREE_NCCL_IGNORE_ERROR(channel->nccl_symbols, ncclCommDestroy(split_comm));
  }
  return status;
}

static void iree_hal_cuda2_nccl_channel_query_rank_and_count(
    const iree_hal_channel_t* base_channel, int32_t* out_rank,
    int32_t* out_count) {
  IREE_ASSERT_ARGUMENT(base_channel);
  IREE_ASSERT_ARGUMENT(out_count);
  const iree_hal_cuda2_nccl_channel_t* channel =
      iree_hal_cuda2_nccl_channel_const_cast(base_channel);
  // NOTE: since it's cheap we keep rank/count local - this lets us trace them
  // out without needing to call into NCCL each time.
  *out_rank = channel->rank;
  *out_count = channel->count;
}

// Returns the NCCL communicator for the given |channel|, if available.
static ncclComm_t iree_hal_cuda2_nccl_channel_comm(
    iree_hal_channel_t* base_channel) {
  IREE_ASSERT_ARGUMENT(base_channel);
  iree_hal_cuda2_nccl_channel_t* channel =
      iree_hal_cuda2_nccl_channel_cast(base_channel);
  return channel->comm;
}

static iree_status_t iree_hal_cuda2_get_nccl_data_type(
    iree_hal_collective_element_type_t in, ncclDataType_t* out) {
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

static iree_status_t iree_hal_cuda2_get_nccl_reduction_type(
    iree_hal_collective_reduction_t in, ncclRedOp_t* out) {
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

static iree_status_t iree_hal_cuda2_nccl_submit_batch_entry(
    const iree_hal_collective_batch_entry_t* entry, CUstream stream) {
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT_ARGUMENT(stream);

  iree_hal_cuda2_nccl_channel_t* channel =
      iree_hal_cuda2_nccl_channel_cast(entry->channel);
  const iree_hal_cuda2_nccl_dynamic_symbols_t* symbols = channel->nccl_symbols;
  ncclComm_t comm = iree_hal_cuda2_nccl_channel_comm(entry->channel);

  ncclDataType_t datatype;
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda2_get_nccl_data_type(entry->op.element_type, &datatype));

  switch (entry->op.kind) {
    case IREE_HAL_COLLECTIVE_KIND_ALL_GATHER: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      IREE_NCCL_RETURN_IF_ERROR(
          symbols,
          ncclAllGather((const void*)sendbuff, (void*)recvbuff,
                        entry->element_count, datatype, comm, stream),
          "ncclAllGather");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_ALL_REDUCE: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      ncclRedOp_t redop;
      IREE_RETURN_IF_ERROR(
          iree_hal_cuda2_get_nccl_reduction_type(entry->op.reduction, &redop));
      IREE_NCCL_RETURN_IF_ERROR(
          symbols,
          ncclAllReduce((const void*)sendbuff, (void*)recvbuff,
                        entry->element_count, datatype, redop, comm, stream),
          "ncclAllReduce");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_ALL_TO_ALL: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      iree_device_size_t send_count = entry->element_count / channel->count;
      iree_device_size_t element_size_bytes =
          iree_hal_collective_element_byte_count(entry->op.element_type);
      iree_device_size_t rank_offset = send_count * element_size_bytes;
      // These calls are already grouped by iree_hal_cuda2_nccl_submit_batch.
      for (iree_host_size_t r = 0; r < channel->count; ++r) {
        IREE_NCCL_RETURN_IF_ERROR(
            symbols,
            ncclSend((const void*)(sendbuff + r * rank_offset), send_count,
                     datatype, r, comm, stream),
            "ncclSend");
        IREE_NCCL_RETURN_IF_ERROR(
            symbols,
            ncclRecv((void*)(recvbuff + r * rank_offset), send_count, datatype,
                     r, comm, stream),
            "ncclRecv");
      }
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_BROADCAST: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      IREE_NCCL_RETURN_IF_ERROR(
          symbols,
          ncclBroadcast((const void*)sendbuff, (void*)recvbuff,
                        entry->element_count, datatype, entry->param, comm,
                        stream),
          "ncclBroadcast");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_REDUCE: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      ncclRedOp_t redop;
      IREE_RETURN_IF_ERROR(
          iree_hal_cuda2_get_nccl_reduction_type(entry->op.reduction, &redop));
      IREE_NCCL_RETURN_IF_ERROR(
          symbols,
          ncclReduce((const void*)sendbuff, (void*)recvbuff,
                     entry->element_count, datatype, redop, entry->param, comm,
                     stream),
          "ncclReduce");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_REDUCE_SCATTER: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      ncclRedOp_t redop;
      IREE_RETURN_IF_ERROR(
          iree_hal_cuda2_get_nccl_reduction_type(entry->op.reduction, &redop));
      IREE_NCCL_RETURN_IF_ERROR(
          symbols,
          ncclReduceScatter((const void*)sendbuff, (void*)recvbuff,
                            entry->element_count, datatype, redop, comm,
                            stream),
          "ncclReduceScatter");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_SEND: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      IREE_NCCL_RETURN_IF_ERROR(
          symbols,
          ncclSend((const void*)sendbuff, entry->element_count, datatype,
                   entry->param, comm, stream),
          "ncclSend");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_RECV: {
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      IREE_NCCL_RETURN_IF_ERROR(symbols,
                                ncclRecv((void*)recvbuff, entry->element_count,
                                         datatype, entry->param, comm, stream),
                                "ncclRecv");
      break;
    }
    case IREE_HAL_COLLECTIVE_KIND_SEND_RECV: {
      CUdeviceptr sendbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->send_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->send_binding.buffer) +
          entry->send_binding.offset;
      CUdeviceptr recvbuff =
          iree_hal_cuda2_buffer_device_pointer(
              iree_hal_buffer_allocated_buffer(entry->recv_binding.buffer)) +
          iree_hal_buffer_byte_offset(entry->recv_binding.buffer) +
          entry->recv_binding.offset;
      int16_t sendid;
      int16_t recvid;
      memcpy(&sendid, &entry->param, 2);
      memcpy(&recvid, (char*)&entry->param + 2, 2);
      if (sendid != -1) {
        IREE_NCCL_RETURN_IF_ERROR(
            symbols,
            ncclSend((const void*)sendbuff, entry->element_count, datatype,
                     sendid, comm, stream),
            "ncclSend");
      }
      if (recvid != -1) {
        IREE_NCCL_RETURN_IF_ERROR(
            symbols,
            ncclRecv((void*)recvbuff, entry->element_count, datatype, recvid,
                     comm, stream),
            "ncclRecv");
      } else {
        // Zero out recvbuff if this rank is not receiving any data.
        iree_device_size_t num_bytes =
            entry->element_count *
            iree_hal_collective_element_byte_count(entry->op.element_type);
        IREE_CUDA_RETURN_IF_ERROR(
            channel->cuda_symbols,
            cuMemsetD8Async(recvbuff, 0, num_bytes, stream), "cuMemsetD8Async");
      }
      break;
    }
  }  // switch
  return iree_ok_status();
}

iree_status_t iree_hal_cuda2_nccl_submit_batch(
    const iree_hal_cuda2_nccl_dynamic_symbols_t* symbols,
    iree_hal_cuda2_tracing_context_t* tracing_context,
    const iree_hal_collective_batch_t* batch, CUstream stream) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(batch);
  IREE_ASSERT_ARGUMENT(stream);

  // Begin one zone for each entry in the batch. Each entry will show stacked on
  // top of each other and unfortunately use independent CUDA events. We could
  // optimize this by changing the tracing context to expose an API with event
  // reservation and then zone commit using an existing event.
  IREE_TRACE({
    iree_bitfield_string_temp_t string_temp;
    for (iree_host_size_t i = 0; i < batch->count; ++i) {
      iree_hal_collective_batch_entry_t* entry = &batch->entries[i];
      iree_string_view_t collective_str =
          iree_hal_collective_op_format(&entry->op, &string_temp);
      IREE_CUDA_TRACE_ZONE_BEGIN_EXTERNAL(
          tracing_context, stream, __FILE__, strlen(__FILE__),
          (uint32_t)__LINE__, __FUNCTION__, strlen(__FUNCTION__),
          collective_str.data, collective_str.size);
    }
  });

  // Issue all collective operations in the batch as part of a group.
  // NCCL may be able to fuse or reduce overheads by issuing like this.
  IREE_NCCL_RETURN_IF_ERROR(symbols, ncclGroupStart(), "ncclGroupStart");
  for (iree_host_size_t i = 0; i < batch->count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_cuda2_nccl_submit_batch_entry(&batch->entries[i], stream));
  }
  IREE_NCCL_RETURN_IF_ERROR(symbols, ncclGroupEnd(), "ncclGroupEnd");

  // End all zones we began above - note that these are just simply nested so
  // order doesn't matter so long as we end the right number of zones.
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < batch->count; ++i) {
      IREE_CUDA_TRACE_ZONE_END(tracing_context, stream);
    }
  });

  return iree_ok_status();
}

static const iree_hal_channel_vtable_t iree_hal_cuda2_nccl_channel_vtable = {
    .destroy = iree_hal_cuda2_nccl_channel_destroy,
    .split = iree_hal_cuda2_nccl_channel_split,
    .query_rank_and_count = iree_hal_cuda2_nccl_channel_query_rank_and_count,
};
