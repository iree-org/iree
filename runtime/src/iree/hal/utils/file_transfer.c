// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/file_transfer.h"

#include "iree/base/internal/math.h"
#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// TODO(benvanik): make these either compile-time configuration options so we
// can prune code paths or flags (somehow).

#if !defined(IREE_HAL_TRANSFER_WORKER_LIMIT)
// Maximum number of workers that will be used. This is something we can derive
// from the transfer size and the loop; small transfers or synchronous loops
// should have 1 and we can measure to see how many others we need.
#define IREE_HAL_TRANSFER_WORKER_LIMIT 1
#endif  // !IREE_HAL_TRANSFER_WORKER_LIMIT

#if !defined(IREE_HAL_TRANSFER_CHUNK_SIZE)
// Bytes per worker to stage chunks of data. Larger chunks will result in less
// overhead as fewer copy operations are required.
#define IREE_HAL_TRANSFER_CHUNK_SIZE (64 * 1024 * 1024)
#endif  // !IREE_HAL_TRANSFER_CHUNK_SIZE

#if !defined(IREE_HAL_TRANSFER_CHUNKS_PER_WORKER)
// Estimated number of chunks each worker should process used to determine how
// many workers are needed as part of a transfer operation. Larger numbers will
// reduce memory overhead at the cost of latency reductions.
#define IREE_HAL_TRANSFER_CHUNKS_PER_WORKER 8
#endif  // IREE_HAL_TRANSFER_CHUNKS_PER_WORKER

//===----------------------------------------------------------------------===//
// iree_hal_transfer_operation_t
//===----------------------------------------------------------------------===//

// TODO(benvanik): move to utils/ without relying on iree_hal_memory_file_t.

// Maximum number of transfer workers that can be used; common usage should be
// 1-4 but on very large systems with lots of bandwidth we may be able to
// use more.
#define IREE_HAL_TRANSFER_WORKER_MAX_COUNT 64

// Each bit represents one worker within a transfer matching its ordinal in
// the operation workers array.
typedef uint64_t iree_hal_transfer_worker_bitmask_t;

// Counts the total number of workers indicated by the given worker bitmask.
#define iree_hal_transfer_worker_live_count(bitmask) \
  iree_math_count_ones_u64(bitmask)

// Describes the direction of a transfer operation.
typedef enum {
  // Transferring from the file to the buffer (read).
  IREE_HAL_TRANSFER_READ_FILE_TO_BUFFER = 0,
  // Transferring from the buffer to the file (write).
  IREE_HAL_TRANSFER_WRITE_BUFFER_TO_FILE,
} iree_hal_transfer_direction_t;

typedef struct iree_hal_transfer_operation_t iree_hal_transfer_operation_t;

// A worker greedily processing subranges of a larger transfer operation.
// Since transfers are 99% IO bound we avoid real threads and use workers as
// coroutines (or something like them): workers submit operations and schedule
// an async wait on a loop for when the operation completes on the device - when
// woken the worker will try to grab another subrange of the transfer and
// continue running. When there are no remaining subranges the workers will
// exit and when the last does the transfer is marked complete.
typedef struct iree_hal_transfer_worker_t {
  // Parent operation this worker is a part of.
  iree_hal_transfer_operation_t* operation;
  // Used to associate tracing events with this worker.
  IREE_TRACE(int32_t trace_id;)
  // Aligned offset into the staging buffer of the worker storage.
  iree_device_size_t staging_buffer_offset;
  // Aligned length of the staging buffer storage reserved for the worker.
  iree_device_size_t staging_buffer_length;
  // Semaphore representing the timeline of the worker. The payload is a
  // monotonically increasing operation count.
  iree_hal_semaphore_t* semaphore;
  // Pending timepoint representing an in-flight operation. Upon completion of
  // the operation the semaphore will reach this value.
  uint64_t pending_timepoint;
  // Offset into the transfer operation this worker is currently processing.
  iree_device_size_t pending_transfer_offset;
  // Length of the current worker transfer; usually staging_buffer_length but
  // may be less if this worker is processing the end of the file.
  iree_device_size_t pending_transfer_length;
} iree_hal_transfer_worker_t;

// Manages an asynchronous transfer operation.
typedef struct iree_hal_transfer_operation_t {
  // Some loop implementations are re-entrant and we need to be able to handle
  // the operation completing immediately upon allocation instead of
  // asynchronously and the ref count lets us have the top-level call clean up.
  iree_atomic_ref_count_t ref_count;
  // Device this transfer operation is acting on.
  iree_hal_device_t* device;
  // Queue affinity all operations should be assigned.
  iree_hal_queue_affinity_t queue_affinity;
  // Used to associate tracing events with this worker.
  IREE_TRACE(int32_t trace_id;)

  // Direction of the operation (read file->buffer or write buffer->file).
  iree_hal_transfer_direction_t direction;
  // Retained file resource.
  iree_hal_file_t* file;
  // Offset into the file where the operation begins.
  uint64_t file_offset;
  // Retained buffer resource.
  iree_hal_buffer_t* buffer;
  // Offset into the buffer where the operation begins.
  iree_device_size_t buffer_offset;
  // Total length of the operation.
  iree_device_size_t length;

  // Sticky error status; when any worker fails this will be set to non-OK and
  // when all workers end the staging buffer will be deallocated and the signal
  // semaphores will be marked as failing.
  iree_status_t error_status;
  // Original user semaphores to signal at the end of the transfer operation.
  // Contents are stored at the end of the struct.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Shared staging buffer; contains storage for all workers.
  // We avoid a subspan buffer here to reduce overheads.
  iree_hal_buffer_t* staging_buffer;
  iree_device_size_t staging_buffer_size;

  // Offset to where the transfer head is in the operation.
  // Ranges from 0 at the start and length at the end.
  // Workers use this to consume chunks of the operation.
  iree_device_size_t transfer_head;
  // Total number of chunks remaining in the transfer.
  iree_host_size_t remaining_chunks;

  // Total number of workers participating in the operation.
  iree_host_size_t worker_count;
  // State for each worker in the operation.
  // Stored at the end of the struct.
  iree_hal_transfer_worker_t* workers;
  // One bit per worker indicating whether they are live and ticking the loop.
  // A worker exits by not rescheduling itself and clearing this bit. When all
  // workers have exited the operation has completed.
  // When reading workers exit after enqueuing their final transfer such that
  // the final staging buffer dealloca can be asynchronously chained.
  // When writing workers exit after flushing their final chunk to the file.
  iree_hal_transfer_worker_bitmask_t live_workers;
} iree_hal_transfer_operation_t;

static void iree_hal_transfer_operation_release(
    iree_hal_transfer_operation_t* operation);
static void iree_hal_transfer_operation_destroy(
    iree_hal_transfer_operation_t* operation);

static iree_status_t iree_hal_transfer_operation_create(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_transfer_direction_t direction, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_hal_file_transfer_options_t options,
    iree_hal_transfer_operation_t** out_operation) {
  IREE_ASSERT_ARGUMENT(out_operation);
  *out_operation = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_hal_device_host_allocator(device);

  // Determine how many workers are required and their staging reservation.
  iree_device_size_t worker_chunk_size = options.chunk_size;
  if (worker_chunk_size == IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT) {
    worker_chunk_size = iree_min(IREE_HAL_TRANSFER_CHUNK_SIZE, length);
  }
  iree_device_size_t total_chunk_count =
      iree_device_size_ceil_div(length, worker_chunk_size);
  iree_host_size_t worker_count = options.chunk_count;
  if (worker_count == IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT) {
    // Try to give each worker a couple chunks.
    worker_count = (iree_host_size_t)iree_device_size_ceil_div(
        total_chunk_count, IREE_HAL_TRANSFER_CHUNKS_PER_WORKER);
  }
  worker_count =
      iree_min(worker_count, iree_min(IREE_HAL_TRANSFER_WORKER_LIMIT,
                                      IREE_HAL_TRANSFER_WORKER_MAX_COUNT));

  // Calculate total size of the structure with all its associated data.
  iree_hal_transfer_operation_t* operation = NULL;
  iree_host_size_t total_size = sizeof(*operation);
  iree_host_size_t semaphores_offset =
      iree_host_align(total_size, iree_max_align_t);
  total_size = semaphores_offset + sizeof(signal_semaphore_list.semaphores[0]) *
                                       signal_semaphore_list.count;
  iree_host_size_t payload_values_offset =
      iree_host_align(total_size, iree_max_align_t);
  total_size =
      payload_values_offset + sizeof(signal_semaphore_list.payload_values[0]) *
                                  signal_semaphore_list.count;
  iree_host_size_t worker_offset =
      iree_host_align(total_size, iree_max_align_t);
  total_size = worker_offset + sizeof(operation->workers[0]) * worker_count;

  // Allocate and initialize the struct.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&operation));
  iree_atomic_ref_count_init(&operation->ref_count);
  operation->device = device;
  iree_hal_device_retain(device);
  operation->queue_affinity = queue_affinity;
  operation->direction = direction;
  operation->file = file;
  iree_hal_file_retain(file);
  operation->file_offset = file_offset;
  operation->buffer = buffer;
  iree_hal_buffer_retain(buffer);
  operation->buffer_offset = buffer_offset;
  operation->length = length;
  operation->staging_buffer_size = worker_count * worker_chunk_size;
  operation->transfer_head = 0;
  operation->remaining_chunks = (iree_host_size_t)total_chunk_count;
  operation->worker_count = worker_count;

  // Assign all pointers to the struct suffix storage.
  // We do this first so that if we have to free the struct we have valid
  // pointers.
  operation->signal_semaphore_list.count = signal_semaphore_list.count;
  operation->signal_semaphore_list.semaphores =
      (iree_hal_semaphore_t**)((uintptr_t)operation + semaphores_offset);
  operation->signal_semaphore_list.payload_values =
      (uint64_t*)((uintptr_t)operation + payload_values_offset);
  operation->workers =
      (iree_hal_transfer_worker_t*)((uintptr_t)operation + worker_offset);

  // Assign a unique ID we'll use to make it easier to track what individual
  // steps are part of this transfer.
  IREE_TRACE({
    static iree_atomic_int32_t next_trace_id = IREE_ATOMIC_VAR_INIT(0);
    operation->trace_id =
        iree_atomic_fetch_add(&next_trace_id, 1, iree_memory_order_seq_cst);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation->trace_id);
  });

  // Retain each signal semaphore for ourselves as we don't know if the caller
  // will hold them for the lifetime of the operation.
  memcpy(operation->signal_semaphore_list.semaphores,
         signal_semaphore_list.semaphores,
         sizeof(signal_semaphore_list.semaphores[0]) *
             signal_semaphore_list.count);
  memcpy(operation->signal_semaphore_list.payload_values,
         signal_semaphore_list.payload_values,
         sizeof(signal_semaphore_list.payload_values[0]) *
             signal_semaphore_list.count);
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_semaphore_retain(signal_semaphore_list.semaphores[i]);
  }

  // Initialize all workers.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    iree_hal_transfer_worker_t* worker = &operation->workers[i];
    worker->operation = operation;

    // Assign a unique ID we'll use to make it easier to track what individual
    // steps are part of this worker. It only needs to be unique within the
    // operation.
    IREE_TRACE(worker->trace_id = (int64_t)i);

    // View into the staging buffer where the worker keeps its memory.
    worker->staging_buffer_offset = worker_chunk_size * i;
    worker->staging_buffer_length = worker_chunk_size;

    // Create semaphore for tracking worker progress.
    worker->pending_timepoint = 0ull;
    status = iree_hal_semaphore_create(device, worker->pending_timepoint,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &worker->semaphore);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "worker count: ");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)worker_count);
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "worker chunk size: ");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)worker_chunk_size);
    *out_operation = operation;
  } else {
    iree_hal_transfer_operation_release(operation);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_transfer_operation_retain(
    iree_hal_transfer_operation_t* operation) {
  if (IREE_LIKELY(operation)) {
    iree_atomic_ref_count_inc(&operation->ref_count);
  }
}

static void iree_hal_transfer_operation_release(
    iree_hal_transfer_operation_t* operation) {
  if (IREE_LIKELY(operation) &&
      iree_atomic_ref_count_dec(&operation->ref_count) == 1) {
    iree_hal_transfer_operation_destroy(operation);
  }
}

static void iree_hal_transfer_operation_destroy(
    iree_hal_transfer_operation_t* operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation->trace_id);
  iree_allocator_t host_allocator =
      iree_hal_device_host_allocator(operation->device);

  // We don't want any pending loop operations when freeing as the loop event
  // handlers will try to access the memory.
  IREE_ASSERT(operation->live_workers == 0, "all workers must have exited");

  for (iree_host_size_t i = 0; i < operation->worker_count; ++i) {
    iree_hal_semaphore_release(operation->workers[i].semaphore);
  }
  iree_hal_buffer_release(operation->staging_buffer);
  for (iree_host_size_t i = 0; i < operation->signal_semaphore_list.count;
       ++i) {
    iree_hal_semaphore_release(operation->signal_semaphore_list.semaphores[i]);
  }
  iree_hal_buffer_release(operation->buffer);
  iree_hal_file_release(operation->file);
  iree_hal_device_release(operation->device);
  iree_status_ignore(operation->error_status);

  iree_allocator_free(host_allocator, operation);

  IREE_TRACE_ZONE_END(z0);
}

// Notifies listeners that the operation has completed and releases its memory.
// If this was a read then the staging buffer dealloca will be chained to the
// last asynchronous copies. In writes the last flush to the file happened
// synchronously so the dealloca happens immediately.
//
// Pre-condition: all workers have exited and there are no operations in flight.
// Post-condition: the operation is freed.
static void iree_hal_transfer_operation_notify_completion(
    iree_hal_transfer_operation_t* operation) {
  IREE_ASSERT_ARGUMENT(operation);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation->trace_id);

  // We can only free the operation if no workers have pending work.
  IREE_ASSERT(operation->live_workers == 0, "no workers can be live");

  // Deallocating the staging buffer can only happen after all workers have
  // completed copies into/out-of it. In reads it's expected there are copies
  // in-flight and we can wait on all worker semaphores. In writes the last
  // flush to the file happened synchronously so we don't need to wait at all.
  iree_hal_semaphore_list_t wait_semaphore_list =
      iree_hal_semaphore_list_empty();
  if (operation->direction == IREE_HAL_TRANSFER_READ_FILE_TO_BUFFER) {
    wait_semaphore_list.count = operation->worker_count;
    wait_semaphore_list.semaphores = (iree_hal_semaphore_t**)iree_alloca(
        wait_semaphore_list.count * sizeof(wait_semaphore_list.semaphores[0]));
    wait_semaphore_list.payload_values =
        (uint64_t*)iree_alloca(wait_semaphore_list.count *
                               sizeof(wait_semaphore_list.payload_values[0]));
    for (iree_host_size_t i = 0; i < operation->worker_count; ++i) {
      iree_hal_transfer_worker_t* worker = &operation->workers[i];
      wait_semaphore_list.semaphores[i] = worker->semaphore;
      wait_semaphore_list.payload_values[i] = worker->pending_timepoint;
    }
  }

  // When the dealloca completes signal the original semaphores passed in to the
  // operation. If the transfer failed then we need to signal them all to
  // failure after the dealloca so we use the same semaphores but set the
  // failure payload.
  iree_hal_semaphore_list_t signal_semaphore_list =
      operation->signal_semaphore_list;
  if (!iree_status_is_ok(operation->error_status)) {
    for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
      signal_semaphore_list.payload_values[i] =
          IREE_HAL_SEMAPHORE_FAILURE_VALUE;
    }
  }

  // Schedule staging buffer deallocation.
  // Note that we need to do this even if the operation failed and we want it to
  // be scheduled after any copies that may be in-flight (say worker 4 failed
  // on a chunk but workers 0-3 succeeded).
  iree_status_t status = iree_hal_device_queue_dealloca(
      operation->device, operation->queue_affinity, wait_semaphore_list,
      signal_semaphore_list, operation->staging_buffer);

  // If the dealloca failed we don't have a great way of letting anyone know.
  // We'll just drop it on the floor for now and let the buffer be freed by
  // reference counting.
  iree_status_ignore(status);

  IREE_TRACE_ZONE_END(z0);
}

// Exits the |worker| indicating it will process no more chunks.
// If this is the last worker to exit the transfer is considered completed.
// The first non-OK |status| provided will be set as a stick error on the
// |operation| and all workers will check it and exit themselves asynchronously.
//
// NOTE: this may end the entire operation and free the operation (and worker)
// memory. Callers must not touch either |operation| or |worker| after calling
// this method.
static iree_status_t iree_hal_transfer_worker_exit(
    iree_hal_transfer_operation_t* operation,
    iree_hal_transfer_worker_t* worker, iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)operation->trace_id);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)worker->trace_id);

  // Check if this is the first failure of an operation and set the error bit.
  // Otherwise we ignore the error here as it's probably just telling us the
  // worker has aborted.
  if (iree_status_is_ok(operation->error_status) &&
      !iree_status_is_ok(status)) {
    operation->error_status = status;
  } else {
    iree_status_ignore(status);
  }

  // Clear the worker live bit and see if there are any more workers live. So
  // long as there is at least one we need to keep the operation running.
  iree_host_size_t worker_index =
      (iree_host_size_t)(worker - operation->workers);
  operation->live_workers &= ~(1ull << worker_index);
  if (operation->live_workers > 0) {
    // Other workers are still live - this is just one worker exiting by not
    // rescheduling itself.
    iree_hal_transfer_operation_release(operation);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // This was the last worker; the operation has completed!
  iree_hal_transfer_operation_notify_completion(operation);

  // Free the operation - the dealloca (and maybe even some copies) may still be
  // in-flight but all on the device side and not using any resources on the
  // operation.
  iree_hal_transfer_operation_release(operation);
  // NOTE: at this point the worker may have freed itself and its memory can
  // no longer be used!

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();  // always ok; just for convenience.
}

static iree_status_t iree_hal_transfer_worker_copy_file_to_buffer(
    void* user_data, iree_loop_t loop, iree_status_t status) {
  iree_hal_transfer_worker_t* worker = (iree_hal_transfer_worker_t*)user_data;
  iree_hal_transfer_operation_t* operation = worker->operation;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)operation->trace_id);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)worker->trace_id);

  // Bail immediately if the operation has failed.
  if (!iree_status_is_ok(status) ||
      !iree_status_is_ok(operation->error_status)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "bail: loop error");
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_transfer_worker_exit(operation, worker, status);
  }

  // Early-exit if we're out of chunks to process.
  // This can happen with some loop implementations that run things in batches.
  if (operation->remaining_chunks == 0) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "exit: no remaining chunks");
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_transfer_worker_exit(operation, worker, iree_ok_status());
  }

  // Grab a piece of the transfer to operate on.
  --operation->remaining_chunks;
  iree_device_size_t transfer_offset = operation->transfer_head;
  iree_device_size_t transfer_length = iree_min(
      operation->length - transfer_offset, worker->staging_buffer_length);
  IREE_ASSERT(transfer_length > 0,
              "should not have ticked if there was no work to do");
  operation->transfer_head += transfer_length;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)transfer_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)transfer_length);

  // Timeline increments by one.
  uint64_t wait_timepoint = worker->pending_timepoint;
  iree_hal_semaphore_list_t wait_semaphore_list = {
      .count = 1,
      .semaphores = &worker->semaphore,
      .payload_values = &wait_timepoint,
  };
  uint64_t signal_timepoint = ++worker->pending_timepoint;
  iree_hal_semaphore_list_t signal_semaphore_list = {
      .count = 1,
      .semaphores = &worker->semaphore,
      .payload_values = &signal_timepoint,
  };

  // Track the pending copy operation so we know where to place it in the
  // buffer.
  worker->pending_transfer_offset = transfer_offset;
  worker->pending_transfer_length = transfer_length;

  // Synchronously copy the contents from the file to the staging buffer.
  status = iree_hal_file_read(
      operation->file, operation->file_offset + worker->pending_transfer_offset,
      operation->staging_buffer, worker->staging_buffer_offset,
      worker->pending_transfer_length);

  // Issue asynchronous copy from the staging buffer into the target buffer.
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        operation->device, operation->queue_affinity, wait_semaphore_list,
        signal_semaphore_list, operation->staging_buffer,
        worker->staging_buffer_offset, operation->buffer,
        operation->buffer_offset + transfer_offset, transfer_length,
        IREE_HAL_COPY_FLAG_NONE);
  }

  // Wait for the copy to complete and tick again if we expect there to be more
  // work. If there are no more chunks to copy (or they are spoken for by other
  // live workers) we can avoid the loop wait and exit such that the dealloca
  // can chain on to the copy operations.
  if (iree_status_is_ok(status)) {
    if (iree_hal_transfer_worker_live_count(operation->live_workers) >
        operation->remaining_chunks) {
      // Enough workers to cover all remaining chunks so we can exit now and
      // avoid an additional host wake (+ latency) by the loop event.
      IREE_TRACE_ZONE_APPEND_TEXT(z0,
                                  "exit: remaining chunks covered by workers");
      status = iree_hal_transfer_worker_exit(operation, worker, status);
    } else {
      status = iree_loop_wait_one(
          loop,
          iree_hal_semaphore_await(worker->semaphore,
                                   worker->pending_timepoint),
          iree_infinite_timeout(), iree_hal_transfer_worker_copy_file_to_buffer,
          worker);
    }
  }

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "bail: copy/wait failure");
    status = iree_hal_transfer_worker_exit(operation, worker, status);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Begins the transfer operation after |wait_semaphore_list| is satisfied.
// Note that if this fails then the transfer never started and it's safe to
// immediately tear down.
static iree_status_t iree_hal_transfer_operation_launch_read(
    iree_hal_transfer_operation_t* operation,
    iree_hal_semaphore_list_t wait_semaphore_list, iree_loop_t loop) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)operation->trace_id);
  IREE_ASSERT(operation->direction == IREE_HAL_TRANSFER_READ_FILE_TO_BUFFER);

  // Staging buffers get allocated based on the direction we are transferring.
  // This optimizes for access patterns such as sequential writes from the host
  // when staging into the buffer and sequential cached reads from the host when
  // staging out of the buffer.
  iree_hal_buffer_params_t staging_buffer_params = {
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      // TODO(benvanik): make staging alignment an option/device query?
      .min_alignment = 64,
      .queue_affinity = operation->queue_affinity,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
  };

  // Queue the staging buffer allocation.
  // When it completes we'll do the first host->device copy via mapping.
  iree_hal_semaphore_list_t alloca_semaphore_list = {
      .count = operation->worker_count,
      .semaphores =
          iree_alloca(sizeof(iree_hal_semaphore_t*) * operation->worker_count),
      .payload_values = iree_alloca(sizeof(uint64_t) * operation->worker_count),
  };
  for (iree_host_size_t i = 0; i < operation->worker_count; ++i) {
    iree_hal_transfer_worker_t* worker = &operation->workers[i];
    alloca_semaphore_list.semaphores[i] = worker->semaphore;
    alloca_semaphore_list.payload_values[i] = ++worker->pending_timepoint;
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_alloca(
              operation->device, operation->queue_affinity, wait_semaphore_list,
              alloca_semaphore_list, IREE_HAL_ALLOCATOR_POOL_DEFAULT,
              staging_buffer_params, operation->staging_buffer_size,
              &operation->staging_buffer));

  // After the alloca completes each worker will be at the same starting point.
  // We'll wait on each and start the worker-specific coroutines.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t worker_index = 0;
       worker_index < operation->worker_count; ++worker_index) {
    iree_hal_transfer_worker_t* worker = &operation->workers[worker_index];
    operation->live_workers |= 1ull << worker_index;
    iree_hal_transfer_operation_retain(operation);
    status = iree_loop_wait_one(
        loop,
        iree_hal_semaphore_await(worker->semaphore, worker->pending_timepoint),
        iree_infinite_timeout(), iree_hal_transfer_worker_copy_file_to_buffer,
        worker);
    if (!iree_status_is_ok(status)) {
      operation->live_workers &= ~(1ull << worker_index);
      iree_hal_transfer_operation_release(operation);
      break;
    }

    // It's possible that the entire operation completed inline.
    if (operation->remaining_chunks == 0) break;
  }
  if (!iree_status_is_ok(status)) {
    // Failed to wait on one of the workers. This is a fatal error but we may
    // have already waited on some workers and need to instead set the sticky
    // error flag so that when any complete they stop processing.
    operation->error_status = status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();  // return ok as loop is fine but operation is not
}

static iree_status_t iree_hal_transfer_worker_copy_staging_to_file(
    void* user_data, iree_loop_t loop, iree_status_t status);

static iree_status_t iree_hal_transfer_worker_copy_buffer_to_staging(
    iree_hal_transfer_operation_t* operation,
    iree_hal_transfer_worker_t* worker, iree_loop_t loop) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)operation->trace_id);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)worker->trace_id);

  // If there's been an error we bail.
  if (!iree_status_is_ok(operation->error_status)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "bail: error bit set");
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_transfer_worker_exit(operation, worker, iree_ok_status());
  }

  // Grab a piece of the transfer to operate on.
  IREE_ASSERT(operation->remaining_chunks > 0,
              "should not have ticked if there was no work to do");
  --operation->remaining_chunks;
  iree_device_size_t transfer_offset = operation->transfer_head;
  iree_device_size_t transfer_length = iree_min(
      operation->length - transfer_offset, worker->staging_buffer_length);
  IREE_ASSERT(transfer_length > 0,
              "should not have ticked if there was no work to do");
  operation->transfer_head += transfer_length;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)transfer_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)transfer_length);

  // Timeline increments by one.
  iree_hal_semaphore_list_t wait_semaphore_list = {
      .count = 1,
      .semaphores = &worker->semaphore,
      .payload_values = &worker->pending_timepoint,
  };
  ++worker->pending_timepoint;
  iree_hal_semaphore_list_t signal_semaphore_list = {
      .count = 1,
      .semaphores = &worker->semaphore,
      .payload_values = &worker->pending_timepoint,
  };

  // Track the pending copy operation so we know where to place it in the file.
  worker->pending_transfer_offset = transfer_offset;
  worker->pending_transfer_length = transfer_length;

  // Issue an asynchronous copy from the source buffer to the staging buffer.
  iree_status_t status = iree_hal_device_queue_copy(
      operation->device, operation->queue_affinity, wait_semaphore_list,
      signal_semaphore_list, operation->buffer,
      operation->buffer_offset + transfer_offset, operation->staging_buffer,
      worker->staging_buffer_offset, transfer_length, IREE_HAL_COPY_FLAG_NONE);

  // Wait for the copy to complete so we can write it to the file.
  if (iree_status_is_ok(status)) {
    status = iree_loop_wait_one(
        loop,
        iree_hal_semaphore_await(worker->semaphore, worker->pending_timepoint),
        iree_infinite_timeout(), iree_hal_transfer_worker_copy_staging_to_file,
        worker);
  }

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "bail: copy/wait failure");
    status = iree_hal_transfer_worker_exit(operation, worker, status);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_transfer_worker_copy_staging_to_file(
    void* user_data, iree_loop_t loop, iree_status_t status) {
  iree_hal_transfer_worker_t* worker = (iree_hal_transfer_worker_t*)user_data;
  iree_hal_transfer_operation_t* operation = worker->operation;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)operation->trace_id);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)worker->trace_id);

  // Bail immediately if the operation has failed.
  if (!iree_status_is_ok(status) ||
      !iree_status_is_ok(operation->error_status)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "bail: loop error");
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_transfer_worker_exit(operation, worker, status);
  }

  // Synchronously copy the contents from the staging buffer to the file.
  status = iree_hal_file_write(
      operation->file, operation->file_offset + worker->pending_transfer_offset,
      operation->staging_buffer, worker->staging_buffer_offset,
      worker->pending_transfer_length);

  if (iree_status_is_ok(status) && operation->remaining_chunks == 0) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "exit: no more chunks remaining to write");
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_transfer_worker_exit(operation, worker, iree_ok_status());
  }

  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "bail: file write error");
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_transfer_worker_exit(operation, worker, status);
  }

  IREE_TRACE_ZONE_END(z0);

  // Tail call: tick the worker so that it transfers another chunk.
  return iree_hal_transfer_worker_copy_buffer_to_staging(operation, worker,
                                                         loop);
}

// Begins the transfer operation after |wait_semaphore_list| is satisfied.
// Note that if this fails then the transfer never started and it's safe to
// immediately tear down.
static iree_status_t iree_hal_transfer_operation_launch_write(
    iree_hal_transfer_operation_t* operation,
    iree_hal_semaphore_list_t wait_semaphore_list, iree_loop_t loop) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)operation->trace_id);
  IREE_ASSERT(operation->direction == IREE_HAL_TRANSFER_WRITE_BUFFER_TO_FILE);

  // Staging buffers get allocated based on the direction we are transferring.
  // This optimizes for access patterns such as sequential writes from the host
  // when staging into the buffer and sequential cached reads from the host when
  // staging out of the buffer.
  iree_hal_buffer_params_t staging_buffer_params = {
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      // TODO(benvanik): make staging alignment an option/device query?
      .min_alignment = 64,
      .queue_affinity = operation->queue_affinity,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST |
              IREE_HAL_MEMORY_TYPE_HOST_CACHED |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM,
  };

  // Queue the staging buffer allocation.
  // When it completes we'll signal each worker to start its first transfer.
  iree_hal_semaphore_list_t alloca_semaphore_list = {
      .count = operation->worker_count,
      .semaphores =
          iree_alloca(sizeof(iree_hal_semaphore_t*) * operation->worker_count),
      .payload_values = iree_alloca(sizeof(uint64_t) * operation->worker_count),
  };
  for (iree_host_size_t i = 0; i < operation->worker_count; ++i) {
    iree_hal_transfer_worker_t* worker = &operation->workers[i];
    alloca_semaphore_list.semaphores[i] = worker->semaphore;
    alloca_semaphore_list.payload_values[i] = ++worker->pending_timepoint;
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_alloca(
              operation->device, operation->queue_affinity, wait_semaphore_list,
              alloca_semaphore_list, IREE_HAL_ALLOCATOR_POOL_DEFAULT,
              staging_buffer_params, operation->staging_buffer_size,
              &operation->staging_buffer));

  // After the alloca completes each worker will be at the same starting point.
  // We'll wait on each and start the worker-specific coroutines.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t worker_index = 0;
       worker_index < operation->worker_count; ++worker_index) {
    iree_hal_transfer_worker_t* worker = &operation->workers[worker_index];
    operation->live_workers |= 1ull << worker_index;
    iree_hal_transfer_operation_retain(operation);

    // Issue the initial asynchronous copy from the source buffer to the worker
    // chunk. This will wait for the alloca to complete so that the staging
    // buffer is available for use. After the copy completes the worker will
    // tick itself so long as there are chunks remaining to write.
    status = iree_hal_transfer_worker_copy_buffer_to_staging(operation, worker,
                                                             loop);
    if (!iree_status_is_ok(status)) break;

    // It's possible that the entire operation completed inline.
    if (operation->remaining_chunks == 0) break;
  }
  if (!iree_status_is_ok(status)) {
    // Failed to wait on one of the workers. This is a fatal error but we may
    // have already waited on some workers and need to instead set the sticky
    // error flag so that when any complete they stop processing.
    operation->error_status = status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();  // return ok as loop is fine but operation is not
}

//===----------------------------------------------------------------------===//
// Memory file IO API
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_file_validate_access(
    iree_hal_file_t* file, iree_hal_memory_access_t required_access) {
  const iree_hal_memory_access_t allowed_access =
      iree_hal_file_allowed_access(file);
  if (IREE_LIKELY(iree_all_bits_set(allowed_access, required_access))) {
    return iree_ok_status();
  }
#if IREE_STATUS_MODE
  iree_bitfield_string_temp_t temp0, temp1;
  iree_string_view_t allowed_access_str =
      iree_hal_memory_access_format(allowed_access, &temp0);
  iree_string_view_t required_access_str =
      iree_hal_memory_access_format(required_access, &temp1);
  return iree_make_status(
      IREE_STATUS_PERMISSION_DENIED,
      "file operation cannot be performed; file allows %.*s, operation "
      "requires %.*s",
      (int)allowed_access_str.size, allowed_access_str.data,
      (int)required_access_str.size, required_access_str.data);
#else
  return iree_make_status(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_read_streaming(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags,
    iree_hal_file_transfer_options_t options) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ));

  // If the file implicitly supports device transfer then we can simply issue a
  // device copy.
  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(source_file);
  if (storage_buffer) {
    return iree_hal_device_queue_copy(
        device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        storage_buffer, (iree_device_size_t)source_offset, target_buffer,
        target_offset, length, IREE_HAL_COPY_FLAG_NONE);
  }

  // Allocate full transfer operation.
  iree_hal_transfer_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_transfer_operation_create(
      device, queue_affinity, signal_semaphore_list,
      IREE_HAL_TRANSFER_READ_FILE_TO_BUFFER, source_file, source_offset,
      target_buffer, target_offset, length, options, &operation));

  // Kick off the streaming transfer.
  // This will queue allocation of the staging buffer and then issue one or more
  // copy commands. The operation will manage its own lifetime and emit errors
  // as part of signal semaphore failures.
  iree_status_t status = iree_hal_transfer_operation_launch_read(
      operation, wait_semaphore_list, options.loop);

  iree_hal_transfer_operation_release(operation);

  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_queue_write_streaming(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags,
    iree_hal_file_transfer_options_t options) {
  // EXPERIMENTAL: assume memory files only today (as that's all we have).
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(target_file, IREE_HAL_MEMORY_ACCESS_WRITE));

  // If the file implicitly supports device transfer then we can simply issue a
  // device copy.
  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(target_file);
  if (storage_buffer) {
    return iree_hal_device_queue_copy(
        device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        source_buffer, source_offset, storage_buffer,
        (iree_device_size_t)target_offset, length, IREE_HAL_COPY_FLAG_NONE);
  }

  // Allocate full transfer operation.
  iree_hal_transfer_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_transfer_operation_create(
      device, queue_affinity, signal_semaphore_list,
      IREE_HAL_TRANSFER_WRITE_BUFFER_TO_FILE, target_file, target_offset,
      source_buffer, source_offset, length, options, &operation));

  // Kick off the streaming transfer.
  // This will queue allocation of the staging buffer and then issue one or more
  // copy commands. The operation will manage its own lifetime and emit errors
  // as part of signal semaphore failures.
  iree_status_t status = iree_hal_transfer_operation_launch_write(
      operation, wait_semaphore_list, options.loop);

  iree_hal_transfer_operation_release(operation);

  return status;
}
