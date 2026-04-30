// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pure HAL sample: buffer fill, copy, and readback via command buffer.
//
// Demonstrates the core HAL workflow without any VM, bytecode modules, or
// compiled executables. This is the minimal path through the HAL:
//
//   1. Create a device from the registry by URI.
//   2. Allocate two buffers with appropriate memory types and usage flags.
//   3. Record a command buffer that fills one buffer and copies to the other.
//   4. Submit the command buffer to the device queue with a signal semaphore.
//   5. Wait for the semaphore, then read back and verify results.
//
// No compiled shaders are needed -- fill and copy are built-in transfer
// operations supported by every HAL driver.
//
// Usage:
//   hello --device=local-sync
//   hello --device=local-task
//   hello --device=vulkan

#include <inttypes.h>
#include <stdio.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"

IREE_FLAG(string, device, "local-sync",
          "HAL device URI to use for execution.\n"
          "Examples: local-sync, local-task, vulkan, cuda, hip");

// The fill pattern and buffer size are chosen to be small enough to print
// but large enough to exercise the command buffer path (not inlined).
#define BUFFER_ELEMENT_COUNT 16
#define BUFFER_SIZE (BUFFER_ELEMENT_COUNT * sizeof(uint32_t))
#define FILL_PATTERN UINT32_C(0xCAFEF00D)

//===----------------------------------------------------------------------===//
// Command recording, submission, and verification.
//
// This function owns nothing -- all resources are passed in and released by
// the caller. That means every operation can use IREE_RETURN_IF_ERROR without
// worrying about cleanup.
//===----------------------------------------------------------------------===//

static iree_status_t record_and_verify(
    iree_hal_device_t* device, iree_hal_buffer_t* source_buffer,
    iree_hal_buffer_t* destination_buffer,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_semaphore_t* semaphore) {
  //--- Record command buffer --------------------------------------------------

  fprintf(stdout, "3. Recording command buffer...\n");

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));

  // Fill the source buffer with a repeating 32-bit pattern.
  // The fill operation writes directly into device memory via the transfer
  // engine -- no shader or dispatch is needed.
  uint32_t fill_pattern = FILL_PATTERN;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(source_buffer, /*offset=*/0, BUFFER_SIZE),
      &fill_pattern, sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));

  // Barrier: the fill must complete before we read the source buffer for copy.
  iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
  };
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL));

  // Copy the entire source buffer into the destination buffer.
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(source_buffer, /*offset=*/0, BUFFER_SIZE),
      iree_hal_make_buffer_ref(destination_buffer, /*offset=*/0, BUFFER_SIZE),
      IREE_HAL_COPY_FLAG_NONE));

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));

  //--- Submit to queue with semaphore signaling -------------------------------

  fprintf(stdout, "4. Submitting to device queue...\n");

  // Submit: no wait semaphores (execute immediately), signal semaphore to 1
  // when the command buffer retires.
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_list, command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));

  //--- Wait and verify --------------------------------------------------------

  fprintf(stdout, "5. Waiting for completion...\n");

  // Block until the semaphore reaches value 1 (command buffer retired).
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(semaphore, signal_value,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));

  fprintf(stdout, "   Command buffer completed.\n");

  // Read back the destination buffer and verify every element matches the fill
  // pattern. This uses a synchronous map -- for production code you would use
  // queue_copy to a staging buffer instead.
  uint32_t readback[BUFFER_ELEMENT_COUNT] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
      destination_buffer, /*source_offset=*/0, readback, sizeof(readback)));

  fprintf(stdout, "   Verifying %d elements...\n", BUFFER_ELEMENT_COUNT);
  for (int i = 0; i < BUFFER_ELEMENT_COUNT; ++i) {
    if (readback[i] != FILL_PATTERN) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "mismatch at element %d: expected 0x%08" PRIX32
                              ", got 0x%08" PRIX32,
                              i, (uint32_t)FILL_PATTERN, readback[i]);
    }
  }

  fprintf(stdout, "\nAll %d elements verified: 0x%08" PRIX32 "\n",
          BUFFER_ELEMENT_COUNT, (uint32_t)FILL_PATTERN);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Resource creation and teardown.
//
// All resources are initialized to NULL and unconditionally released at the
// end -- every iree_hal_*_release is safe to call with NULL.
//===----------------------------------------------------------------------===//

static iree_status_t run_sample(void) {
  iree_allocator_t host_allocator = iree_allocator_system();

  // All NULL-initialized so cleanup is unconditional.
  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_async_frontier_tracker_t* frontier_tracker = NULL;
  iree_hal_device_group_t* device_group = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_buffer_t* source_buffer = NULL;
  iree_hal_buffer_t* destination_buffer = NULL;
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_hal_semaphore_t* semaphore = NULL;

  //--- 1. Create device from registry -----------------------------------------

  fprintf(stdout, "1. Creating device (--device=%s)...\n", FLAG_device);

  // Register all linked-in drivers with the default registry.
  iree_status_t status = iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default());

  // Create a shared proactor pool for async I/O. The device retains the pool
  // so we release our reference immediately after device creation.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), host_allocator,
        &proactor_pool);
  }

  // Create a device directly from the URI. This handles driver lookup,
  // instantiation, and default device selection in one step.
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(iree_hal_driver_registry_default(),
                                    iree_make_cstring_view(FLAG_device),
                                    &create_params, host_allocator, &device);
  }
  iree_async_proactor_pool_release(proactor_pool);

  // Queue operations require devices to be assigned to a causal frontier. A
  // single-device group gives direct HAL users the same production contract the
  // VM HAL module uses for queue ordering and transient-memory reuse.
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), host_allocator,
        &frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(
        device, frontier_tracker, host_allocator, &device_group);
  }
  iree_async_frontier_tracker_release(frontier_tracker);

  if (iree_status_is_ok(status)) {
    iree_string_view_t device_id = iree_hal_device_id(device);
    if (!device_id.size) device_id = iree_make_cstring_view("(unnamed)");
    fprintf(stdout, "   Device: '%.*s'\n", (int)device_id.size, device_id.data);
  }

  //--- 2. Allocate buffers, command buffer, semaphore -------------------------

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "2. Allocating buffers (%" PRIhsz " bytes each)...\n",
            (iree_host_size_t)BUFFER_SIZE);
  }

  // Source buffer: lives on-device. Filled by the command buffer and read
  // during copy -- the host never touches it, so it can be device-local for
  // optimal placement (VRAM on GPUs, same as host memory on CPU backends).
  iree_hal_buffer_params_t source_params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
               IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), source_params, BUFFER_SIZE,
        &source_buffer);
  }

  // Destination buffer: host-visible for readback. The device writes it via
  // copy, then the host maps it to verify results. HOST_LOCAL + DEVICE_VISIBLE
  // places this in system RAM accessible to both host and device.
  iree_hal_buffer_params_t destination_params = {
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET | IREE_HAL_BUFFER_USAGE_MAPPING,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), destination_params, BUFFER_SIZE,
        &destination_buffer);
  }

  // One-shot command buffer for transfer operations.
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_create(
        device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &command_buffer);
  }

  // Timeline semaphore: initial value 0, will be signaled to 1 on completion.
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
  }

  //--- 3-5. Record, submit, wait, verify --------------------------------------

  if (iree_status_is_ok(status)) {
    status = record_and_verify(device, source_buffer, destination_buffer,
                               command_buffer, semaphore);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "HAL hello sample completed successfully.\n");
  }

  //--- Cleanup (all NULL-safe) ------------------------------------------------

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(destination_buffer);
  iree_hal_buffer_release(source_buffer);
  iree_hal_device_release(device);
  iree_hal_device_group_release(device_group);
  return status;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status = run_sample();
  if (iree_status_is_ok(status)) {
    return 0;
  }
  iree_status_fprint(stderr, status);
  fprintf(stderr, "\n");
  int code = (int)iree_status_code(status);
  iree_status_free(status);
  return code;
}
