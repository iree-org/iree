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
//   1. Create a driver and device from the registry.
//   2. Create a device group (assigns topology info to the device).
//   3. Allocate two buffers via the device allocator.
//   4. Record a command buffer that fills one buffer and copies to the other.
//   5. Submit the command buffer to the device queue with a signal semaphore.
//   6. Wait for the semaphore, then read back and verify results.
//
// No compiled shaders are needed -- fill and copy are built-in transfer
// operations supported by every HAL driver.
//
// Usage:
//   hello [driver_name]
//
// Examples:
//   hello                  # Uses local-sync (default)
//   hello local-task       # Uses the task system driver
//   hello vulkan           # Uses Vulkan (if available)

#include <inttypes.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"

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

  fprintf(stdout, "4. Recording command buffer...\n");

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

  fprintf(stdout, "5. Submitting to device queue...\n");

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

  fprintf(stdout, "6. Waiting for completion...\n");

  // Block until the semaphore reaches value 1 (command buffer retired).
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(semaphore, signal_value,
                                               iree_infinite_timeout(),
                                               IREE_HAL_WAIT_FLAG_DEFAULT));

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

static iree_status_t run_sample(const char* driver_name) {
  iree_allocator_t host_allocator = iree_allocator_system();

  // All NULL-initialized so cleanup is unconditional.
  iree_hal_driver_t* driver = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_device_group_t* device_group = NULL;
  iree_hal_buffer_t* source_buffer = NULL;
  iree_hal_buffer_t* destination_buffer = NULL;
  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_hal_semaphore_t* semaphore = NULL;

  //--- 1. Register drivers and create device ----------------------------------

  fprintf(stdout, "1. Creating device (driver=%s)...\n", driver_name);

  // Register all linked-in drivers with the default registry.
  iree_status_t status = iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default());

  // Create the requested driver. This looks up the driver factory by name and
  // instantiates it.
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view(driver_name),
        host_allocator, &driver);
  }

  // Create the default device from the driver. For drivers with multiple
  // devices (multi-GPU) this picks the first one.
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_driver_create_default_device(driver, host_allocator, &device);
  }
  // Driver is no longer needed -- the device is self-contained.
  iree_hal_driver_release(driver);
  driver = NULL;

  //--- 2. Create device group (topology) --------------------------------------

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "2. Creating device group...\n");

    // The device group owns the topology matrix and assigns topology info to
    // each device. Even with a single device this is required -- it populates
    // the device's self-edge with capability information and gives the device a
    // topology index.
    status = iree_hal_device_group_create_from_device(device, host_allocator,
                                                      &device_group);
  }

  if (iree_status_is_ok(status)) {
    iree_string_view_t device_id = iree_hal_device_id(device);
    if (!device_id.size) device_id = iree_make_cstring_view("(unnamed)");
    fprintf(stdout, "   Device: '%.*s'\n", (int)device_id.size, device_id.data);
    fprintf(stdout, "   Devices in group: %" PRIhsz "\n",
            iree_hal_device_group_device_count(device_group));
  }

  //--- 3. Allocate buffers and create command buffer / semaphore
  //---------------

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "3. Allocating buffers (%" PRIhsz " bytes each)...\n",
            (iree_host_size_t)BUFFER_SIZE);
  }

  // Both buffers need host-visible memory so we can read back results.
  // TRANSFER usage allows fill and copy operations.
  // MAPPING allows host-side map/read for readback.
  iree_hal_buffer_params_t buffer_params = {
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
               IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_MAPPING,
  };

  // Source buffer: will be filled with a pattern by the command buffer.
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), buffer_params, BUFFER_SIZE,
        &source_buffer);
  }

  // Destination buffer: will receive the copy from source.
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), buffer_params, BUFFER_SIZE,
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

  //--- 4-6. Record, submit, wait, verify --------------------------------------

  if (iree_status_is_ok(status)) {
    status = record_and_verify(device, source_buffer, destination_buffer,
                               command_buffer, semaphore);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "HAL hello sample completed successfully.\n");
  }

  //--- Cleanup (all NULL-safe, reverse order) ---------------------------------

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(destination_buffer);
  iree_hal_buffer_release(source_buffer);
  // Release device before device group -- the device holds a raw pointer
  // to the group's embedded topology.
  iree_hal_device_release(device);
  iree_hal_device_group_release(device_group);
  return status;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  const char* driver_name = "local-sync";
  if (argc > 1) {
    driver_name = argv[1];
  }
  iree_status_t status = run_sample(driver_name);
  if (iree_status_is_ok(status)) {
    return 0;
  }
  iree_status_fprint(stderr, status);
  fprintf(stderr, "\n");
  int code = (int)iree_status_code(status);
  iree_status_free(status);
  return code;
}
