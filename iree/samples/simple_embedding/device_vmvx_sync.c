// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A example of setting up the vmvx-sync driver.

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#include "iree/hal/local/sync_device.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/simple_embedding/simple_embedding_test_bytecode_module_vmvx_c.h"

iree_status_t create_sample_device(iree_hal_device_t** device) {
  // Set parameters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance));

  iree_hal_executable_loader_t* loader = NULL;
  iree_status_t status = iree_hal_vmvx_module_loader_create(
      instance, iree_allocator_system(), &loader);
  iree_vm_instance_release(instance);

  iree_string_view_t identifier = iree_make_cstring_view("vmvx");
  if (iree_status_is_ok(status)) {
    // Create the synchronous device.
    status =
        iree_hal_sync_device_create(identifier, &params, /*loader_count=*/1,
                                    &loader, iree_allocator_system(), device);
  }
  iree_hal_executable_loader_release(loader);
  return status;
}

const iree_const_byte_span_t load_bytecode_module_data() {
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_simple_embedding_test_module_vmvx_create();
  return iree_make_const_byte_span(module_file_toc->data,
                                   module_file_toc->size);
}
