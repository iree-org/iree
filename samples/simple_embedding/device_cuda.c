// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A example of setting up the the cuda driver.

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/registration/driver_module.h"

// Compiled module embedded here to avoid file IO:
#include "samples/simple_embedding/simple_embedding_test_bytecode_module_cuda_c.h"

iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device) {
  // Only register the CUDA HAL driver.
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_driver_module_register(iree_hal_driver_registry_default()));

  // Create the HAL driver from the name.
  iree_hal_driver_t* driver = NULL;
  iree_string_view_t identifier = iree_make_cstring_view("cuda");
  iree_status_t status = iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(), identifier, host_allocator, &driver);

  // Create the default device (primary GPU).
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(driver, host_allocator,
                                                   out_device);
  }

  iree_hal_driver_release(driver);
  return iree_ok_status();
}

const iree_const_byte_span_t load_bytecode_module_data() {
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_simple_embedding_test_module_cuda_create();
  return iree_make_const_byte_span(module_file_toc->data,
                                   module_file_toc->size);
}
