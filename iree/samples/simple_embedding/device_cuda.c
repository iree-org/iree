// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A example of setting up the the cuda driver.

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cuda/api.h"
#include "iree/hal/cuda/registration/driver_module.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/simple_embedding/simple_embedding_test_bytecode_module_vulkan_c.h"

iree_status_t create_sample_device(iree_hal_device_t** device) {
  // Only register the cuda HAL driver.
  IREE_RETURN_IF_ERROR(iree_hal_cuda_driver_module_register(
      iree_hal_driver_registry_default()));
  // Create the hal driver from the name.
  iree_hal_driver_t* driver = NULL;
  iree_string_view_t identifier = iree_make_cstring_view("cuda");
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create_by_name(
      iree_hal_driver_registry_default(), identifier, iree_allocator_system(),
      &driver));
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), device));
  iree_hal_driver_release(driver);
  return iree_ok_status();
}

const iree_const_byte_span_t load_bytecode_module_data() {
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_simple_embedding_test_module_vulkan_create();
  return iree_make_const_byte_span(module_file_toc->data,
                                   module_file_toc->size);
}
