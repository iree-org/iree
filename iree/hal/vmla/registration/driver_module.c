// Copyright 2020 Google LLC
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

#include "iree/hal/vmla/registration/driver_module.h"

#include <inttypes.h>

#include "iree/hal/local/loaders/vmla_module_loader.h"
#include "iree/hal/local/task_driver.h"

// TODO(#4298): remove this driver registration and wrapper.

#define IREE_HAL_VMLA_DRIVER_ID 0x564D4C41u  // VMLA

static iree_status_t iree_hal_vmla_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  static const iree_hal_driver_info_t driver_infos[1] = {
      {
          .driver_id = IREE_HAL_VMLA_DRIVER_ID,
          .driver_name = iree_string_view_literal("vmla"),
          .full_name =
              iree_string_view_literal("Reference backend (deprecated)"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_vmla_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  if (driver_id != IREE_HAL_VMLA_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }

  iree_hal_task_device_params_t default_params;
  iree_hal_task_device_params_initialize(&default_params);

  // NOTE: VMLA doesn't tile so we don't really need many workers - having
  // multiple does make it easier to test overlapping execution, though.
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(4, &topology);

  iree_vm_instance_t* instance = NULL;
  iree_status_t status = iree_vm_instance_create(allocator, &instance);

  iree_hal_executable_loader_t* vmla_loader = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vmla_module_loader_create(instance, allocator, &vmla_loader);
  }
  iree_hal_executable_loader_t* loaders[1] = {vmla_loader};

  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create(IREE_TASK_SCHEDULING_MODE_RESERVED,
                                       &topology, allocator, &executor);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_task_driver_create(
        iree_make_cstring_view("vmla"), &default_params, executor,
        IREE_ARRAYSIZE(loaders), loaders, allocator, out_driver);
  }

  iree_task_executor_release(executor);
  iree_task_topology_deinitialize(&topology);
  iree_hal_executable_loader_release(vmla_loader);
  iree_vm_instance_release(instance);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_vmla_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      /*self=*/NULL,
      iree_hal_vmla_driver_factory_enumerate,
      iree_hal_vmla_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
