// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/init.h"

#if defined(IREE_HAVE_HAL_CUDA_DRIVER_MODULE)
#include "iree/hal/drivers/cuda2/registration/driver_module.h"
#endif  // IREE_HAVE_HAL_CUDA_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_LOCAL_SYNC_DRIVER_MODULE)
#include "iree/hal/drivers/local_sync/registration/driver_module.h"
#endif  // IREE_HAVE_HAL_LOCAL_SYNC_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_LOCAL_TASK_DRIVER_MODULE)
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#endif  // IREE_HAVE_HAL_LOCAL_TASK_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_VULKAN_DRIVER_MODULE)
#include "iree/hal/drivers/vulkan/registration/driver_module.h"
#endif  // IREE_HAVE_HAL_VULKAN_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_METAL_DRIVER_MODULE)
#include "iree/hal/drivers/metal/registration/driver_module.h"
#endif  // IREE_HAVE_HAL_METAL_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_EXTERNAL_DRIVERS)
// Defined in the generated init_external.c file:
extern iree_status_t iree_hal_register_external_drivers(
    iree_hal_driver_registry_t* registry);
#else
static iree_status_t iree_hal_register_external_drivers(
    iree_hal_driver_registry_t* registry) {
  return iree_ok_status();
}
#endif  // IREE_HAVE_HAL_EXTERNAL_DRIVERS

IREE_API_EXPORT iree_status_t
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t* registry) {
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_HAVE_HAL_CUDA_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda2_driver_module_register(registry));
#endif  // IREE_HAVE_HAL_CUDA_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_LOCAL_SYNC_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_local_sync_driver_module_register(registry));
#endif  // IREE_HAVE_HAL_LOCAL_SYNC_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_LOCAL_TASK_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_local_task_driver_module_register(registry));
#endif  // IREE_HAVE_HAL_LOCAL_TASK_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_VULKAN_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_driver_module_register(registry));
#endif  // IREE_HAVE_HAL_VULKAN_DRIVER_MODULE

#if defined(IREE_HAVE_HAL_METAL_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_metal_driver_module_register(registry));
#endif  // IREE_HAVE_HAL_METAL_DRIVER_MODULE

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_register_external_drivers(registry));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
