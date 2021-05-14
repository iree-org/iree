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

#include "iree/hal/drivers/init.h"

#include "iree/base/tracing.h"

#if defined(IREE_HAL_HAVE_CUDA_DRIVER_MODULE)
#include "iree/hal/cuda/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_CUDA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_DRIVER_MODULE)
#include "iree/hal/dylib/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_DYLIB_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE)
#include "iree/hal/dylib/registration/driver_module_sync.h"
#endif  // IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMLA_DRIVER_MODULE)
#include "iree/hal/vmla/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_VMLA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMVX_DRIVER_MODULE)
#include "iree/hal/vmvx/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_VMVX_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VULKAN_DRIVER_MODULE)
#include "iree/hal/vulkan/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_VULKAN_DRIVER_MODULE

IREE_API_EXPORT iree_status_t
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t* registry) {
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_HAL_HAVE_CUDA_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_CUDA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_DYLIB_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_sync_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMLA_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vmla_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_VMLA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMVX_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vmvx_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_VMVX_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VULKAN_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_VULKAN_DRIVER_MODULE

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
