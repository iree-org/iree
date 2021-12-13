// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/vulkan/registration/driver_module.h"

namespace iree {
namespace hal {
namespace cts {

iree_status_t register_test_driver(iree_hal_driver_registry_t* registry) {
  return iree_hal_vulkan_driver_module_register(registry);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree
