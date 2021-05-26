// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bindings/python/iree/runtime/binding.h"
#include "bindings/python/iree/runtime/function_abi.h"
#include "bindings/python/iree/runtime/hal.h"
#include "bindings/python/iree/runtime/host_types.h"
#include "bindings/python/iree/runtime/status_utils.h"
#include "bindings/python/iree/runtime/vm.h"
#include "iree/base/status.h"
#include "iree/hal/drivers/init.h"

namespace iree {
namespace python {

PYBIND11_MODULE(binding, m) {
  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  m.doc() = "IREE Binding Backend Helpers";
  SetupFunctionAbiBindings(m);
  SetupHostTypesBindings(m);
  SetupHalBindings(m);
  SetupVmBindings(m);
}

}  // namespace python
}  // namespace iree
