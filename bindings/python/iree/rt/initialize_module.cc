// Copyright 2019 Google LLC
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

#include "bindings/python/iree/rt/binding.h"
#include "bindings/python/iree/rt/function_abi.h"
#include "bindings/python/iree/rt/hal.h"
#include "bindings/python/iree/rt/host_types.h"
#include "bindings/python/iree/rt/status_utils.h"
#include "bindings/python/iree/rt/vm.h"
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
