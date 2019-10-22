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

#include "iree/bindings/python/pyiree/binding.h"

#include "iree/bindings/python/pyiree/compiler.h"
#include "iree/bindings/python/pyiree/hal.h"
#include "iree/bindings/python/pyiree/rt.h"
#include "iree/bindings/python/pyiree/status_utils.h"
#include "iree/bindings/python/pyiree/vm.h"

namespace iree {
namespace python {

PYBIND11_MODULE(binding, m) {
  m.doc() = "IREE Binding Backend Helpers";
  py::class_<OpaqueBlob, std::shared_ptr<OpaqueBlob>>(m, "OpaqueBlob");

  auto compiler_m = m.def_submodule("compiler", "IREE compiler support");
  SetupCompilerBindings(compiler_m);

  auto hal_m = m.def_submodule("hal", "IREE HAL support");
  SetupHalBindings(hal_m);

  auto rt_m = m.def_submodule("rt", "IREE RT api");
  SetupRtBindings(rt_m);

  auto vm_m = m.def_submodule("vm", "IREE VM api");
  SetupVmBindings(vm_m);
}

}  // namespace python
}  // namespace iree
