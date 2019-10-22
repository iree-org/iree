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

#include "iree/bindings/python/pyiree/vm.h"

#include "iree/bindings/python/pyiree/status_utils.h"

namespace iree {
namespace python {

RtModule CreateModuleFromBlob(std::shared_ptr<OpaqueBlob> blob) {
  iree_rt_module_t* module;
  auto free_fn = OpaqueBlob::CreateFreeFn(blob);
  auto status = iree_vm_bytecode_module_create_from_buffer(
      {static_cast<const uint8_t*>(blob->data()), blob->size()}, free_fn.first,
      free_fn.second, IREE_ALLOCATOR_DEFAULT, &module);
  CheckApiStatus(status, "Error creating vm module from blob");
  return RtModule::CreateRetained(module);
}

void SetupVmBindings(pybind11::module m) {
  m.def("create_module_from_blob", CreateModuleFromBlob);
}

}  // namespace python
}  // namespace iree
