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

#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/compiler.h"
#include "bindings/python/pyiree/hal.h"
#include "bindings/python/pyiree/initialize.h"
#include "bindings/python/pyiree/rt.h"
#include "bindings/python/pyiree/status_utils.h"
#include "bindings/python/pyiree/tensorflow/register_tensorflow.h"
#include "bindings/python/pyiree/vm.h"

namespace iree {
namespace python {

PYBIND11_MODULE(binding, m) {
  m.doc() = "IREE Binding Backend Helpers";
  py::class_<OpaqueBlob, std::shared_ptr<OpaqueBlob>>(m, "OpaqueBlob");
  m.def("initialize_extension", &InitializeExtension);

  auto compiler_m = m.def_submodule("compiler", "IREE compiler support");
  SetupCompilerBindings(compiler_m);

  auto hal_m = m.def_submodule("hal", "IREE HAL support");
  SetupHalBindings(hal_m);

  auto rt_m = m.def_submodule("rt", "IREE RT api");
  SetupRtBindings(rt_m);

  auto vm_m = m.def_submodule("vm", "IREE VM api");
  SetupVmBindings(vm_m);

// TensorFlow.
#if defined(IREE_TENSORFLOW_ENABLED)
  auto tf_m = m.def_submodule("tf_interop", "IREE TensorFlow interop");
  SetupTensorFlowBindings(tf_m);
#endif
}

}  // namespace python
}  // namespace iree
