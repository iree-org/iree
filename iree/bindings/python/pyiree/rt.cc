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

#include "iree/bindings/python/pyiree/rt.h"

namespace iree {
namespace python {

void SetupRtBindings(pybind11::module m) {
  // BufferPlacement.
  py::enum_<BufferPlacement>(m, "BufferPlacement")
      .value("HEAP", BufferPlacement::kHeap)
      .value("DEVICE_VISIBLE", BufferPlacement::kDeviceVisible)
      .value("DEVICE_LOCAL", BufferPlacement::kDeviceLocal)
      .export_values();

  // RtModule.
  py::class_<RtModule>(m, "Module")
      .def_property_readonly("name", &RtModule::name)
      .def("lookup_function_by_ordinal", &RtModule::lookup_function_by_ordinal)
      .def("lookup_function_by_name", &RtModule::lookup_function_by_name);
  // RtFunction.
  py::class_<RtFunction>(m, "Function")
      .def_property_readonly("name", &RtFunction::name)
      .def_property_readonly("signature", &RtFunction::signature);
  py::class_<iree_rt_function_signature_t>(m, "FunctionSignature")
      .def_readonly("argument_count",
                    &iree_rt_function_signature_t::argument_count)
      .def_readonly("result_count",
                    &iree_rt_function_signature_t::result_count);

  // RtPolicy.
  py::class_<RtPolicy>(m, "Policy").def(py::init(&RtPolicy::Create));

  // RtInstance.
  py::class_<RtInstance>(m, "Instance")
      .def(py::init(&RtInstance::Create),
           py::arg_v("driver_name", absl::optional<std::string>()));

  // RtContext.
  py::class_<RtContext>(m, "Context")
      .def(py::init(&RtContext::Create), py::arg("instance"), py::arg("policy"))
      .def_property_readonly("context_id", &RtContext::context_id)
      .def("register_modules", &RtContext::RegisterModules, py::arg("modules"))
      .def("register_module", &RtContext::RegisterModule, py::arg("module"))
      .def("lookup_module_by_name", &RtContext::LookupModuleByName,
           py::arg("name"))
      .def("resolve_function", &RtContext::ResolveFunction,
           py::arg("full_name"))
      .def("allocate", &RtContext::Allocate, py::arg("allocation_size"),
           py::arg("placement") = BufferPlacement::kHeap,
           py::arg("usage") = IREE_HAL_BUFFER_USAGE_ALL)
      .def("allocate_device_local", &RtContext::AllocateDeviceLocal,
           py::arg("allocation_size"),
           py::arg("usage") = IREE_HAL_BUFFER_USAGE_ALL)
      .def("allocate_device_visible", &RtContext::AllocateDeviceVisible,
           py::arg("allocation_size"),
           py::arg("usage") = IREE_HAL_BUFFER_USAGE_ALL)
      .def("invoke", &RtContext::Invoke, py::arg("f"), py::arg("policy"),
           py::arg("arguments"), py::arg("results"));

  // RtInvocation.
  py::class_<RtInvocation>(m, "Invocation")
      .def("query_status", &RtInvocation::QueryStatus);
}

}  // namespace python
}  // namespace iree
