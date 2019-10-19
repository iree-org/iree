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
  py::class_<RtModule>(m, "Module")
      .def_property_readonly("name", &RtModule::name)
      .def("lookup_function_by_ordinal", &RtModule::lookup_function_by_ordinal)
      .def("lookup_function_by_name", &RtModule::lookup_function_by_name);
  py::class_<RtFunction>(m, "Function")
      .def_property_readonly("name", &RtFunction::name)
      .def_property_readonly("signature", &RtFunction::signature);
  py::class_<iree_rt_function_signature_t>(m, "FunctionSignature")
      .def_readonly("argument_count",
                    &iree_rt_function_signature_t::argument_count)
      .def_readonly("result_count",
                    &iree_rt_function_signature_t::result_count);
}

}  // namespace python
}  // namespace iree
