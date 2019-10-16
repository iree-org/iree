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

#include "iree/bindings/python/binding.h"

#include "iree/bindings/python/compiler.h"
#include "iree/bindings/python/status_utils.h"

namespace iree {
namespace python {

PYBIND11_MODULE(binding, m) {
  m.doc() = "IREE Binding Backend Helpers";

  auto compiler_m = m.def_submodule("compiler", "IREE compiler support");
  py::class_<MemoryModuleFile, std::shared_ptr<MemoryModuleFile>>(
      compiler_m, "MemoryModuleFile")
      .def(py::init<>());
  compiler_m.def("compile_module_from_asm", CompileModuleFromAsm);
}

}  // namespace python
}  // namespace iree
