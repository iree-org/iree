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

#include <mutex>  // NOLINT

#include "bindings/python/pyiree/common/binding.h"
#include "bindings/python/pyiree/compiler/compiler.h"
#include "bindings/python/pyiree/compiler/tf_interop/register_tensorflow.h"

namespace iree {
namespace python {

PYBIND11_MODULE(binding, m) {
  m.doc() = "IREE Compiler Interface";
  py::class_<OpaqueBlob, std::shared_ptr<OpaqueBlob>>(m, "OpaqueBlob",
                                                      py::buffer_protocol())
      .def_buffer([](OpaqueBlob* self) -> py::buffer_info {
        return py::buffer_info(
            self->data(),                           // Pointer to buffer
            sizeof(uint8_t),                        // Size of one scalar
            py::format_descriptor<uint8_t>::value,  // Python struct-style
                                                    // format
            1,                                      // Number of dimensions
            {self->size()},                         // Buffer dimensions
            {self->size()}                          // Strides
        );
      })
      .def_property_readonly("bytes",
                             [](OpaqueBlob* self) -> py::bytes {
                               return py::bytes(
                                   static_cast<const char*>(self->data()),
                                   self->size());
                             })
      .def_property_readonly("text", [](OpaqueBlob* self) -> py::str {
        return py::str(static_cast<const char*>(self->data()), self->size());
      });

  SetupCompilerBindings(m);

// TensorFlow.
#if defined(IREE_TENSORFLOW_ENABLED)
  auto tf_m = m.def_submodule("tf_interop", "IREE TensorFlow interop");
  SetupTensorFlowBindings(tf_m);
#endif
}

}  // namespace python
}  // namespace iree
