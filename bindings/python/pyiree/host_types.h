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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_HOST_TYPES_H_
#define IREE_BINDINGS_PYTHON_PYIREE_HOST_TYPES_H_

#include "absl/types/span.h"
#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/hal.h"
#include "iree/base/signature_mangle.h"

namespace iree {
namespace python {

class HostTypeFactory {
 public:
  virtual ~HostTypeFactory() = default;

  // Creates a C-contiguous ndarray of the given element_type/dims and backed
  // by the given buffer. The resulting array has no synchronization and is
  // available for use immediately.
  virtual py::object CreateImmediateNdarray(
      AbiConstants::ScalarType element_type, absl::Span<int> dims,
      HalBuffer buffer);

  // TODO(laurenzo): Add a CreateDelayedNdarray() which is conditioned on
  // a semaphore. This is actually what should be used for async results.
};

void SetupHostTypesBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_HOST_TYPES_H_
