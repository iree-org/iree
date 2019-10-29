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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_TF_INTEROP_REGISTER_TENSORFLOW_H_
#define IREE_BINDINGS_PYTHON_PYIREE_TF_INTEROP_REGISTER_TENSORFLOW_H_

#include <string>

#include "bindings/python/pyiree/binding.h"

namespace iree {
namespace python {

void SetupTensorFlowBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_TF_INTEROP_REGISTER_TENSORFLOW_H_
