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

#include "iree/bindings/python/status_utils.h"

namespace iree {
namespace python {

namespace {

PyObject* StatusToPyExcClass(const Status& status) {
  switch (status.code()) {
    case StatusCode::kInvalidArgument:
      return PyExc_ValueError;
    case StatusCode::kOutOfRange:
      return PyExc_IndexError;
    case StatusCode::kUnimplemented:
      return PyExc_NotImplementedError;
    default:
      return PyExc_RuntimeError;
  }
}

}  // namespace

pybind11::error_already_set StatusToPyExc(const Status& status) {
  assert(!status.ok());
  PyErr_SetString(StatusToPyExcClass(status), status.error_message().c_str());
  return pybind11::error_already_set();
}

}  // namespace python
}  // namespace iree
