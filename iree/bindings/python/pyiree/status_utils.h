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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_STATUS_UTILS_H_
#define IREE_BINDINGS_PYTHON_PYIREE_STATUS_UTILS_H_

#include "iree/base/status.h"
#include "pybind11/pytypes.h"

namespace iree {
namespace python {

// Converts a failing status to a throwable exception, setting Python
// error information.
// Correct usage is something like:
//   if (!status.ok()) {
//     throw StatusToPyExc(status);
//   }
pybind11::error_already_set StatusToPyExc(const Status& status);

// Consumes a StatusOr<T>, returning an rvalue reference to the T if the
// status is ok(). Otherwise, throws an exception.
template <typename T>
T&& PyConsumeStatusOr(iree::StatusOr<T>&& sor) {
  if (sor.ok()) {
    return std::move(*sor);
  }
  throw StatusToPyExc(sor.status());
}

}  // namespace python
}  // namespace iree

namespace pybind11 {
namespace detail {}  // namespace detail
}  // namespace pybind11

#endif  // IREE_BINDINGS_PYTHON_PYIREE_STATUS_UTILS_H_
