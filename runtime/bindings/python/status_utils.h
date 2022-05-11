// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_COMMON_STATUS_UTILS_H_
#define IREE_BINDINGS_PYTHON_IREE_COMMON_STATUS_UTILS_H_

#include "iree/base/api.h"
#include "pybind11/pybind11.h"

namespace iree {
namespace python {

// Raises a value error with the given message.
// Correct usage:
//   throw RaiseValueError(PyExc_ValueError, "Foobar'd");
pybind11::error_already_set RaisePyError(PyObject* exc_class,
                                         const char* message);

// Raises a value error with the given message.
// Correct usage:
//   throw RaiseValueError("Foobar'd");
inline pybind11::error_already_set RaiseValueError(const char* message) {
  return RaisePyError(PyExc_ValueError, message);
}

pybind11::error_already_set ApiStatusToPyExc(iree_status_t status,
                                             const char* message);

inline void CheckApiStatus(iree_status_t status, const char* message) {
  if (iree_status_is_ok(status)) {
    return;
  }
  throw ApiStatusToPyExc(status, message);
}

inline void CheckApiNotNull(const void* p, const char* message) {
  if (!p) {
    throw RaiseValueError(message);
  }
}

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_COMMON_STATUS_UTILS_H_
