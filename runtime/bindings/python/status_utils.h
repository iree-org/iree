// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_COMMON_STATUS_UTILS_H_
#define IREE_BINDINGS_PYTHON_IREE_COMMON_STATUS_UTILS_H_

#include "iree/base/api.h"
#include "nanobind/nanobind.h"

namespace iree {
namespace python {

// Raises a value error with the given message.
// Correct usage:
//   throw RaiseValueError(PyExc_ValueError, "Foobar'd");
nanobind::python_error RaisePyError(PyObject* exc_class, const char* message);

// Raises a value error with the given message.
// Correct usage:
//   throw RaiseValueError("Foobar'd");
inline nanobind::python_error RaiseValueError(const char* message) {
  return RaisePyError(PyExc_ValueError, message);
}

std::string ApiStatusToString(iree_status_t status);

nanobind::python_error ApiStatusToPyExc(iree_status_t status,
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
