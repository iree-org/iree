// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./status_utils.h"

namespace iree {
namespace python {

namespace {

PyObject* ApiStatusToPyExcClass(iree_status_t status) {
  switch (iree_status_code(status)) {
    case IREE_STATUS_INVALID_ARGUMENT:
      return PyExc_ValueError;
    case IREE_STATUS_NOT_FOUND:
      return PyExc_ValueError;
    case IREE_STATUS_OUT_OF_RANGE:
      return PyExc_IndexError;
    case IREE_STATUS_UNIMPLEMENTED:
      return PyExc_NotImplementedError;
    default:
      return PyExc_RuntimeError;
  }
}

}  // namespace

std::string ApiStatusToString(iree_status_t status) {
  iree_host_size_t buffer_length = 0;
  if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                        /*buffer=*/NULL, &buffer_length))) {
    return "";
  }
  std::string result;
  result.resize(buffer_length);
  // NOTE: buffer capacity needs to be +1 for the NUL terminator in snprintf.
  return iree_status_format(status, result.size() + 1,
                            const_cast<char*>(result.data()), &buffer_length)
             ? result
             : "";
}

nanobind::python_error ApiStatusToPyExc(iree_status_t status,
                                        const char* message) {
  assert(!iree_status_is_ok(status));
  std::string full_message;

  auto status_str = ApiStatusToString(status);
  if (status_str.empty()) {
    full_message = std::string(message) + ": " +
                   iree_status_code_string(iree_status_code(status));
  } else {
    full_message = std::string(message) + ": " + status_str;
  }

  PyErr_SetString(ApiStatusToPyExcClass(status), full_message.c_str());
  iree_status_ignore(status);
  return nanobind::python_error();
}

nanobind::python_error RaisePyError(PyObject* exc_class, const char* message) {
  PyErr_SetString(exc_class, message);
  return nanobind::python_error();
}

}  // namespace python
}  // namespace iree
