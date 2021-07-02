// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bindings/python/iree/runtime/status_utils.h"

namespace iree {
namespace python {

namespace {

PyObject* ApiStatusToPyExcClass(iree_status_t status) {
  switch (iree_status_code(status)) {
    case IREE_STATUS_INVALID_ARGUMENT:
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

pybind11::error_already_set ApiStatusToPyExc(iree_status_t status,
                                             const char* message) {
  assert(!iree_status_is_ok(status));
  std::string full_message;

  char* iree_message;
  size_t iree_message_length;
  if (iree_status_to_string(status, &iree_message, &iree_message_length)) {
    full_message = std::string(message) + ": " +
                   std::string(iree_message, iree_message_length);
    iree_allocator_free(iree_allocator_system(), iree_message);
  } else {
    full_message = std::string(message) + ": " +
                   iree_status_code_string(iree_status_code(status));
  }

  PyErr_SetString(ApiStatusToPyExcClass(status), full_message.c_str());
  iree_status_ignore(status);
  return pybind11::error_already_set();
}

pybind11::error_already_set RaisePyError(PyObject* exc_class,
                                         const char* message) {
  PyErr_SetString(exc_class, message);
  return pybind11::error_already_set();
}

}  // namespace python
}  // namespace iree
