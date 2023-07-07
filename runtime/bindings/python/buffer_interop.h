// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Nanobind removed buffer protocol interop in favor of a new and improved
// ndarray thingy. This thingy is mostly better for a subset of cases but is
// not great for just generically accessing chunks of memory. For cases where
// we do the latter (mapping files, etc), we just have some helpers over the
// low level Python buffer protocol to ease the transition.

#ifndef IREE_BINDINGS_PYTHON_IREE_BUFFER_INTEROP_H_
#define IREE_BINDINGS_PYTHON_IREE_BUFFER_INTEROP_H_

#include "./binding.h"

namespace iree::python {

// Represents a Py_buffer obtained via PyObject_GetBuffer() and terminated via
// PyBuffer_Release().
class PyBufferRequest {
 public:
  PyBufferRequest(py::object &exporter, int flags) {
    int rc = PyObject_GetBuffer(exporter.ptr(), &view_, flags);
    if (rc != 0) {
      throw py::python_error();
    }
  }
  ~PyBufferRequest() { PyBuffer_Release(&view_); }
  PyBufferRequest(const PyBufferRequest &) = delete;
  void operator=(const PyBufferRequest &) = delete;

  Py_buffer &view() { return view_; }

 private:
  Py_buffer view_;
};

}  // namespace iree::python

#endif  // IREE_BINDINGS_PYTHON_IREE_BUFFER_INTEROP_H_
