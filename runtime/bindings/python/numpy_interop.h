// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_NUMPY_INTEROP_H_
#define IREE_BINDINGS_PYTHON_NUMPY_INTEROP_H_

#include "./binding.h"
#include "iree/hal/api.h"

namespace iree {
namespace python {

// Must be called in init of extension module.
void InitializeNumPyInterop();

// Converts an IREE element type to a NumPy NPY_TYPES value.
int ConvertHalElementTypeToNumPyTypeNum(iree_hal_element_type_t t);

// Delegates to PyArray_SimpleNewFromData and sets the base_object.
py::object SimpleNewFromData(int nd, intptr_t const *dims, int typenum,
                             void *data, py::handle base_object);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_NUMPY_INTEROP_H_
