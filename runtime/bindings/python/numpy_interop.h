// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_NUMPY_INTEROP_H_
#define IREE_BINDINGS_PYTHON_NUMPY_INTEROP_H_

#include "./binding.h"
#include "iree/hal/api.h"

namespace iree::python::numpy {

// Creates a numpy.dtype object for an IREE element type.
py::object DescrNewFromType(iree_hal_element_type_t t);

// Creates a numpy array from data using numpy.frombuffer and reshapes it.
// base_object is kept alive by the returned array via its memoryview base.
py::object SimpleNewFromData(int nd, intptr_t const* dims,
                             py::handle dtype_descr, void* data,
                             py::handle base_object);

}  // namespace iree::python::numpy

#endif  // IREE_BINDINGS_PYTHON_NUMPY_INTEROP_H_
