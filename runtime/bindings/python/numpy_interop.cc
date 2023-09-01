// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./numpy_interop.h"

#include "./binding.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

namespace iree::python::numpy {

namespace {

int internal_import_array() {
  import_array1(-1);
  return 0;
}

}  // namespace

void InitializeNumPyInterop() {
  if (internal_import_array() < 0) {
    throw py::import_error("numpy.core.multiarray failed to import");
  }
}

int ConvertHalElementTypeToNumPyTypeNum(iree_hal_element_type_t t) {
  switch (t) {
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:
      return NPY_BOOL;
    case IREE_HAL_ELEMENT_TYPE_INT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      return NPY_INT8;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return NPY_UINT8;
    case IREE_HAL_ELEMENT_TYPE_INT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      return NPY_INT16;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return NPY_UINT16;
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return NPY_INT32;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return NPY_UINT32;
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return NPY_INT64;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return NPY_UINT64;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return NPY_FLOAT16;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return NPY_FLOAT32;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return NPY_FLOAT64;
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      return NPY_COMPLEX64;
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      return NPY_COMPLEX128;
    default:
      throw py::value_error("Unsupported VM Buffer -> numpy dtype mapping");
  }
}

py::object DescrNewFromType(int typenum) {
  PyArray_Descr *dtype = PyArray_DescrNewFromType(typenum);
  if (!dtype) {
    throw py::python_error();
  }
  return py::steal((PyObject *)dtype);
}

int TypenumFromDescr(py::handle dtype) {
  if (!PyArray_DescrCheck(dtype.ptr())) {
    throw py::cast_error();
  }
  PyArray_Descr *descr = (PyArray_Descr *)dtype.ptr();
  return descr->type_num;
}

py::object SimpleNewFromData(int nd, intptr_t const *dims, int typenum,
                             void *data, py::handle base_object) {
  PyObject *array_c = PyArray_SimpleNewFromData(nd, dims, typenum, data);
  if (!array_c) throw py::python_error();
  py::object array = py::steal(array_c);
  if (base_object) {
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(array.ptr()),
                              base_object.ptr())) {
      throw py::python_error();
    }
    base_object.inc_ref();
  }
  return array;
}

}  // namespace iree::python::numpy
