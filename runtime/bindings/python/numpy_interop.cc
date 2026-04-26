// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./numpy_interop.h"

#include "./binding.h"

namespace iree::python::numpy {

static const char* ConvertHalElementTypeToDtypeName(iree_hal_element_type_t t) {
  switch (t) {
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:
      return "bool";
    case IREE_HAL_ELEMENT_TYPE_INT_8:
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      return "int8";
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return "uint8";
    case IREE_HAL_ELEMENT_TYPE_INT_16:
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      return "int16";
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return "uint16";
    case IREE_HAL_ELEMENT_TYPE_INT_32:
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return "int32";
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return "uint32";
    case IREE_HAL_ELEMENT_TYPE_INT_64:
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return "int64";
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return "uint64";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return "float16";
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      return "bfloat16";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return "float32";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return "float64";
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      return "complex64";
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      return "complex128";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2:
      return "float8_e5m2";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN:
      return "float8_e4m3fn";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2_FNUZ:
      return "float8_e5m2fnuz";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FNUZ:
      return "float8_e4m3fnuz";
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E8M0_FNU:
      return "float8_e8m0fnu";
    default:
      throw py::value_error("Unsupported VM Buffer -> numpy dtype mapping");
  }
}

py::object DescrNewFromType(iree_hal_element_type_t t) {
  const char* name = ConvertHalElementTypeToDtypeName(t);
  // import_() is a sys.modules dict lookup, not a full import. Could cache
  // the module reference if this becomes a hot path.
  return py::module_::import_("numpy").attr("dtype")(name);
}

py::object SimpleNewFromData(int nd, intptr_t const* dims,
                             py::handle dtype_descr, void* data) {
  int itemsize = py::cast<int>(dtype_descr.attr("itemsize"));
  Py_ssize_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    total_elems *= dims[i];
  }
  Py_ssize_t byte_len = total_elems * itemsize;

  // Create a writable memoryview over the raw data. Lifetime of the
  // underlying memory is managed by the caller via py::keep_alive on
  // the binding that calls this function, not by the memoryview itself.
  py::object buf = py::steal(
      PyMemoryView_FromMemory(static_cast<char*>(data), byte_len, PyBUF_WRITE));
  if (!buf.ptr()) throw py::python_error();

  // import_() is a sys.modules dict lookup, not a full import. Could cache
  // the module reference if this becomes a hot path.
  py::object array = py::module_::import_("numpy").attr("frombuffer")(
      buf, py::arg("dtype") = dtype_descr);

  // Reshape if needed (frombuffer always returns 1-D).
  if (nd != 1) {
    PyObject* shape_raw = PyTuple_New(nd);
    if (!shape_raw) throw py::python_error();
    for (int i = 0; i < nd; ++i) {
      PyObject* item = PyLong_FromSsize_t(dims[i]);
      if (!item) {
        Py_DECREF(shape_raw);
        throw py::python_error();
      }
      PyTuple_SetItem(shape_raw, i, item);
    }
    py::object shape_tuple = py::steal(shape_raw);
    array = array.attr("reshape")(shape_tuple);
  }

  return array;
}

}  // namespace iree::python::numpy
