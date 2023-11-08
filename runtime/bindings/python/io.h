// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_RT_IO_PARAMETERS_H_
#define IREE_BINDINGS_PYTHON_IREE_RT_IO_PARAMETERS_H_

#include <vector>

#include "./binding.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"

namespace iree::python {

template <>
struct ApiPtrAdapter<iree_io_file_handle_t> {
  static void Retain(iree_io_file_handle_t *v) {
    iree_io_file_handle_retain(v);
  }
  static void Release(iree_io_file_handle_t *v) {
    iree_io_file_handle_release(v);
  }
};

template <>
struct ApiPtrAdapter<iree_io_parameter_provider_t> {
  static void Retain(iree_io_parameter_provider_t *v) {
    iree_io_parameter_provider_retain(v);
  }
  static void Release(iree_io_parameter_provider_t *v) {
    iree_io_parameter_provider_release(v);
  }
};

template <>
struct ApiPtrAdapter<iree_io_parameter_index_t> {
  static void Retain(iree_io_parameter_index_t *v) {
    iree_io_parameter_index_retain(v);
  }
  static void Release(iree_io_parameter_index_t *v) {
    iree_io_parameter_index_release(v);
  }
};

class FileHandle : public ApiRefCounted<FileHandle, iree_io_file_handle_t> {};

class ParameterProvider
    : public ApiRefCounted<ParameterProvider, iree_io_parameter_provider_t> {};

class ParameterIndex
    : public ApiRefCounted<ParameterIndex, iree_io_parameter_index_t> {};

void SetupIoBindings(py::module_ &m);

}  // namespace iree::python

#endif  // IREE_BINDINGS_PYTHON_IREE_RT_IO_PARAMETERS_H_