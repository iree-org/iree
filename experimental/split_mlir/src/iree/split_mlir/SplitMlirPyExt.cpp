// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "OperationListImpl.h"

namespace py = pybind11;

namespace mlir {
namespace iree {
namespace split_mlir {

PYBIND11_MODULE(_split_mlir, m) {
  m.doc() = "Split MLIR C++ extension";

  m.def(
      "extract_operation_list",
      [](const std::string& mlirFilePath, const std::string& functionName) {
        auto res = extractOperationList(mlirFilePath.c_str(), functionName);
        if (failed(res)) {
          throw std::runtime_error("");
        }
        return res.value();
      },
      py::arg("mlir_file_path"), py::arg("function_name"));
}

}  // namespace split_mlir
}  // namespace iree
}  // namespace mlir
