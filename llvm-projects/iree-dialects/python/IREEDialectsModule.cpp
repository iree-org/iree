// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Dialects.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

PYBIND11_MODULE(_ireeDialects, m) {
  m.doc() = "iree-dialects main python extension";

  m.def(
      "register_iree_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__iree__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
}
