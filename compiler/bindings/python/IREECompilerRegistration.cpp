// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/API/MLIRInterop.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_site_initialize_0, m) {
  m.doc() = "iree-compile registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    ireeCompilerRegisterDialects(registry);
  });
}
