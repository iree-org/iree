// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/mlir_interop.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

namespace {

class GlobalInitializer {
public:
  GlobalInitializer() { ireeCompilerGlobalInitialize(); }
  ~GlobalInitializer() { ireeCompilerGlobalShutdown(); }
};

} // namespace

PYBIND11_MODULE(_site_initialize_0, m) {
  m.doc() = "iree-compile registration";

  // Make sure that GlobalInitialize and GlobalShutdown are called with module
  // lifetime.
  py::class_<GlobalInitializer>(m, "_GlobalInitializer");
  m.attr("_global_init_hook") =
      py::cast(new GlobalInitializer, py::return_value_policy::take_ownership);

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    ireeCompilerRegisterDialects(registry);
  });

  m.def("context_init_hook",
        [](MlirContext context) { ireeCompilerInitializeContext(context); });

  // Multi-threading is configured as part of the context_init_hook and
  // not left to default MLIR heuristics.
  m.attr("disable_multithreading") = true;
}
