// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace mlir {
// Forward declarations
class DialectRegistry;
}  // namespace mlir

namespace mlir::iree_compiler {

// Forward declarations to make the API dependency free.
class OptionsBinder;

// Interface used by a plugin to register compiler extensions that it wishes
// to exploit. Each instance of the registrar is scoped to a single plugin
// and only exists for the life of the call to the registration function.
class PluginRegistrar {
 public:
  PluginRegistrar(DialectRegistry &dialectRegistry)
      : dialectRegistry(dialectRegistry) {}

  // Get the global dialect registry and register anything that is needed.
  DialectRegistry &getDialectRegistry();

 private:
  DialectRegistry &dialectRegistry;
};

}  // namespace mlir::iree_compiler

// Registration functions are exported with this signature.
using PluginRegistrationFunction =
    bool (*)(mlir::iree_compiler::PluginRegistrar *);
