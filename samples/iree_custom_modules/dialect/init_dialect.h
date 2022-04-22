// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of the custom
// dialect to the system.
//
// Imported from the iree-template-cpp project.

#ifndef IREE_SAMPLES_CUSTOM_MODULES_DIALECT_INIT_DIALECT_H_
#define IREE_SAMPLES_CUSTOM_MODULES_DIALECT_INIT_DIALECT_H_

#include "iree_custom_modules/dialect/custom_dialect.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

// Add custom dialect to the provided registry.
inline void registerCustomDialect(DialectRegistry &registry) {
  registry.insert<IREE::Custom::CustomDialect>();
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_SAMPLES_CUSTOM_MODULES_DIALECT_INIT_DIALECT_H_
