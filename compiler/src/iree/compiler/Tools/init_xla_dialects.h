// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.

#ifndef IREE_COMPILER_TOOLS_INIT_XLA_DIALECTS_H_
#define IREE_COMPILER_TOOLS_INIT_XLA_DIALECTS_H_

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the XLA dialects to the provided registry.
inline void registerXLADialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::chlo::ChloDialect,
                  mlir::mhlo::MhloDialect>();
  // clang-format on
}

}  // namespace mlir

#endif  // IREE_COMPILER_TOOLS_INIT_XLA_DIALECTS_H_
