// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_INIT_TORCH_MLIR_DIALECTS_H_
#define IREE_COMPILER_TOOLS_INIT_TORCH_MLIR_DIALECTS_H_

#include "mlir/IR/Dialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"

namespace mlir {

// Add all the torch mlir dialects to the provided registry.
inline void registerTorchMLIRDialects(DialectRegistry &registry) {
  registry.insert<mlir::torch::TMTensor::TMTensorDialect>();
}

}  // namespace mlir

#endif  // IREE_COMPILER_TOOLS_INIT_TORCH_MLIR_DIALECTS_H_
