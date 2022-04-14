// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_MHLO_PASSES_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_MHLO_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace MHLO {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

void buildMHLOImportPassPipeline(OpPassManager &pm);
void registerMHLOImportPassPipeline();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

inline void registerAllPasses() { registerMHLOImportPassPipeline(); }

}  // namespace MHLO
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_MHLO_PASSES_H_
