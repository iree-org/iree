// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Shardy/InputConversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::shardy {

void buildShardyInputConversionPassPipeline(OpPassManager &passManager) {
  // Strip sdy ops - for single device they're metadata only
  passManager.addPass(createStripShardyDialectPass());
}

void registerShardyInputConversionPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return createStripShardyDialectPass();
  });
}

} // namespace mlir::iree_compiler::shardy
