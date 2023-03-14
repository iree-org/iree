// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/MHLO/Passes.h"

#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_integrations {
namespace MHLO {

void buildMHLOImportPassPipeline(OpPassManager &pm) {
  // We run the inliner for legacy reasons. It shouldn't be necessary anymore,
  // but this entire pipeline will soon be deleted and it isn't worth
  // removing now.
  pm.addPass(mlir::createInlinerPass());

  // Import pipelines should end with canonicalization because they may have
  // access to dialects and patterns that the core compiler does not.
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
}

void registerMHLOImportPassPipeline() {
  mlir::PassPipelineRegistration<> pipeline(
      "iree-mhlo-import-pipeline",
      "Run IREE-specific passes for importing MHLO code into IREE",
      [](OpPassManager &passManager) {
        buildMHLOImportPassPipeline(passManager);
      });
}

}  // namespace MHLO
}  // namespace iree_integrations
}  // namespace mlir
