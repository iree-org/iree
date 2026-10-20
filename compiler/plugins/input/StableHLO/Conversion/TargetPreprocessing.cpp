// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/TargetPreprocessing.h"

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/TargetRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {

void buildTargetPreprocessingPassPipeline(OpPassManager &passManager,
                                          const TargetRegistry &registry) {
  passManager.addPass(createCheckVHLOStableHloMixUsage());
  mlir::stablehlo::createStablehloDeserializePipeline(passManager);
  passManager.addPass(createRewriteRegisteredTargetsPass(registry));
}

} // namespace mlir::iree_compiler::stablehlo
