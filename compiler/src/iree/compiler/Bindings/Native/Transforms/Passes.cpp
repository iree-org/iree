// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::ABI {

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

void buildTransformPassPipeline(OpPassManager &passManager,
                                const InvocationOptions &invocationOptions) {
  // Convert streamable ops prior to wrapping. This lets us use the original
  // types on function boundaries prior to wrapping.
  passManager.addPass(createConvertStreamableOpsPass());

  // Wraps the entry points in an export function.
  passManager.addPass(
      createWrapEntryPointsPass(invocationOptions.invocationModel));

  // Cleanup the IR after manipulating it.
  passManager.addPass(createInlinerPass());
  FunctionLikeNest(passManager).addPass(createCanonicalizerPass);
  FunctionLikeNest(passManager).addPass(createCSEPass);
  passManager.addPass(createSymbolDCEPass());
}

void registerTransformPassPipeline() {
  PassPipelineRegistration<InvocationOptions> transformPassPipeline(
      "iree-abi-transformation-pipeline",
      "Runs the IREE native ABI bindings support pipeline",
      [](OpPassManager &passManager,
         const InvocationOptions &invocationOptions) {
        buildTransformPassPipeline(passManager, invocationOptions);
      });
}

} // namespace mlir::iree_compiler::IREE::ABI
