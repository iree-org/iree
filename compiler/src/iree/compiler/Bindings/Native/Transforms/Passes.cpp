// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace ABI {

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
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());
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

}  // namespace ABI
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
