// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- BufferizeCopyOnlyDispatchesPassPass.cpp ----------------------------===//
//
// This pass converts dispatches that are copy only into a form where backends
// can tile and distribute them appropriately.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

/// Pass to bufferize early copy-only dispatches. This allows backends
/// to use the `linalg.generic` operation generated for lowering the dispatch.
struct BufferizeCopyOnlyDispatchesPass
    : public BufferizeCopyOnlyDispatchesBase<BufferizeCopyOnlyDispatchesPass> {
  BufferizeCopyOnlyDispatchesPass() = default;
  BufferizeCopyOnlyDispatchesPass(const BufferizeCopyOnlyDispatchesPass &pass) {
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, bufferization::BufferizationDialect,
                    IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void BufferizeCopyOnlyDispatchesPass::runOnOperation() {
  ModuleOp module = getOperation();

  SmallVector<Operation *> copyOnlyFunctions;
  auto funcOps = module.getOps<mlir::FunctionOpInterface>();
  for (auto funcOp : funcOps) {
    /// Check if the dispatch has all sources for `flow.dispatch.tensor.store`
    /// operations coming from `flow.dispatch.tensor.load` operations. If so,
    /// this dispatch is just a copy dispatch.
    bool hasFlowDispatchStore = false;
    auto walkResult = funcOp.walk(
        [&](IREE::Flow::DispatchTensorStoreOp storeOp) -> WalkResult {
          hasFlowDispatchStore = true;
          return success(isReadOnly(storeOp.getValue()));
        });
    if (walkResult.wasInterrupted())
      continue;
    // The function is just a copy and is not yet bufferized.
    if (hasFlowDispatchStore)
      copyOnlyFunctions.push_back(funcOp);
  }

  // There are no copy-only functions. So nothing to do.
  if (copyOnlyFunctions.empty())
    return;

  // Bufferize the dispatch to create a `linalg.generic` as a copy operation.
  // This can then be used by the backends to tile and distribute.
  // Currently bufferization does not handle single function bufferization. So
  // check that all functions are copy only and can be bufferized.
  if (copyOnlyFunctions.size() !=
      std::distance(funcOps.begin(), funcOps.end())) {
    module.emitOpError(
        "module contains functions that are both copy only and not copy only. "
        "This is currently unhandled.");
    return signalPassFailure();
  }

  // Apply the bufferization passes.
  OpPassManager bufferizationPipeline(module.getOperationName());
  // The copy-only dispatch shouldnt need an allocation. Error out on
  // allocation.
  bufferization::BufferizationOptions::AllocationFn allocationFn =
      [](OpBuilder &, Location loc, MemRefType, ValueRange,
         unsigned int) -> FailureOr<Value> {
    return emitError(
        loc, "unexpected allocation while bufferizing copy only dispatches");
  };
  bufferization::BufferizationOptions::MemCpyFn memcpyFn =
      [](OpBuilder &builder, Location loc, Value from,
         Value to) -> LogicalResult {
    createLinalgCopyOp(builder, loc, from, to);
    return success();
  };

  addIREEComprehensiveBufferizePasses(bufferizationPipeline, allocationFn,
                                      memcpyFn);
  if (failed(runPipeline(bufferizationPipeline, module))) {
    return signalPassFailure();
  }

  // Check that there are no allocs created.
  auto hasAlloc = module.walk(
      [&](memref::AllocOp /*op*/) -> WalkResult { return failure(); });
  if (hasAlloc.wasInterrupted()) {
    module.emitOpError(
        "unexpected allocations while bufferizing copy dispatch");
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createBufferizeCopyOnlyDispatchesPass() {
  return std::make_unique<BufferizeCopyOnlyDispatchesPass>();
}

} // namespace mlir::iree_compiler
