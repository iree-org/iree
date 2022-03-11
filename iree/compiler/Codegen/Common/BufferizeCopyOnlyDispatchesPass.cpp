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

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

/// Adds to `OpFoldResult`s and returns the result as an `OpFoldResult`.
static OpFoldResult add(OpBuilder &builder, Location loc, OpFoldResult lhs,
                        OpFoldResult rhs) {
  auto lhsAttr = lhs.dyn_cast<Attribute>();
  auto rhsAttr = rhs.dyn_cast<Attribute>();
  if (lhsAttr && rhsAttr) {
    int64_t result = lhsAttr.cast<IntegerAttr>().getInt() +
                     rhsAttr.cast<IntegerAttr>().getInt();
    return builder.getIndexAttr(result);
  }
  // Generate the affine.apply that computes the result
  SmallVector<Value> operands;
  AffineExpr resultExpr = nullptr;
  auto addToResult = [&](OpFoldResult ofr) {
    AffineExpr e;
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      e = getAffineConstantExpr(attr.cast<IntegerAttr>().getInt(),
                                builder.getContext());
    } else {
      e = getAffineSymbolExpr(operands.size(), builder.getContext());
      operands.push_back(ofr.get<Value>());
    }
    resultExpr = resultExpr ? resultExpr + e : e;
  };
  addToResult(lhs);
  addToResult(rhs);
  AffineMap map = AffineMap::get(0, operands.size(), resultExpr);
  return builder.create<AffineApplyOp>(loc, map, operands).getResult();
}

/// Returns the offsets to use when combining two operations that implement the
/// `OffsetSizeAndStrideOpInterface`. Also checks that the strides are 1.
static FailureOr<SmallVector<OpFoldResult>> foldOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, OffsetSizeAndStrideOpInterface producer,
    OffsetSizeAndStrideOpInterface consumer) {
  auto checkOne = [](OpFoldResult ofr) -> bool {
    auto attr = ofr.dyn_cast<Attribute>();
    return attr && attr.cast<IntegerAttr>().getInt() == 1;
  };
  auto producerStrides = producer.getMixedStrides();
  auto consumerStrides = consumer.getMixedStrides();
  if (producerStrides.size() != consumerStrides.size()) {
    return static_cast<LogicalResult>(producer->emitOpError(
        "expected same number of offsets/sizes/strides for producer and "
        "consumer"));
  }

  if (!llvm::all_of(producer.getMixedStrides(), checkOne)) {
    return static_cast<LogicalResult>(
        producer->emitOpError("expected all strides to be 1"));
  }
  if (!llvm::all_of(consumer.getMixedStrides(), checkOne)) {
    return static_cast<LogicalResult>(
        consumer->emitOpError("expected all strides to be 1"));
  }

  // Combined offsets is the addition of the two offsets.
  return llvm::to_vector(llvm::map_range(
      llvm::zip(producer.getMixedOffsets(), consumer.getMixedOffsets()),
      [&](std::tuple<OpFoldResult, OpFoldResult> t) {
        return add(builder, loc, std::get<0>(t), std::get<1>(t));
      }));
}

/// Returns true if the `v` is from a `flow.dispatch.tensor.load` operation.
static bool isDirectlyFromDispatchTensorLoad(Value v) {
  // Might eventually need to walk the use-def chain a bit, but for now,
  // just check for the value defined by a flow.dispatch.tensor.load.
  return v.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>() != nullptr;
}

namespace {

/// Pattern to fold `flow.dispatch.tensor.load` -> `tensor.extract_slice`.
// TODO(ravishankarm): Eventually this should go in as a canonicalization at the
// Flow level.
struct FoldTensorLoadWithExtractSlice
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    auto dispatchTensorLoadOp =
        extractSliceOp.source()
            .getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!dispatchTensorLoadOp) return failure();

    FailureOr<SmallVector<OpFoldResult>> offsets =
        foldOffsetsSizesAndStrides(rewriter, dispatchTensorLoadOp->getLoc(),
                                   dispatchTensorLoadOp, extractSliceOp);
    if (failed(offsets)) {
      return failure();
    }

    SmallVector<OpFoldResult> strides(offsets->size(),
                                      rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        extractSliceOp, extractSliceOp.getType(), dispatchTensorLoadOp.source(),
        dispatchTensorLoadOp.source_dims(), offsets.getValue(),
        extractSliceOp.getMixedSizes(), strides);
    return success();
  }
};

/// Pattern to fold `tensor.insert_slice` with `flow.dispatch.tensor.store`
/// oeprations.
// TODO(ravishankarm): Eventually this should go in as a canonicalization at the
// Flow level.
struct FoldInsertSliceWithTensorStoreOp
    : OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorStoreOp dispatchTensorStoreOp,
      PatternRewriter &rewriter) const override {
    auto insertSliceOp =
        dispatchTensorStoreOp.value().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) return failure();

    FailureOr<SmallVector<OpFoldResult>> offsets =
        foldOffsetsSizesAndStrides(rewriter, dispatchTensorStoreOp->getLoc(),
                                   insertSliceOp, dispatchTensorStoreOp);
    if (failed(offsets)) {
      return failure();
    }

    SmallVector<OpFoldResult> strides(offsets->size(),
                                      rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        dispatchTensorStoreOp, insertSliceOp.source(),
        dispatchTensorStoreOp.target(), dispatchTensorStoreOp.target_dims(),
        offsets.getValue(), insertSliceOp.getMixedSizes(), strides);
    return success();
  }
};

/// Pass to bufferize early copy-only dispatches. This allows backends
/// to use the `linalg.generic` operation generated for lowering the dispatch.
struct BufferizeCopyOnlyDispatchesPass
    : public BufferizeCopyOnlyDispatchesBase<BufferizeCopyOnlyDispatchesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                memref::MemRefDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void BufferizeCopyOnlyDispatchesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  /// First apply the `flow.dispatch.tensor.load` -> `tensor.extract_slice` and
  /// `tensor.insert_slice` -> `flow.dispatch.tensor.store` patterns.
  RewritePatternSet patterns(context);
  patterns
      .insert<FoldInsertSliceWithTensorStoreOp, FoldTensorLoadWithExtractSlice>(
          context);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    return signalPassFailure();
  }

  SmallVector<Operation *> copyOnlyFunctions;
  auto funcOps = module.getOps<FuncOp>();
  for (auto funcOp : funcOps) {
    /// Check if the dispatch has all sources for `flow.dispatch.tensor.store`
    /// operations coming from `flow.dispatch.tensor.load` operations. If so,
    /// this dispatch is just a copy dispatch.
    auto walkResult = funcOp.walk(
        [&](IREE::Flow::DispatchTensorStoreOp storeOp) -> WalkResult {
          return success(isDirectlyFromDispatchTensorLoad(storeOp.value()));
        });
    if (walkResult.wasInterrupted()) continue;
    // The function is just a copy.
    copyOnlyFunctions.push_back(funcOp);
  }

  // There are no copy-only functions. So nothing to do.
  if (copyOnlyFunctions.empty()) return;

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
  addLinalgBufferizePasses(bufferizationPipeline);
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

}  // namespace iree_compiler
}  // namespace mlir
