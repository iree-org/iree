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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

/// Helper function to create `AffineExpr` from `OpFoldResult`. If the
/// `OpFoldResult` is a `Value`, creates a `AffineSymbolExpr` and appends it to
/// `symbols`.
static AffineExpr getAffineExpr(OpFoldResult ofr, SmallVector<Value> &symbols) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    return getAffineConstantExpr(attr.cast<IntegerAttr>().getInt(),
                                 attr.getContext());
  }
  Value v = ofr.get<Value>();
  AffineExpr expr = getAffineSymbolExpr(symbols.size(), v.getContext());
  symbols.push_back(v);
  return expr;
}
/// Converts an `AffineExpr` to `OpFoldResult` by generating an `affine.apply`
/// operation.
static OpFoldResult getOpFoldResult(OpBuilder &builder, Location loc,
                                    AffineExpr expr,
                                    SmallVector<Value> &symbols) {
  AffineMap m = AffineMap::get(0, symbols.size(), expr);
  return applyMapToValues(builder, loc, m, symbols)[0];
}

/// Methods to build the Affine Expr for arithmetic operations.
static AffineExpr add(AffineExpr expr, OpFoldResult ofr,
                      SmallVector<Value> &symbols) {
  return expr + getAffineExpr(ofr, symbols);
}
static AffineExpr add(OpFoldResult lhs, OpFoldResult rhs,
                      SmallVector<Value> &symbols) {
  return getAffineExpr(lhs, symbols) + getAffineExpr(rhs, symbols);
}
static AffineExpr mul(AffineExpr expr, OpFoldResult ofr,
                      SmallVector<Value> &symbols) {
  return expr * getAffineExpr(ofr, symbols);
}
static AffineExpr mul(OpFoldResult lhs, OpFoldResult rhs,
                      SmallVector<Value> &symbols) {
  return getAffineExpr(lhs, symbols) * getAffineExpr(rhs, symbols);
}

/// Returns the offsets to use when combining two operations that implement the
/// `OffsetSizeAndStrideOpInterface`. Also checks that the strides are 1.
static LogicalResult foldOffsetsSizesAndStrides(
    PatternRewriter &rewriter, Location loc,
    OffsetSizeAndStrideOpInterface producer,
    OffsetSizeAndStrideOpInterface consumer,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  SmallVector<OpFoldResult> producerOffsets = producer.getMixedOffsets();
  SmallVector<OpFoldResult> producerStrides = producer.getMixedStrides();
  SmallVector<OpFoldResult> consumerOffsets = consumer.getMixedOffsets();
  SmallVector<OpFoldResult> consumerStrides = consumer.getMixedStrides();
  if (producerOffsets.size() != consumerOffsets.size()) {
    return rewriter.notifyMatchFailure(
        consumer,
        "expected op and producer to have same number of offset values");
  }

  combinedOffsets.resize(producerOffsets.size());
  combinedSizes.resize(producerOffsets.size());
  combinedStrides.resize(producerOffsets.size());
  for (auto i : llvm::seq<unsigned>(0, producerOffsets.size())) {
    SmallVector<Value> offsetSymbols, strideSymbols;
    // The combined offset is computed as
    //    producer_offset + consumer_offset * producer_strides.
    combinedOffsets[i] = getOpFoldResult(
        rewriter, loc,
        add(mul(consumerOffsets[i], producerStrides[i], offsetSymbols),
            producerOffsets[i], offsetSymbols),
        offsetSymbols);
    // The combined stride is computed as
    //    producer_stride * consumer_stride.
    combinedStrides[i] = getOpFoldResult(
        rewriter, loc,
        mul(producerStrides[i], consumerStrides[i], strideSymbols),
        strideSymbols);
  }
  combinedSizes = consumer.getMixedSizes();
  return success();
}

/// Returns the `hal.interface.binding` a value comes from.
static Optional<IREE::HAL::InterfaceBindingSubspanOp> getBindingSubspanOp(
    Value v) {
  Operation *definingOp = v.getDefiningOp();
  if (!definingOp) return llvm::None;
  if (auto interfaceOp =
          dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(definingOp)) {
    return interfaceOp;
  }
  if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(definingOp)) {
    return getBindingSubspanOp(loadOp.source());
  }
  return llvm::None;
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

    SmallVector<OpFoldResult> offsets, sizes, strides;
    if (failed(foldOffsetsSizesAndStrides(
            rewriter, dispatchTensorLoadOp->getLoc(), dispatchTensorLoadOp,
            extractSliceOp, offsets, sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        extractSliceOp, extractSliceOp.getType(), dispatchTensorLoadOp.source(),
        dispatchTensorLoadOp.source_dims(), offsets, sizes, strides);
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

    // Check that the `dest` of the `tensor.insert_slice` and target of the
    // `flow.dispatch.tensor.store` are the same interface binding.
    Optional<IREE::HAL::InterfaceBindingSubspanOp> destBinding =
        getBindingSubspanOp(insertSliceOp.dest());
    Optional<IREE::HAL::InterfaceBindingSubspanOp> targetBinding =
        getBindingSubspanOp(dispatchTensorStoreOp.target());
    if (!destBinding || !targetBinding ||
        destBinding.getValue() != targetBinding.getValue()) {
      return failure();
    }

    SmallVector<OpFoldResult> offsets, sizes, strides;
    // Treat the `flow.dispatch.tensor.store` as the producer and the
    // `tensor.insert_slice` as the consumer since that would be the case for
    // the final subview created.
    if (failed(foldOffsetsSizesAndStrides(
            rewriter, dispatchTensorStoreOp->getLoc(), dispatchTensorStoreOp,
            insertSliceOp, offsets, sizes, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        dispatchTensorStoreOp, insertSliceOp.source(),
        dispatchTensorStoreOp.target(), dispatchTensorStoreOp.target_dims(),
        offsets, sizes, strides);
    return success();
  }
};

/// Pass to bufferize early copy-only dispatches. This allows backends
/// to use the `linalg.generic` operation generated for lowering the dispatch.
struct BufferizeCopyOnlyDispatchesPass
    : public BufferizeCopyOnlyDispatchesBase<BufferizeCopyOnlyDispatchesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, bufferization::BufferizationDialect,
                    IREE::Flow::FlowDialect, linalg::LinalgDialect,
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
          return success(isReadOnly(storeOp.value()));
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
