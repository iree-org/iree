// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/LLVMGPU/TransposeUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-tensor-pad"

namespace mlir {
namespace iree_compiler {

namespace {

static FailureOr<SmallVector<Value>> rewriteAsPaddedOp(
    IRRewriter &rewriter, linalg::LinalgOp linalgOp,
    linalg::LinalgOp &paddedOp) {
  Location loc = linalgOp.getLoc();

  IRRewriter::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(linalgOp);

  // Pad each input operand in shared memory up to the targets bounding box
  // size. In this case, this corresponds with the maximum tile size from
  // distributing to workgroups.
  SmallVector<Value> paddedOperands;
  paddedOperands.reserve(linalgOp.getNumDpsInputs() +
                         linalgOp.getNumDpsInits());
  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    // Find DispatchTensorLoadOp's feeding into the linalg or abort.
    auto tensorLoad = dyn_cast_or_null<IREE::Flow::DispatchTensorLoadOp>(
        opOperand.get().getDefiningOp());
    if (!tensorLoad) {
      return rewriter.notifyMatchFailure(linalgOp, "does not have tensor load");
    }

    // Determine the padded shape from the load
    ArrayRef<int64_t> shape = linalgOp.getShape(&opOperand);
    SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
    for (const auto &en : llvm::enumerate(tensorLoad.getMixedSizes())) {
      if (Optional<int64_t> cst = getConstantIntValue(en.value())) {
        paddedShape[en.index()] = cst.value();
      } else {
        FailureOr<int64_t> upperBound =
            linalg::getConstantUpperBoundForIndex(en.value().get<Value>());
        if (failed(upperBound)) {
          return rewriter.notifyMatchFailure(
              linalgOp, "No constant bounding box can be found for padding");
        }
        paddedShape[en.index()] = *upperBound;
      }
    }

    Value paddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(getElementTypeOrSelf(tensorLoad)));
    auto paddedTensorType =
        RankedTensorType::get(paddedShape, getElementTypeOrSelf(tensorLoad));
    Value paddedValue = linalg::makeComposedPadHighOp(
        rewriter, loc, paddedTensorType, tensorLoad, paddingValue,
        /*nofold=*/false);
    paddedOperands.push_back(paddedValue);
  }

  // Clone linalgOp to paddedOp with padded input/output shapes.
  auto resultTensorTypes = ValueRange(paddedOperands)
                               .take_back(linalgOp.getNumDpsInits())
                               .getTypes();
  paddedOp = linalgOp.clone(rewriter, loc, resultTensorTypes, paddedOperands);

  // Slice out the original shape from the padded result to pass on to
  // consumers. The original linalg op is used to provide the dims for the reify
  // result shapes.
  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(cast<ReifyRankedShapedTypeOpInterface>(linalgOp.getOperation())
                 .reifyResultShapes(rewriter, reifiedResultShapes))) {
    return failure();
  }

  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(paddedOp->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = paddedResult.getType().cast<RankedTensorType>().getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (Value v : reifiedResultShapes[resultNumber])
      sizes.push_back(getAsOpFoldResult(v));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubviewResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, sizes, strides));
  }
  return paddedSubviewResults;
}

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

struct LLVMGPUTensorPadPass
    : public LLVMGPUTensorPadBase<LLVMGPUTensorPadPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    IRRewriter rewriter(funcOp->getContext());
    rewriter.setInsertionPoint(funcOp);
    funcOp.walk([&](linalg::GenericOp linalgOp) {
      LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);
      // Checks preconditions for shared mem transpose. Only pad if op is
      // dynamic.
      if (!opInfo.isTranspose() || !opInfo.isDynamic() ||
          !hasTwoOrThreeLoopsInfo(linalgOp)) {
        funcOp.emitWarning("failed preconditions");
        return;
      }

      linalg::LinalgOp paddedOp;
      FailureOr<SmallVector<Value>> newResults =
          rewriteAsPaddedOp(rewriter, linalgOp, paddedOp);
      if (failed(newResults)) {
        return;
      }

      // Replace the original operation to pad.
      rewriter.replaceOp(linalgOp, *newResults);
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass() {
  return std::make_unique<LLVMGPUTensorPadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
