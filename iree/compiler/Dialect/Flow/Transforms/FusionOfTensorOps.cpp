// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionOfTensorsOps.cpp - Pass to fuse operations on tensors-------===//
//
// Pass to fuse operations on tensors after conversion to Linalg. Uses the
// patterns from MLIR for fusion linalg operations on tensors, and a few
// patterns to fuse these with IREE specific operations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

static llvm::cl::opt<bool> clEnableFusionWithReductionOps(
    "iree-enable-fusion-with-reduction-ops",
    llvm::cl::desc("Allow fusing generic ops with reductions"),
    llvm::cl::init(false));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

using linalg::LinalgOp;

/// Pass to fuse linalg on tensor operations as well as fusion of hal.interface*
/// operations with linalg.tensor_reshape operation.
struct FusionOfTensorOpsPass
    : public FusionOfTensorOpsBase<FusionOfTensorOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList fusionPatterns(&getContext());
    OwningRewritePatternList interfacePatterns(&getContext());
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    // Only fuse operations where all uses of the producer are generic
    // operations. If an operation is used in a named op, it will be computed
    // anyway, so the consumers can just use that value.
    linalg::ControlElementwiseOpsFusionFn controlFn =
        [](const OpResult &producerResult, OpOperand &consumerOperand) {
          Operation *producer = producerResult.getOwner();
          Operation *consumer = consumerOperand.getOwner();

          // TODO(#5611): Enable fusion with reduction consumer for all targets.
          // Currently vectorization doesn't handle generic ops with reduction
          // iterators we will disable for now to allow vectorizing producer
          // pointwise ops to avoid performance regressions on CPU.
          if (!clEnableFusionWithReductionOps) {
            if (auto genericOp = dyn_cast<linalg::GenericOp>(consumer)) {
              if (genericOp.getNumReductionLoops()) return false;
            }
          }

          // Limit the number of operands. We have hard limit (32) of bindings
          // passing down to HAL. Set the number to be as same as the limit --
          // IREE_HAL_MODULE_MAX_DESCRIPTOR_BINDING_COUNT.
          constexpr int64_t kIreeMaxOperandCount = 32;
          DenseSet<Value> operands;
          operands.insert(producer->operand_begin(), producer->operand_end());
          operands.insert(consumer->operand_begin(),
                          std::next(consumer->operand_begin(),
                                    consumerOperand.getOperandNumber()));
          operands.insert(std::next(consumer->operand_begin(),
                                    consumerOperand.getOperandNumber() + 1),
                          consumer->operand_end());
          if (operands.size() >= kIreeMaxOperandCount) return false;

          bool isBroadcast = false;
          if (auto genericOp = dyn_cast<linalg::GenericOp>(producer)) {
            // Detect op that only broadcast input as fusing them makes the new
            // op cheaper.
            if (genericOp.getNumParallelLoops() == genericOp.getNumLoops() &&
                isa<linalg::YieldOp>(genericOp.getBody()->front())) {
              for (OpOperand *opOperand : genericOp.getInputOperands()) {
                AffineMap indexingMap = genericOp.getTiedIndexingMap(opOperand);
                if (indexingMap.isProjectedPermutation() &&
                    indexingMap.getNumDims() != indexingMap.getNumResults()) {
                  isBroadcast = true;
                  break;
                }
              }
            }
          }
          // Only fuse if it has a single linalg generic user. It is a
          // simplistic heuristic to avoid duplicating ops that may be
          // expensive.
          // TODO: Add a cost model to allow ops to be duplicated.
          bool hasI1ReturnType =
              llvm::any_of(producer->getResultTypes(), [](Type t) {
                if (t.isInteger(1)) return true;
                if (auto shapedType = t.dyn_cast<ShapedType>()) {
                  if (shapedType.getElementType().isInteger(1)) return true;
                }
                return false;
              });
          if (!isBroadcast && !isa<arith::ConstantOp>(producer) &&
              !hasI1ReturnType &&
              !llvm::hasSingleElement(producerResult.getUsers())) {
            return false;
          }
          return llvm::all_of(producerResult.getUsers(), [](Operation *user) {
            return isa<linalg::GenericOp>(user);
          });
        };
    // Simple heuristic to decide if reshaope should be folded in the linalg.
    // If the source of the reshape is a linalg op fold to potentially allow the
    // two linalg ops to be fused. Otherwise leave it to avoid adding dimensions
    // to the consumer linalg op.
    linalg::ControlElementwiseOpsFusionFn foldReshapeBetweenLinalgFn =
        [](const OpResult &producer, const OpOperand &consumer) {
          auto collapseOp = producer.getDefiningOp<tensor::CollapseShapeOp>();
          if (collapseOp) {
            return collapseOp.src().getDefiningOp<LinalgOp>() != nullptr;
          }
          auto expandOp = producer.getDefiningOp<tensor::ExpandShapeOp>();
          if (expandOp) {
            return expandOp.src().getDefiningOp<LinalgOp>() != nullptr;
          }
          return false;
        };
    linalg::populateElementwiseOpsFusionPatterns(
        fusionPatterns,
        linalg::LinalgElementwiseFusionOptions()
            .setControlFoldingReshapes(foldReshapeBetweenLinalgFn)
            .setControlElementwiseOpsFusionFn(controlFn));

    if (failed(applyPatternsAndFoldGreedily(op->getRegions(),
                                            std::move(fusionPatterns)))) {
      return signalPassFailure();
    }

    OwningRewritePatternList reshapeCanonicalizations(&getContext());
    linalg::populateFoldUnitDimsReshapeOpsByLinearizationPatterns(
        reshapeCanonicalizations);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(
        reshapeCanonicalizations, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(reshapeCanonicalizations,
                                                       context);
    linalg::InitTensorOp::getCanonicalizationPatterns(reshapeCanonicalizations,
                                                      context);
    linalg::FillOp::getCanonicalizationPatterns(reshapeCanonicalizations,
                                                context);
    if (failed(applyPatternsAndFoldGreedily(
            op->getRegions(), std::move(reshapeCanonicalizations)))) {
      return signalPassFailure();
    }

    // Push the remaining reshapes down the graphs.
    OwningRewritePatternList pushReshapePatterns(&getContext());
    linalg::populatePushReshapeOpsPatterns(pushReshapePatterns);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(pushReshapePatterns,
                                                         context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(pushReshapePatterns,
                                                       context);
    linalg::InitTensorOp::getCanonicalizationPatterns(pushReshapePatterns,
                                                      context);
    linalg::FillOp::getCanonicalizationPatterns(pushReshapePatterns, context);
    if (failed(applyPatternsAndFoldGreedily(op->getRegions(),
                                            std::move(pushReshapePatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createFusionOfTensorOpsPass() {
  return std::make_unique<FusionOfTensorOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
