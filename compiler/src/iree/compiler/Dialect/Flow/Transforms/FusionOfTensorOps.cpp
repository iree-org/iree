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

#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-fusion-of-tensor-ops"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Check if the producer generic op is fusable with the consumer generic op.
static bool areFusableOps(MLIRContext *context, Operation *producerOp,
                          Operation *consumerOp) {
  // Check for i1 return types, if so aggressively fuse to avoid `i1` buffers.
  if (llvm::all_of(producerOp->getResultTypes(), [](Type t) {
        if (t.isInteger(1)) return true;
        if (auto shapedType = t.dyn_cast<ShapedType>()) {
          if (shapedType.getElementType().isInteger(1)) return true;
        }
        return false;
      })) {
    return true;
  }

  // If producer has a single user, always fuse
  if (producerOp->hasOneUse()) return true;

  // If the generic op is "just" copy, then fuse always.
  Block &body = producerOp->getRegion(0).front();
  if (std::begin(body)->hasTrait<OpTrait::IsTerminator>()) return true;

  // All other cases dont fuse.
  return false;
}

namespace {

/// Pass to fuse linalg on tensor operations as well as fusion of hal.interface*
/// operations with linalg.tensor_reshape operation.
struct FusionOfTensorOpsPass
    : public FusionOfTensorOpsBase<FusionOfTensorOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet fusionPatterns(&getContext());
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    // Only fuse operations where all uses of the producer are generic
    // operations. If an operation is used in a named op, it will be computed
    // anyway, so the consumers can just use that value.
    linalg::ControlFusionFn fuseElementwiseOpsControlFn =
        [&](const OpResult &producerResult, OpOperand &consumerOperand) {
          Operation *producer = producerResult.getOwner();
          Operation *consumer = consumerOperand.getOwner();

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

          return areFusableOps(context, producer, consumer);
        };
    linalg::populateElementwiseOpsFusionPatterns(fusionPatterns,
                                                 fuseElementwiseOpsControlFn);

    // Always fold reshape by expansion.
    linalg::ControlFusionFn fuseByExpansionControlFn =
        [](const OpResult &producer, const OpOperand &consumer) {
          // Do not fuse producer generic op if it has more than one user.
          if (auto producerGenericOp =
                  dyn_cast<linalg::GenericOp>(producer.getOwner())) {
            return producerGenericOp->hasOneUse();
          }
          // Fuse in all other cases.
          return true;
        };
    linalg::populateFoldReshapeOpsByExpansionPatterns(fusionPatterns,
                                                      fuseByExpansionControlFn);

    // Constant fold Linalg operations.
    auto constantFoldControlFn = [](const OpResult &producer,
                                    OpOperand &consumer) {
      return producer.getOwner()->hasOneUse();
    };
    linalg::populateConstantFoldLinalgOperations(fusionPatterns,
                                                 constantFoldControlFn);

    AffineApplyOp::getCanonicalizationPatterns(fusionPatterns, context);
    linalg::GenericOp::getCanonicalizationPatterns(fusionPatterns, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(fusionPatterns, context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(fusionPatterns,
                                                         context);
    context->getLoadedDialect<linalg::LinalgDialect>()
        ->getCanonicalizationPatterns(fusionPatterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(fusionPatterns);

    if (failed(applyPatternsAndFoldGreedily(op->getRegions(),
                                            std::move(fusionPatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\n--- After first fixed point ---\n";
      op->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // For fusion by collapsing, do so if the reshape is blocking tile and fuse.
    linalg::ControlFusionFn fuseByCollapsingControlFn =
        [](const OpResult &producer, OpOperand &consumer) {
          auto reshapeOp = dyn_cast<tensor::ExpandShapeOp>(producer.getOwner());
          if (!reshapeOp) return true;

          return reshapeOp.getSrc().getDefiningOp<linalg::LinalgOp>() !=
                 nullptr;
        };

    RewritePatternSet collapsingReshapePatterns(&getContext());
    linalg::populateFoldReshapeOpsByCollapsingPatterns(
        collapsingReshapePatterns, fuseByCollapsingControlFn);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(
        collapsingReshapePatterns, context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(
        collapsingReshapePatterns, context);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(
        collapsingReshapePatterns);
    if (failed(applyPatternsAndFoldGreedily(
            op->getRegions(), std::move(collapsingReshapePatterns)))) {
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
