// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/VectorLayoutProvider.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Verifier.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                      TypedValue<VectorType> value,
                                      LayoutProvider *provider) {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    value = cast<TypedValue<VectorType>>(toSIMD.getInput());
    return value;
  }
  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape = provider->getDistributedShape(value);
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = rewriter.create<IREE::VectorExt::ToSIMTOp>(
      value.getLoc(), distributedType, value);
  return toSIMT.getResult();
}

void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    LayoutProvider *provider,
                                    ValueRange values) {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (OpResult opResult : op->getOpResults()) {
    Value replacement = values[opResult.getResultNumber()];
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<TypedValue<VectorType>>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      auto toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldResult.getLoc(), oldResult.getType(), replacement);
      // Clone the layout to the new value.
      provider->getAnalysis().cloneLayoutInformationToNewValue(
          oldResult, toSIMD.getResult());
      // Add to replacements.
      replacement = toSIMD.getResult();
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

class DistributionRewriter : public IRRewriter, public RewriterBase::Listener {
public:
  DistributionRewriter(MLIRContext *ctx, DenseSet<Operation *> &erasedOps,
                       SmallVector<Operation *> &worklist,
                       VectorLayoutAnalysis &analysis)
      : IRRewriter(ctx), erasedOps(erasedOps), worklist(worklist),
        analysis(analysis) {
    setListener(this);
  }

protected:
  void notifyOperationRemoved(Operation *op) override { erasedOps.insert(op); }

  void notifyOperationInserted(Operation *op) override {
    // For now, do nothing.
    // TODO: Check if the operation can still be distributed and try to
    // distribute it. Not sure how to check this? Maybe we can check if the
    // analysis knows about this value and assume that this value needs to be
    // distributed. This is needed for SIMD -> SIMD rewrites.
  }

private:
  // A reference to the set of operations that have been erased.
  DenseSet<Operation *> &erasedOps;
  // A reference to the worklist of operations that need to be distributed.
  SmallVector<Operation *> &worklist;
  // A reference to the analysis that provides the layout information.
  VectorLayoutAnalysis &analysis;
};

VectorDistribution::VectorDistribution(func::FuncOp root,
                                       VectorLayoutAnalysis &analysis,
                                       LayoutProvider *provider)
    : root(root), analysis(analysis), provider(provider) {
  provider->setAnchorOps();
  if (failed(analysis.run()))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Completed Successfully :\n");
  LLVM_DEBUG(analysis.print(llvm::dbgs()));
}

static bool canDistribute(Operation *op, VectorLayoutAnalysis &analysis) {
  bool needsDistribution = false;
  // Check if this operation has any operands with a vector type. If so,
  // then they need to have a layout.
  for (Value operand : op->getOperands()) {
    if (isa<VectorType>(operand.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(operand)) {
        return false;
      }
    }
  }

  // Check if this operation has any results with a vector type. If so,
  // then they need to have a layout.
  for (OpResult result : op->getResults()) {
    if (isa<VectorType>(result.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(result)) {
        return false;
      }
    }
  }

  return needsDistribution;
}

void VectorDistribution::distribute() {
  SmallVector<Operation *> worklist;
  DenseSet<Operation *> erasedOps;
  DistributionRewriter rewriter(root.getContext(), erasedOps, worklist,
                                analysis);

  // 1: Collect all operations that need to be distributed.
  LLVM_DEBUG(llvm::dbgs() << "Step 1: Collecting operations to distribute\n");
  root->walk([&](Operation *op) {
    if (canDistribute(op, analysis)) {
      worklist.push_back(op);
    }
  });

  // 2. Set the insertion point to the beginning of the root.
  LLVM_DEBUG(
      llvm::dbgs() << "Step 2: Set insertion point to beginning of root\n");
  rewriter.setInsertionPointToStart(&root.getBody().getBlocks().front());

  // 3. Distribute all operations in the worklist until we reach a fixed
  // point.
  LLVM_DEBUG(
      llvm::dbgs() << "Step 3: Starting distributing collected operations\n");
  bool changed = true;
  while (changed) {
    changed = false;
    for (unsigned i = 0; i < worklist.size(); ++i) {
      Operation *op = worklist[i];
      if (erasedOps.count(op))
        continue;

      rewriter.setInsertionPoint(op);

      LLVM_DEBUG(llvm::dbgs() << "Trying to distribute: ");
      LLVM_DEBUG(op->print(llvm::dbgs(), OpPrintingFlags().skipRegions()));
      LLVM_DEBUG(llvm::dbgs() << "\n");

      if (provider->specializedDistribution(rewriter, op).succeeded()) {
        LLVM_DEBUG(llvm::dbgs() << "Specialized Distribution Successful\n");
        changed = true;
        continue;
      }

      LogicalResult distributed =
          TypeSwitch<Operation *, LogicalResult>(op)
              .Case([&](arith::ConstantOp constantOp) {
                return distributeConstants(rewriter, constantOp);
              })
              .Default([&](Operation *op) { return failure(); });

      // If the operation was distributed, continue with the next one.
      if (distributed.succeeded()) {
        LLVM_DEBUG(llvm::dbgs() << "Distribution Successful\n");
        changed = true;
        continue;
      }

      // Try distributing as an elementwise operation.
      if (OpTrait::hasElementwiseMappableTraits(op)) {
        if (distributeElementwise(rewriter, op).succeeded()) {
          LLVM_DEBUG(llvm::dbgs() << "Distribution Successful\n");
          changed = true;
        }
      }

      LLVM_DEBUG(llvm::dbgs() << "Distribution Failed\n");
    }
  }

  // 4. Ideally, we should error out here if everything was not distributed.
  // Currently, I'm not adding it for debugging purposes.
  // TODO: Add a check here if something was not distributed.
  LLVM_DEBUG(llvm::dbgs() << "Distribution Finished\n");
}

LogicalResult
VectorDistribution::distributeConstants(RewriterBase &rewriter,
                                        arith::ConstantOp constantOp) {
  Value constantResult = constantOp.getResult();
  if (!isa<VectorType>(constantResult.getType()))
    return failure();
  auto constant = cast<TypedValue<VectorType>>(constantResult);
  auto attr = llvm::cast<DenseElementsAttr>(constantOp.getValue());

  // Only handle splat values for now.
  if (!attr.isSplat())
    return failure();

  // Replace the original op with the distributed op.
  Type elementType = constant.getType().getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(constant), elementType);
  replaceOpWithNewDistributedOp<arith::ConstantOp>(
      provider, rewriter, constantOp, vectorType,
      DenseElementsAttr::get(vectorType, attr.getSplatValue<APFloat>()));
  return success();
}

LogicalResult VectorDistribution::distributeElementwise(RewriterBase &rewriter,
                                                        Operation *op) {
  assert(OpTrait::hasElementwiseMappableTraits(op) &&
         "expected elementwise op");

  // Get the distributed operands.
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    if (auto vectorOperand = dyn_cast<TypedValue<VectorType>>(operand)) {
      operand = getDistributed(rewriter, vectorOperand, provider);
    }
    operands.push_back(operand);
  }

  // Get the new distributed vector types for the operation.
  SmallVector<Type> resultTypes;
  for (Value result : op->getResults()) {
    if (auto vectorResult = dyn_cast<TypedValue<VectorType>>(result)) {
      // Distribute vector result types.
      auto newType =
          VectorType::get(provider->getDistributedShape(vectorResult),
                          vectorResult.getType().getElementType());
      resultTypes.push_back(newType);
    } else {
      resultTypes.push_back(result.getType());
    }
  }

  // Replace the original op with the distributed op.
  Operation *distributedOp =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                      resultTypes, op->getAttrs());
  replaceOpWithDistributedValues(rewriter, op, provider,
                                 distributedOp->getResults());
  return success();
}

} // namespace mlir::iree_compiler
