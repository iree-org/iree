// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTACCGEMMTOGEMMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

/// Check whether `outputOperand` is a reduction with a single combiner
/// operation. Return the combiner operation of the reduction. Return
/// nullptr otherwise. Multiple reduction operations would impose an
/// ordering between reduction dimensions and is currently unsupported in
/// Linalg. This limitation is motivated by the fact that e.g. min(max(X)) !=
/// max(min(X))
/// TODO(hanchung): This is a method copied from upstream `Vectorization.cpp`.
/// Expose the upstream method to public and delete the method.
static Operation *matchLinalgReduction(OpOperand *outputOperand) {
  auto linalgOp = cast<linalg::LinalgOp>(outputOperand->getOwner());
  unsigned outputPos =
      outputOperand->getOperandNumber() - linalgOp.getNumDpsInputs();
  // Only single combiner operations are supported for now.
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(linalgOp.getRegionOutputArgs(), outputPos, combinerOps) ||
      combinerOps.size() != 1)
    return nullptr;

  // Return the combiner operation.
  return combinerOps[0];
}

static Value getIdentityValue(OpBuilder &b, DestinationStyleOpInterface op,
                              Type elementType) {
  Location loc = op->getLoc();
  if (isa<IREE::Codegen::InnerTiledOp>(op)) {
    return b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
  }
  Operation *reduceOp = matchLinalgReduction(op.getDpsInitOperand(0));
  if (!reduceOp) {
    return nullptr;
  }
  std::optional<TypedAttr> identityAttr = arith::getNeutralElement(reduceOp);
  if (!identityAttr) {
    return nullptr;
  }
  std::optional<vector::CombiningKind> combinerOp =
      linalg::getCombinerOpKind(reduceOp);
  if (!combinerOp) {
    return nullptr;
  }
  return b.create<arith::ConstantOp>(loc, elementType, identityAttr.value());
}

static bool accReductionToReductionPrecondition(Operation *op) {
  if (auto innerTiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op)) {
    return isa<IREE::GPU::MmaInterfaceAttr, IREE::GPU::ScaledMMAAttr>(
        innerTiledOp.getKind());
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return false;
  }
  if (!linalg::isaContractionOpInterface(linalgOp) &&
      !isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
    return false;
  }
  if (!linalgOp.hasPureTensorSemantics()) {
    return false;
  }
  return linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0))
      .isProjectedPermutation();
}

static void convertAccGemmToGemm(RewriterBase &rewriter,
                                 DestinationStyleOpInterface dpsOp) {
  // Fill to ReadOnly tensor can be handled as broadcasting in the new
  // elementwise op.
  SmallVector<OpOperand *> outputOperands =
      llvm::to_vector(llvm::make_pointer_range(dpsOp.getDpsInitsMutable()));
  Value outputOperand = outputOperands.front()->get();
  auto origFillOp = outputOperand.getDefiningOp<linalg::FillOp>();
  if (origFillOp && !isReadOnly(origFillOp.getDpsInitOperand(0)->get())) {
    return;
  }
  if (!origFillOp && !isReadOnly(outputOperand)) {
    return;
  }

  auto outputType = cast<RankedTensorType>(outputOperand.getType());
  if (!outputType.getElementType().isIntOrFloat()) {
    return;
  }
  auto elementType = outputType.getElementType();

  Location loc = dpsOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(dpsOp);

  // Create a init tensor as the new output tensor operand to the reduction op.
  Value identityVal = getIdentityValue(rewriter, dpsOp, elementType);
  if (!identityVal) {
    return;
  }
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, outputOperand);
  Value initOp = rewriter.create<tensor::EmptyOp>(loc, mixedSizes, elementType);
  Value fill =
      rewriter.create<linalg::FillOp>(loc, identityVal, initOp).result();

  // Update the reduction op to use the new `identityVal` tensor as output
  // operand.
  rewriter.modifyOpInPlace(dpsOp, [&]() { dpsOp.setDpsInitOperand(0, fill); });

  // Create a generic op to accumulate back the original output tensor operand.
  rewriter.setInsertionPointAfter(dpsOp);
  int64_t outputRank = outputType.getRank();
  SmallVector<Value> inputs = {dpsOp->getResult(0), outputOperand};
  SmallVector<utils::IteratorType> iterators(outputRank,
                                             utils::IteratorType::parallel);
  SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(outputRank));
  if (origFillOp) {
    inputs[1] = origFillOp.getInputs()[0];
    maps[1] = AffineMap::get(maps[0].getNumDims(), 0, dpsOp.getContext());
  }
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, outputType, inputs, ValueRange{initOp}, maps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value result;
        // Only InnerTiledOp ops do not have compute body.
        if (isa<IREE::Codegen::InnerTiledOp>(dpsOp)) {
          if (llvm::isa<FloatType>(elementType)) {
            result = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          } else {
            result = b.create<arith::AddIOp>(nestedLoc, args[0], args[1]);
          }
        } else {
          Operation *combinerOp =
              matchLinalgReduction(dpsOp.getDpsInitOperand(0));
          result = b.create(nestedLoc, combinerOp->getName().getIdentifier(),
                            {args[0], args[1]}, args[1].getType())
                       ->getResult(0);
        }
        b.create<linalg::YieldOp>(nestedLoc, result);
      });
  dpsOp->getResult(0).replaceAllUsesExcept(genericOp->getResult(0), genericOp);
}

namespace {

struct ConvertAccGEMMToGEMMPass final
    : impl::ConvertAccGEMMToGEMMPassBase<ConvertAccGEMMToGEMMPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<Operation *> candidates = llvm::filter_to_vector(
        llvm::make_pointer_range(funcOp.getFunctionBody().getOps()),
        accReductionToReductionPrecondition);
    IRRewriter rewriter(&getContext());
    for (Operation *candidate : candidates) {
      convertAccGemmToGemm(rewriter,
                           cast<DestinationStyleOpInterface>(candidate));
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
