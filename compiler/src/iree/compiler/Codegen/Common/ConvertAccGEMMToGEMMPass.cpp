// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertAccGEMMtoGEMMpass.cpp ----------------------------------===//
//
// Converts Accumulating GEMM to GEMM + elementwise add.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
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

static bool accGemmToGemmPrecondition(Operation *op) {
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
  SmallVector<OpOperand *> outputOperands =
      llvm::to_vector(llvm::make_pointer_range(dpsOp.getDpsInitsMutable()));
  Value outputOperand = outputOperands.front()->get();
  auto outsDefiningOp = outputOperand.getDefiningOp();
  // If not DispatchTensorLoadOp or LoadFromBufferOp then do nothing.
  if (!isa_and_nonnull<IREE::TensorExt::DispatchTensorLoadOp,
                       IREE::Codegen::LoadFromBufferOp>(outsDefiningOp)) {
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

  int64_t outputRank = outputType.getRank();
  SmallVector<utils::IteratorType> iterators(outputRank,
                                             utils::IteratorType::parallel);
  SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(outputRank));

  // Create a zero tensor as the new output tensor operand to the Linalg
  // contraction op.
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, outputOperand);
  Value initOp = rewriter.create<tensor::EmptyOp>(loc, mixedSizes, elementType);
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(elementType));
  Value fill = rewriter.create<linalg::FillOp>(loc, zero, initOp).result();

  // Update the contraction op to use the new zero tensor as output operand.
  rewriter.modifyOpInPlace(dpsOp, [&]() { dpsOp.setDpsInitOperand(0, fill); });

  // Create a generic op to add back the original output tensor operand.
  rewriter.setInsertionPointAfter(dpsOp);
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, outputType, ValueRange{dpsOp->getResult(0), outputOperand},
      ValueRange{initOp}, maps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value result;
        if (llvm::isa<FloatType>(elementType)) {
          result = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
        } else {
          result = b.create<arith::AddIOp>(nestedLoc, args[0], args[1]);
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
        accGemmToGemmPrecondition);
    IRRewriter rewriter(&getContext());
    for (Operation *candidate : candidates) {
      convertAccGemmToGemm(rewriter,
                           cast<DestinationStyleOpInterface>(candidate));
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
