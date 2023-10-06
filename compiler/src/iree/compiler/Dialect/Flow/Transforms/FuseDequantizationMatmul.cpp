// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

//----------------------------------------------------------------------------//
//                                Utility
//----------------------------------------------------------------------------//

// Creates a new flow.dipatch.region op and places the
// passed ops inside as long as the dequant op is a
// producer for the matmul op
static LogicalResult fuseDequantAndMatmul(RewriterBase &rewriter,
                                          Operation *dequant, Operation *matmul,
                                          std::optional<Operation *> fill) {

  Flow::DispatchRegionOp regionOp = matmul->getParentOfType<DispatchRegionOp>();
  if (!regionOp) {
    FailureOr<DispatchRegionOp> maybeRegionOp =
        wrapOpInDispatchRegion(rewriter, matmul);
    if (failed(maybeRegionOp))
      return failure();
    regionOp = maybeRegionOp.value();
  }

  FailureOr<DispatchRegionOp> maybeFusedRegionOp =
      clonePrecedingOpIntoDispatchRegion(rewriter, dequant, regionOp);
  if (failed(maybeFusedRegionOp))
    return failure();

  if (fill && *fill) {
    FailureOr<DispatchRegionOp> maybeFusedFillRegionOp =
        clonePrecedingOpIntoDispatchRegion(rewriter, fill.value(), regionOp);
    if (failed(maybeFusedFillRegionOp))
      return failure();
  }

  return success();
}

// Checks if the passed op is a contraction on grouped input
// This function checks that the genericOp:
// 1. isaContractionOpInterface
// 2. Has 2 reduction dimensions (for grouped input)
static LogicalResult isGroupedContractionOp(linalg::GenericOp genericOp) {
  unsigned numLoops = genericOp.getNumLoops();
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(genericOp.getOperation());
  if (numLoops == 0)
    return failure();
  if (!linalg::isaContractionOpInterface(linalgOp))
    return failure();
  if (genericOp.getNumReductionLoops() != 2)
    return failure();
  return success();
}

// Checks if the passed op is a dequantization on grouped input
// This function checks that the genericOp:
// 1. Has a body like:
//      arith.extui
//      arith.uitofp
//      arith.subf
//      arith.mulf
// 2. Increases the bit width of the input
// 3. Has 3 parallel dims
// 4. Has 2 (weights, scales) or 3 (weights, scales, zero points)
//    inputs and 1 output
static LogicalResult isGroupedDequantizationOp(linalg::GenericOp genericOp) {
  // Check for 1 result, and 2 (input, scales) or 3 (input, scales, zero points)
  // inputs
  if (genericOp.getNumDpsInits() != 1)
    return failure();
  if (genericOp.getNumDpsInputs() != 2 && genericOp.getNumDpsInputs() != 3)
    return failure();

  // Check that the rank is at least 3 and all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops < 3)
    return failure();
  if (numLoops != numParallelLoops)
    return failure();

  // Work back from linalg.yield and check body of genericOp.
  // The genericOp should yield the result of an arith.mulf,
  // preceded by an arith.subf, arith.uitofp, and arith.extui
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;

  // Producer of linalg.yield op is arith.mulf
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return failure();
    if (!matchPattern(producer, m_Op<arith::MulFOp>()))
      return failure();
  }

  // Producer of arith.mulf op is arith.subf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return failure();
    if (!matchPattern(producer, m_Op<arith::SubFOp>()))
      return failure();
  }

  // Producer of arith.subf op is arith.uitofp
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return failure();
    if (!matchPattern(producer, m_Op<arith::UIToFPOp>()))
      return failure();
  }

  // Producer of arith.uitofp op is arith.extui
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer)
      return failure();
    if (!matchPattern(producer, m_Op<arith::ExtUIOp>()))
      return failure();
  }

  // Ensure that the dequantization increases the
  // bitwidth from the input to the output
  auto elementTypeOut =
      llvm::cast<ShapedType>(genericOp.getOutputs()[0].getType())
          .getElementType();
  if (!elementTypeOut.isIntOrFloat())
    return failure();
  unsigned bitWidthOut = elementTypeOut.getIntOrFloatBitWidth();
  auto elementTypeIn =
      llvm::cast<ShapedType>(genericOp.getInputs()[0].getType())
          .getElementType();
  if (!elementTypeIn.isIntOrFloat())
    return failure();
  unsigned bitWidthIn = elementTypeIn.getIntOrFloatBitWidth();
  if (bitWidthIn >= bitWidthOut)
    return failure();

  return success();
}

//----------------------------------------------------------------------------//
//                                Patterns
//----------------------------------------------------------------------------//

class FuseDequantizationMatmulPattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Fail if matmul is already in a dispatch
    if (!isNonNullAndOutsideDispatch(genericOp)) {
      return failure();
    }
    // Match first generic op as matmul
    if (failed(isGroupedContractionOp(genericOp)))
      return failure();

    Value genericOpResult = genericOp->getResult(0);
    Operation *matmulOp = genericOpResult.getDefiningOp();

    // Match operands to dequantizations and fuse if matched
    Value lhs = genericOp->getOperand(0);
    Value rhs = genericOp->getOperand(1);
    auto lhsOp = lhs.getDefiningOp<linalg::GenericOp>();
    auto rhsOp = rhs.getDefiningOp<linalg::GenericOp>();

    std::optional<Operation *> maybeFill = std::nullopt;
    if (auto fill = genericOp.getDpsInitOperand(0)
                        ->get()
                        .getDefiningOp<linalg::FillOp>()) {
      maybeFill = fill;
    }

    if (lhsOp)
      if (!failed(isGroupedDequantizationOp(
              llvm::dyn_cast<linalg::GenericOp>(*lhsOp)))) {
        return fuseDequantAndMatmul(rewriter, lhsOp, matmulOp, maybeFill);
      }
    if (rhsOp)
      if (!failed(isGroupedDequantizationOp(
              llvm::dyn_cast<linalg::GenericOp>(*rhsOp)))) {
        return fuseDequantAndMatmul(rewriter, rhsOp, matmulOp, maybeFill);
      }

    return failure();
  }
};

struct FuseDequantizationMatmulPass
    : public FuseDequantizationMatmulBase<FuseDequantizationMatmulPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, Flow::FlowDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Main pattern.
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<FuseDequantizationMatmulPattern>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFuseDequantizationMatmulPass() {
  return std::make_unique<FuseDequantizationMatmulPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
