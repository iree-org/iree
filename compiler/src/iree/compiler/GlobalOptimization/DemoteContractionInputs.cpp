// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_DEMOTECONTRACTIONINPUTSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

// Named matmul-like ops whose inputs may be demoted.
static bool isDemotableNamedMatmul(Operation *op) {
  return IREE::LinalgExt::isPureMatmul(op) ||
         IREE::LinalgExt::isPureBatchMatmul(op) ||
         isa<linalg::MatvecOp, linalg::VecmatOp, linalg::BatchMatvecOp,
             linalg::BatchVecmatOp, linalg::MatmulTransposeAOp,
             linalg::MatmulTransposeBOp, linalg::BatchMatmulTransposeAOp,
             linalg::BatchMatmulTransposeBOp>(op);
}

// Named convolution ops whose inputs may be demoted.
static bool isDemotableNamedConv(Operation *op) {
  return isa<linalg::Conv2DOp, linalg::Conv2DNchwFchwOp,
             linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
             linalg::Conv2DNgchwFgchwOp, linalg::Conv2DNgchwGfchwOp>(op);
}

// Template pattern for demoting contraction inputs to a narrower floating-point
// type.
// SrcType: The source floating-point type (e.g., Float32Type)
// DestType: The destination floating-point type (e.g., BFloat16Type,
// Float16Type)
template <typename SrcType, typename DestType>
struct DemoteContractionInputsPattern
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  explicit DemoteContractionInputsPattern(MLIRContext *ctx,
                                          const DemoteOperation &operation)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx),
        demoteOperation(operation) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    bool demoteMatmul = (demoteOperation == DemoteOperation::All) ||
                        (demoteOperation == DemoteOperation::Matmul);
    bool demoteConv = (demoteOperation == DemoteOperation::All) ||
                      (demoteOperation == DemoteOperation::Conv);

    Operation *op = linalgOp.getOperation();
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    bool isMatmul = genericOp ? linalg::isaContractionOpInterface(linalgOp)
                              : isDemotableNamedMatmul(op);
    // The convolution interface checks only indexing maps, which pooling
    // shares, so a generic op also needs a multiply-accumulate body.
    bool isConv = genericOp ? linalg::isaConvolutionOpInterface(linalgOp) &&
                                  linalg::detail::isContractionBody(
                                      *genericOp.getBlock(),
                                      [](Operation *mul, Operation *add) {
                                        return isa<arith::MulFOp>(mul) &&
                                               isa<arith::AddFOp>(add);
                                      })
                            : isDemotableNamedConv(op);
    if (!(demoteMatmul && isMatmul) && !(demoteConv && isConv)) {
      return failure();
    }

    Type srcType = SrcType::get(rewriter.getContext());
    if (!llvm::all_of(linalgOp->getOperands(), [&](auto operand) {
          auto operandType = dyn_cast<RankedTensorType>(operand.getType());
          return operandType && operandType.getElementType() == srcType;
        })) {
      return failure();
    }

    Location loc = linalgOp.getLoc();
    Type destType = DestType::get(rewriter.getContext());
    SmallVector<Value> demotedOperands;
    for (auto inputOperand : linalgOp.getDpsInputOperands()) {
      auto input = inputOperand->get();
      auto inputType = cast<RankedTensorType>(input.getType());
      auto demotedInputType = RankedTensorType::get(
          inputType.getShape(), destType, inputType.getEncoding());
      SmallVector<AffineMap> maps(
          2, rewriter.getMultiDimIdentityMap(inputType.getRank()));
      SmallVector<utils::IteratorType> iteratorTypes(
          inputType.getRank(), utils::IteratorType::parallel);
      SmallVector<OpFoldResult> mixedSizes =
          tensor::getMixedSizes(rewriter, loc, input);
      Value empty =
          tensor::EmptyOp::create(rewriter, loc, mixedSizes, destType);
      demotedOperands.push_back(
          linalg::GenericOp::create(
              rewriter, loc, TypeRange{demotedInputType}, ValueRange{input},
              ValueRange{empty}, maps, iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                Value result =
                    arith::TruncFOp::create(b, loc, destType, args[0]);
                linalg::YieldOp::create(b, loc, result);
              })
              ->getResult(0));
    }
    llvm::append_range(demotedOperands, linalgOp.getDpsInits());

    if (!genericOp) {
      auto newOp = IREE::LinalgExt::cloneNamedLinalgOpWithNewTypes(
          rewriter, linalgOp, linalgOp->getResultTypes(), demotedOperands);
      rewriter.replaceOp(linalgOp, newOp->getResults());
      return success();
    }

    // Preserve the payload (including fast-math attributes) and accumulate in
    // the original type, just as the named mixed-precision ops do.
    auto newOp = cast<linalg::GenericOp>(rewriter.clone(*op));
    newOp->setOperands(demotedOperands);
    OpBuilder::InsertionGuard guard(rewriter);
    Block &body = newOp.getRegion().front();
    rewriter.setInsertionPointToStart(&body);
    for (BlockArgument arg :
         body.getArguments().take_front(newOp.getNumDpsInputs())) {
      arg.setType(destType);
      Value extended = arith::ExtFOp::create(rewriter, loc, srcType, arg,
                                             arith::FastMathFlagsAttr{});
      arg.replaceAllUsesExcept(extended, extended.getDefiningOp());
    }
    rewriter.replaceOp(linalgOp, newOp->getResults());
    return success();
  }

private:
  DemoteOperation demoteOperation;
};

class DemoteContractionInputsPass
    : public impl::DemoteContractionInputsPassBase<
          DemoteContractionInputsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    DemoteOperation ops = demoteOperation.getValue();

    switch (demoteType.getValue()) {
    case DemoteType::F16:
      patterns.insert<DemoteContractionInputsPattern<Float32Type, Float16Type>>(
          context, ops);
      break;
    case DemoteType::BF16:
      patterns
          .insert<DemoteContractionInputsPattern<Float32Type, BFloat16Type>>(
              context, ops);
      break;
    default:
      llvm_unreachable("Unsupported demotion type");
    }

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
createDemoteContractionInputsPass(DemoteType type, DemoteOperation operation) {
  return std::make_unique<DemoteContractionInputsPass>(
      DemoteContractionInputsPassOptions{type, operation});
}

} // namespace mlir::iree_compiler::GlobalOptimization
