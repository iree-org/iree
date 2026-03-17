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
    if (demoteOperation == DemoteOperation::None) {
      return failure();
    }
    if (!isa<linalg::ContractionOpInterface, linalg::ConvolutionOpInterface>(
            linalgOp.getOperation())) {
      return failure();
    }

    Type srcType = SrcType::get(rewriter.getContext());
    if (!llvm::all_of(linalgOp->getOperands(), [&](auto operand) {
          auto operandType = dyn_cast<RankedTensorType>(operand.getType());
          return operandType && operandType.getElementType() == srcType;
        })) {
      return failure();
    }

    auto replaceOpInputs = [&](auto *typePtr) {
      Location loc = linalgOp.getLoc();
      Type destType = DestType::get(rewriter.getContext());
      SmallVector<Value> demotedInputs;
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
        demotedInputs.push_back(
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
      auto namedOp = cast<std::remove_pointer_t<decltype(typePtr)>>(
          linalgOp.getOperation());
      rewriter.replaceOpWithNewOp<std::remove_pointer_t<decltype(typePtr)>>(
          linalgOp, demotedInputs, linalgOp.getDpsInits(),
          linalg::getPrunedAttributeList(namedOp));
    };

    bool demoteMatmul = (demoteOperation == DemoteOperation::All) ||
                        (demoteOperation == DemoteOperation::Matmul);

    bool demoteConv = (demoteOperation == DemoteOperation::All) ||
                      (demoteOperation == DemoteOperation::Conv);

    Operation *op = linalgOp.getOperation();
    if (demoteMatmul && IREE::LinalgExt::isPureMatmul(op)) {
      replaceOpInputs(static_cast<linalg::MatmulOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::MatvecOp>(op)) {
      replaceOpInputs(static_cast<linalg::MatvecOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::VecmatOp>(op)) {
      replaceOpInputs(static_cast<linalg::VecmatOp *>(nullptr));
    } else if (demoteMatmul && IREE::LinalgExt::isPureBatchMatmul(op)) {
      replaceOpInputs(static_cast<linalg::BatchMatmulOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::BatchMatvecOp>(op)) {
      replaceOpInputs(static_cast<linalg::BatchMatvecOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::BatchVecmatOp>(op)) {
      replaceOpInputs(static_cast<linalg::BatchVecmatOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::MatmulTransposeAOp>(op)) {
      replaceOpInputs(static_cast<linalg::MatmulTransposeAOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::MatmulTransposeBOp>(op)) {
      replaceOpInputs(static_cast<linalg::MatmulTransposeBOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::BatchMatmulTransposeAOp>(op)) {
      replaceOpInputs(static_cast<linalg::BatchMatmulTransposeAOp *>(nullptr));
    } else if (demoteMatmul && isa<linalg::BatchMatmulTransposeBOp>(op)) {
      replaceOpInputs(static_cast<linalg::BatchMatmulTransposeBOp *>(nullptr));
    } else if (demoteConv && isa<linalg::Conv2DOp>(op)) {
      replaceOpInputs(static_cast<linalg::Conv2DOp *>(nullptr));
    } else if (demoteConv && isa<linalg::Conv2DNchwFchwOp>(op)) {
      replaceOpInputs(static_cast<linalg::Conv2DNchwFchwOp *>(nullptr));
    } else if (demoteConv && isa<linalg::Conv2DNhwcHwcfOp>(op)) {
      replaceOpInputs(static_cast<linalg::Conv2DNhwcHwcfOp *>(nullptr));
    } else if (demoteConv && isa<linalg::Conv2DNhwcFhwcOp>(op)) {
      replaceOpInputs(static_cast<linalg::Conv2DNhwcFhwcOp *>(nullptr));
    } else if (demoteConv && isa<linalg::Conv2DNgchwFgchwOp>(op)) {
      replaceOpInputs(static_cast<linalg::Conv2DNgchwFgchwOp *>(nullptr));
    } else if (demoteConv && isa<linalg::Conv2DNgchwGfchwOp>(op)) {
      replaceOpInputs(static_cast<linalg::Conv2DNgchwGfchwOp *>(nullptr));
    } else {
      return failure();
    }

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
  explicit DemoteContractionInputsPass(
      const DemoteContractionInputsPassOptions &operation) {
    this->demoteType = operation.demoteType;
    this->demoteOperation = operation.demoteOperation;
  }

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
