// Copyright 2026 The IREE Authors
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

#define GEN_PASS_DEF_PROMOTECONTRACTIONOUTPUTSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

template <typename T>
void replaceOpOutputs(linalg::LinalgOp linalgOp, PatternRewriter &rewriter,
                      Type srcType, Type destType) {
  Location loc = linalgOp.getLoc();
  assert(linalgOp.getNumDpsInits() == 1);
  auto output = linalgOp.getDpsInits()[0];
  auto outputType = cast<RankedTensorType>(output.getType());
  auto promoteOutputType = RankedTensorType::get(
      outputType.getShape(), destType, outputType.getEncoding());
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(outputType.getRank()));
  SmallVector<utils::IteratorType> iteratorTypes(outputType.getRank(),
                                                 utils::IteratorType::parallel);
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, output);
  Value promoteEmpty =
      tensor::EmptyOp::create(rewriter, loc, mixedSizes, destType);
  Value promoteOutput =
      linalg::GenericOp::create(
          rewriter, loc, TypeRange{promoteOutputType}, ValueRange{output},
          ValueRange{promoteEmpty}, maps, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value result = arith::ExtFOp::create(b, loc, destType, args[0]);
            linalg::YieldOp::create(b, loc, result);
          })
          ->getResult(0);
  auto namedOp = cast<T>(linalgOp.getOperation());
  auto newLinalgOp = T::create(rewriter, loc, linalgOp.getDpsInputs(),
                               ValueRange{promoteOutput},
                               linalg::getPrunedAttributeList(namedOp));
  Value truncEmpty =
      tensor::EmptyOp::create(rewriter, loc, mixedSizes, srcType);
  rewriter.replaceOpWithNewOp<linalg::GenericOp>(
      linalgOp, TypeRange{outputType}, ValueRange{newLinalgOp->getResult(0)},
      ValueRange{truncEmpty}, maps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value trunc = arith::TruncFOp::create(b, loc, srcType, args[0]);
        linalg::YieldOp::create(b, loc, trunc);
      });
}

// Template pattern for promoting contraction outputs from a narrower
// floating-point type.
// SrcType: The source floating-point type (e.g., BFloat16Type, Float16Type)
// DestType: The destination floating-point type (e.g., Float32Type)
template <typename SrcType, typename DestType>
struct PromoteContractionOutputsPattern
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  explicit PromoteContractionOutputsPattern(MLIRContext *ctx,
                                            const PromoteOperation &operation)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx),
        promoteOperation(operation) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (promoteOperation == PromoteOperation::None) {
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

    bool promoteMatmul = (promoteOperation == PromoteOperation::All) ||
                         (promoteOperation == PromoteOperation::Matmul);
    bool promoteConv = (promoteOperation == PromoteOperation::All) ||
                       (promoteOperation == PromoteOperation::Conv);
    Operation *op = linalgOp.getOperation();
    Type destType = DestType::get(rewriter.getContext());
    if (promoteMatmul) {
      if (IREE::LinalgExt::isPureMatmul(op)) {
        replaceOpOutputs<linalg::MatmulOp>(linalgOp, rewriter, srcType,
                                           destType);
        return success();
      }
      if (IREE::LinalgExt::isPureBatchMatmul(op)) {
        replaceOpOutputs<linalg::BatchMatmulOp>(linalgOp, rewriter, srcType,
                                                destType);
        return success();
      }
    }
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .template Case<linalg::MatvecOp, linalg::VecmatOp,
                       linalg::BatchMatvecOp, linalg::BatchVecmatOp,
                       linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp,
                       linalg::BatchMatmulTransposeAOp,
                       linalg::BatchMatmulTransposeBOp>([&](auto T) {
          if (!promoteMatmul) {
            return failure();
          }
          replaceOpOutputs<decltype(T)>(linalgOp, rewriter, srcType, destType);
          return success();
        })
        .template Case<linalg::Conv2DOp, linalg::Conv2DNchwFchwOp,
                       linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNhwcFhwcOp,
                       linalg::Conv2DNgchwFgchwOp, linalg::Conv2DNgchwGfchwOp>(
            [&](auto T) {
              if (!promoteConv) {
                return failure();
              }
              replaceOpOutputs<decltype(T)>(linalgOp, rewriter, srcType,
                                            destType);
              return success();
            })
        .Default(failure);
  }

private:
  PromoteOperation promoteOperation;
};

class PromoteContractionOutputsPass
    : public impl::PromoteContractionOutputsPassBase<
          PromoteContractionOutputsPass> {
public:
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    PromoteOperation ops = promoteOperation.getValue();

    switch (promoteType.getValue()) {
    case PromoteType::F16:
      patterns
          .insert<PromoteContractionOutputsPattern<Float16Type, Float32Type>>(
              context, ops);
      break;
    case PromoteType::BF16:
      patterns
          .insert<PromoteContractionOutputsPattern<BFloat16Type, Float32Type>>(
              context, ops);
      break;
    default:
      llvm_unreachable("Unsupported promotion type");
    }

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
createPromoteContractionOutputsPass(PromoteType type,
                                    PromoteOperation operation) {
  return std::make_unique<PromoteContractionOutputsPass>(
      PromoteContractionOutputsPassOptions{type, operation});
}

} // namespace mlir::iree_compiler::GlobalOptimization
