// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- PromoteMatmulForUKernel ------------------------------===//
// Promote matmul input types to match available ukernel
//===----------------------------------------------------------------------===//

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-widen-matmul-for-ukernel"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

using MatmulTypeTuple = std::tuple<Type, Type, Type>;

static FailureOr<MatmulTypeTuple> findPromotionType(MLIRContext *ctx,
                                                    Type lhsElemType,
                                                    Type rhsElemType,
                                                    Type outElemType) {
  // Considered dst promotion types to match UKernels. See
  // iree/builtins/ukernel/exported_bits.h for all UKernel supported types.
  SmallVector<MatmulTypeTuple> supportedElemTypes = {
      {FloatType::getF32(ctx), FloatType::getF32(ctx), FloatType::getF32(ctx)},
  };
  for (auto [dstLhsElemType, dstRhsElemType, dstOutElemType] :
       supportedElemTypes) {
    // Result promotion is unsupported.
    if (outElemType != dstOutElemType) {
      continue;
    }
    // No need to promote.
    if (lhsElemType == dstLhsElemType && rhsElemType == dstRhsElemType) {
      return failure();
    }
    // Promote when only one side is mismatched.
    if (lhsElemType != dstLhsElemType && rhsElemType != dstRhsElemType) {
      continue;
    }
    auto canPromote = [](Type srcType, Type dstType) {
      // Only promote when the original type is narrower and int->float (can be
      // relaxed if needed).
      return (srcType.getIntOrFloatBitWidth() <
              dstType.getIntOrFloatBitWidth()) &&
             (isa<IntegerType>(srcType) && isa<FloatType>(dstType));
    };
    if (lhsElemType != dstLhsElemType &&
        !canPromote(lhsElemType, dstLhsElemType)) {
      continue;
    }
    if (rhsElemType != dstRhsElemType &&
        !canPromote(rhsElemType, dstRhsElemType)) {
      continue;
    }
    return MatmulTypeTuple({dstLhsElemType, dstRhsElemType, dstOutElemType});
  }
  return failure();
}

struct PromoteMatmulForUKernel
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {

  using OpInterfaceRewritePattern<
      linalg::ContractionOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::ContractionOpInterface op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp || !linalgOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(op, "unsupported contraction op");
    }
    if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
      return rewriter.notifyMatchFailure(op, "unsupported contraction op");
    }

    Value lhs = linalgOp.getDpsInputs()[0];
    Value rhs = linalgOp.getDpsInputs()[1];
    Value out = linalgOp.getDpsInits()[0];

    TensorType lhsType = lhs.getType().cast<TensorType>();
    TensorType rhsType = rhs.getType().cast<TensorType>();
    TensorType outType = out.getType().cast<TensorType>();
    Type lhsElemType = lhsType.getElementType();
    Type rhsElemType = rhsType.getElementType();
    Type outElemType = outType.getElementType();

    MLIRContext *ctx = rewriter.getContext();

    FailureOr<MatmulTypeTuple> promoteTypes =
        findPromotionType(ctx, lhsElemType, rhsElemType, outElemType);
    if (failed(promoteTypes)) {
      return rewriter.notifyMatchFailure(op, "promotion not found");
    }
    Type dstLhsElemType;
    Type dstRhsElemType;
    std::tie(dstLhsElemType, dstRhsElemType, std::ignore) = *promoteTypes;

    auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);

    auto promoteInput = [&](Value input, Type dstElemType) {
      auto inputType = input.getType().cast<TensorType>();
      return rewriter.create<arith::SIToFPOp>(
          loc, inputType.cloneWith(/*shape=*/std::nullopt, dstElemType), input);
    };
    if (lhsElemType != dstLhsElemType) {
      lhs = promoteInput(lhs, dstLhsElemType);
    }
    if (rhsElemType != dstRhsElemType) {
      rhs = promoteInput(rhs, dstRhsElemType);
    }

    auto newOp = TypeSwitch<Operation *, Operation *>(op)
                     .Case<linalg::MatmulOp>([&](auto _) {
                       return rewriter.create<linalg::MatmulOp>(
                           loc, out.getType(), ValueRange{lhs, rhs}, out);
                     })
                     .Case<linalg::BatchMatmulOp>([&](auto _) {
                       return rewriter.create<linalg::BatchMatmulOp>(
                           loc, out.getType(), ValueRange{lhs, rhs}, out);
                     })
                     .Default([&](auto _) { return nullptr; });
    assert(newOp && "unexpected op type");
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct PromoteMatmulForUKernelPass
    : public PromoteMatmulForUKernelPassBase<PromoteMatmulForUKernelPass> {
  void runOnOperation() override;
};

} // namespace

void PromoteMatmulForUKernelPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<PromoteMatmulForUKernel>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createPromoteMatmulForUKernelPass() {
  return std::make_unique<PromoteMatmulForUKernelPass>();
}
} // namespace mlir::iree_compiler::GlobalOptimization
