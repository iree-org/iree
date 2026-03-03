// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Converts torch.aten.mm → mips.matmul.
//
// The pattern runs inside the Torch input-conversion pipeline, BEFORE
// createConvertTorchToLinalgPass(), so it intercepts aten.mm first.
//
// Since torch ops carry ValueTensorType (torch's tensor type), the pattern:
//   1. Casts operands to builtin RankedTensorType via ToBuiltinTensorOp.
//   2. Creates a zero-initialised init tensor (Destination Passing Style).
//   3. Emits mips.matmul on builtin tensors.
//   4. Casts the result back to ValueTensorType via FromBuiltinTensorOp.
//
// This mirrors the approach in ConvertTorchUnstructuredToLinalgExt.cpp.

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTORCHTOMIPSPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper: create a zero-filled tensor of a given shape and element type.
// Accepts (M, N) as dynamic Value dimensions.
//===----------------------------------------------------------------------===//

static Value createZeroTensor(PatternRewriter &rewriter, Location loc,
                               RankedTensorType ty, ValueRange dynSizes) {
  Value empty = tensor::EmptyOp::create(rewriter, loc, ty, dynSizes);
  Attribute zeroAttr = rewriter.getZeroAttr(ty.getElementType());
  Value zero = arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(zeroAttr));
  return linalg::FillOp::create(rewriter, loc, zero, empty).result();
}

//===----------------------------------------------------------------------===//
// Pattern: torch.aten.mm → mips.matmul
//===----------------------------------------------------------------------===//

struct ConvertAtenMmToMIPSMatmul
    : public OpRewritePattern<torch::Torch::AtenMmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(torch::Torch::AtenMmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // ----------------------------------------------------------------
    // 1. Verify that we have supported tensor types.
    // ----------------------------------------------------------------
    auto lhsTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getSelf().getType());
    auto rhsTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getMat2().getType());
    auto resultTorchTy =
        dyn_cast<torch::Torch::ValueTensorType>(op.getType());

    if (!lhsTorchTy || !rhsTorchTy || !resultTorchTy)
      return rewriter.notifyMatchFailure(op, "expected ValueTensorType");

    // Only handle f32 for now (extensible).
    if (!lhsTorchTy.getDtype().isF32())
      return rewriter.notifyMatchFailure(op, "only f32 supported");

    // ----------------------------------------------------------------
    // 2. Cast operands from torch ValueTensorType → builtin RankedTensorType.
    // ----------------------------------------------------------------
    auto lhsBuiltinTy =
        dyn_cast_or_null<RankedTensorType>(lhsTorchTy.toBuiltinTensor());
    auto rhsBuiltinTy =
        dyn_cast_or_null<RankedTensorType>(rhsTorchTy.toBuiltinTensor());
    auto resultBuiltinTy =
        dyn_cast_or_null<RankedTensorType>(resultTorchTy.toBuiltinTensor());

    if (!lhsBuiltinTy || !rhsBuiltinTy || !resultBuiltinTy ||
        lhsBuiltinTy.getRank() != 2 || rhsBuiltinTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2-D ranked tensors");

    Value lhs = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, lhsBuiltinTy, op.getSelf());
    Value rhs = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, rhsBuiltinTy, op.getMat2());

    // ----------------------------------------------------------------
    // 3. Collect dynamic dimension values for the result tensor (M, N).
    // ----------------------------------------------------------------
    SmallVector<Value> dynSizes;
    if (resultBuiltinTy.isDynamicDim(0))
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, lhs, 0));
    if (resultBuiltinTy.isDynamicDim(1))
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, rhs, 1));

    // ----------------------------------------------------------------
    // 4. Create a zero-initialised init tensor for DPS output.
    // ----------------------------------------------------------------
    Value init = createZeroTensor(rewriter, loc, resultBuiltinTy, dynSizes);

    // ----------------------------------------------------------------
    // 5. Emit mips.matmul on builtin tensors.
    // ----------------------------------------------------------------
    Value result =
        IREE::MIPS::MatmulOp::create(rewriter, loc, TypeRange{resultBuiltinTy},
                                     lhs, rhs, init)
            .getResult();

    // ----------------------------------------------------------------
    // 6. Cast result back to ValueTensorType so downstream torch passes can
    //    still operate on it until the type finalisation pass runs.
    // ----------------------------------------------------------------
    Value torchResult = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, resultTorchTy, result);

    rewriter.replaceOp(op, torchResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTorchToMIPSPass
    : impl::ConvertTorchToMIPSPassBase<ConvertTorchToMIPSPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::MIPS::MIPSDialect,
                    torch::TorchConversion::TorchConversionDialect,
                    arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertAtenMmToMIPSMatmul>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::iree_compiler::TorchInput
