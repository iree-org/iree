// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_TORCHTOIREEPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

template <typename ViewAsTy>
class BitCastViewAsType : public OpRewritePattern<ViewAsTy> {
public:
  using OpRewritePattern<ViewAsTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(ViewAsTy op,
                                PatternRewriter &rewriter) const override {

    Value in = op.getSelf();
    auto loc = op.getLoc();
    auto inType = cast<torch::Torch::ValueTensorType>(in.getType());
    auto resultType = cast<torch::Torch::ValueTensorType>(op.getType());

    auto bType = inType.toBuiltinTensor();

    if (auto dtype = dyn_cast<IntegerType>(bType.getElementType())) {
      bType = bType.clone(
          rewriter.getType<IntegerType>(dtype.getIntOrFloatBitWidth()));
    }

    // Cast to the builtin tensor type.
    Value builtinCast =
        rewriter.create<torch::TorchConversion::ToBuiltinTensorOp>(loc, bType,
                                                                   in);

    auto rType = resultType.toBuiltinTensor();
    if (auto dtype = dyn_cast<IntegerType>(rType.getElementType())) {
      rType = rType.clone(
          rewriter.getType<IntegerType>(dtype.getIntOrFloatBitWidth()));
    }

    // Both `view_as_real` and `view_as_complex` only affect the element type
    // and insert/remove the inner most static dimension (size 2), thus the
    // source and result share the same dynamic dims.
    ValueRange dynamicDims =
        IREE::Util::buildDynamicDimsForValue(loc, builtinCast, rewriter);
    // Both view like ops are just bitcasts. Upstream bitcasting operations only
    // allow changing element types without changing bitwidth or any tensor
    // dimensions, so we use our own bitcasting operation here.
    Value flowBitcast = rewriter.create<IREE::Flow::TensorBitCastOp>(
        loc, rType, builtinCast, dynamicDims, dynamicDims);

    auto torchCast =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, resultType, flowBitcast);
    rewriter.replaceOp(op, torchCast);
    return success();
  }
};
} // namespace

namespace {
class TorchToIREEPass final
    : public impl::TorchToIREEPassBase<TorchToIREEPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<torch::TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BitCastViewAsType<torch::Torch::AtenViewAsRealOp>,
                 BitCastViewAsType<torch::Torch::AtenViewAsComplexOp>>(context);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
