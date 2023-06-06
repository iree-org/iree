// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

#include <numeric>
#include <random>

#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Rewriters.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_STATEFULRNG
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h.inc"

namespace {

using GlobalFn = std::function<ml_program::GlobalOp()>;

class ExpandRngUniform : public OpRewritePattern<::mlir::stablehlo::RngOp> {
public:
  ExpandRngUniform(MLIRContext *context, GlobalFn &getGlobal)
      : OpRewritePattern<::mlir::stablehlo::RngOp>::OpRewritePattern(context),
        getGlobal(getGlobal){};

  LogicalResult matchAndRewrite(::mlir::stablehlo::RngOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRngDistribution() != ::mlir::stablehlo::RngDistribution::UNIFORM)
      return failure();

    auto elementType = getElementTypeOrSelf(op.getType());
    int width = 0;
    if (auto t = dyn_cast<IntegerType>(elementType))
      width = t.getWidth();
    else if (auto t = dyn_cast<FloatType>(elementType);
             t && APFloat(t.getFloatSemantics()).isIEEE())
      width = t.getWidth();
    else
      return rewriter.notifyMatchFailure(op, "unsupported type");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ml_program::GlobalOp global = getGlobal();
    auto symbol = SymbolRefAttr::get(global.getSymNameAttr());
    auto val = b.create<ml_program::GlobalLoadOp>(global.getType(), symbol);
    auto algo = ::mlir::stablehlo::RngAlgorithm::THREE_FRY;
    // Generate integral rng output/bits.
    auto intType = IntegerType::get(getContext(), width);
    auto rngOp = b.create<::mlir::stablehlo::RngBitGeneratorOp>(
        TypeRange{val.getType(), cast<TensorType>(op.getType()).clone(intType)},
        algo, val);
    auto bits = rngOp.getOutput();

    Value out;
    if (isa<IntegerType>(elementType)) {
      // Following XLA client's lowering. Has some gaps but consistent.
      auto range =
          b.create<::mlir::stablehlo::SubtractOp>(op.getB(), op.getA());
      auto dist = b.create<::mlir::chlo::BroadcastRemOp>(bits, range, nullptr);
      auto div_2 = b.create<::mlir::stablehlo::ShiftRightLogicalOp>(
          dist,
          b.create<::mlir::stablehlo::ConstantOp>(b.getI32TensorAttr({1})));
      out = b.create<::mlir::stablehlo::SubtractOp>(dist, div_2);
      out = b.create<::mlir::stablehlo::AddOp>(out, div_2);
      out = b.create<::mlir::chlo::BroadcastAddOp>(out, op.getA(), nullptr);
    } else {
      auto floatTensor = cast<TensorType>(op.getType());
      auto intTensor = rngOp.getOutput().getType();
      auto constOne = b.create<::mlir::stablehlo::ConstantOp>(
          DenseFPElementsAttr::get(floatTensor, 1.0f));
      auto oneBits =
          b.create<::mlir::stablehlo::BitcastConvertOp>(intTensor, constOne);
      out = b.create<::mlir::stablehlo::OrOp>(bits, oneBits);
      APInt highBitVal(width, ((1ul << width) - 1) >> 2);
      auto highBit = b.create<::mlir::stablehlo::ConstantOp>(
          DenseIntElementsAttr::get(intTensor, highBitVal));
      out = b.create<::mlir::stablehlo::AndOp>(out, highBit);
      // Get in range [0, 1).
      out = b.create<::mlir::stablehlo::SubtractOp>(
          b.create<::mlir::stablehlo::BitcastConvertOp>(floatTensor, out),
          constOne);
      // Multiply and add to shift to the range [minval, maxval).
      auto range =
          b.create<::mlir::stablehlo::SubtractOp>(op.getB(), op.getA());
      out = b.create<::mlir::chlo::BroadcastMulOp>(out, range, nullptr);
      out = b.create<::mlir::chlo::BroadcastAddOp>(out, op.getA(), nullptr);
    }
    b.create<ml_program::GlobalStoreOp>(symbol, rngOp.getOutputState());

    rewriter.replaceOp(op, out);
    return success();
  }

  GlobalFn &getGlobal;
};

struct StatefulRngPass : public impl::StatefulRngBase<StatefulRngPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ml_program::MLProgramDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    ml_program::GlobalOp global;

    auto getGlobal = [&]() {
      if (global)
        return global;

      ModuleOp module = getOperation();
      OpBuilder globalBuilder(module.getBodyRegion());

      // Use an arbitrary seed value. This value is public so that if desired
      // the seed could be set on lowered program.
      std::vector<uint32_t> vals({12, 34, 56, 78});
      RankedTensorType ty =
          RankedTensorType::get(4, globalBuilder.getIntegerType(32));
      auto initValue = DenseIntElementsAttr::get(ty, vals);

      global = globalBuilder.create<ml_program::GlobalOp>(
          module.getLoc(), "global_hlo_rng_state", ty,
          /*is_mutable=*/true, initValue,
          /*visibility=*/globalBuilder.getStringAttr("public"));
      return global;
    };
    GlobalFn g = getGlobal;

    patterns.insert<ExpandRngUniform>(context, g);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStatefulRngPreprocessingPass() {
  return std::make_unique<StatefulRngPass>();
}

} // namespace mlir::iree_compiler::stablehlo
