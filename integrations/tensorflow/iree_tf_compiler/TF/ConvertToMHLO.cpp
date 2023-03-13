// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

// This is a customized version of the TF to XLA lowering in:
//    tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc
// It does not require the same number of options as we can hardcode as the pass
// the IREE requires.
class ConvertToMHLOPass
    : public PassWrapper<ConvertToMHLOPass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::TF::TensorFlowDialect,
                    mlir::tf_executor::TensorFlowExecutorDialect,
                    mlir::tf_device::TensorFlowDeviceDialect,
                    mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    chlo::ChloDialect, mhlo::MhloDialect, shape::ShapeDialect,
                    mlir::arith::ArithDialect, func::FuncDialect>();
  }

  StringRef getArgument() const override { return "iree-tf-convert-to-mhlo"; }

  StringRef getDescription() const override {
    return "Converts from TensorFlow to the XLA MHLO dialect";
  }

 public:
  ConvertToMHLOPass() = default;
  ConvertToMHLOPass(const ConvertToMHLOPass &) {}

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = op.getContext();

    // Lower TF Patterns must be separate from canonocalization patterns as
    // they are sometimes inversions of eachother.
    RewritePatternSet lowerTfPatterns(&getContext());
    mlir::TF::PopulateTFLoweringBeforeHLOPatterns(context, &lowerTfPatterns);

    RewritePatternSet canonicalizePatterns(&getContext());
    for (auto op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(canonicalizePatterns, context);
    }

    RewritePatternSet patterns(&getContext());
    // Note that the `OperationConverter` orders patterns lexicographically by:
    // 1) Ascending legalization depth (i.e., minimum number of patterns
    // necessary to arrive at conversion target).
    // 2) Descending pattern benefit.
    // 3) Order of patterns in `RewritePatternSet`.

    // Add TF->HLO legalization patterns.
    mhlo::PopulateLegalizeTfPatterns(context, &patterns);

    // IREE Direct TF lowerings.
    populateDirectLoweringPatterns(context, patterns);

    // TF::PopulateLoweringTFPatterns(context, &patterns);

    // ConstantLike op is convenient to create splat constants, but is
    // canonicalized to plain HLO constant if statically shaped. Add the
    // canonicalization pattern to pattern list to enable multi-hop lowering.
    chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addLegalDialect<chlo::ChloDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target
        .addLegalDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<mlir::func::CallOp>();
    target.addLegalOp<mlir::tensor::CastOp>();
    target.addLegalOp<mlir::tensor::DimOp>();

    // TODO(suderman): Enable logicistic op for lowering once the op is
    // supported in IREE. Also, remove the numerically unstable ConvertSigmoidOp
    // pattern in the legalize-tf pass.
    target.addIllegalOp<mhlo::LogisticOp>();

    // In general, IREE does not support DynamicBroadcastInDim ops that do not
    // resolve to a static form. This excludes any TF2XLA expansions which
    // we ultimately lack a linalg lowering for. Matches the corresponding
    // condition in legalize_to_linalg.cc for this op.
    target.addDynamicallyLegalOp<mhlo::DynamicBroadcastInDimOp>(
        [](mhlo::DynamicBroadcastInDimOp op) {
          if (auto t = op.getOperand()
                           .getType()
                           .template dyn_cast<RankedTensorType>()) {
            if (t.hasStaticShape()) {
              return true;
            }
          }
          return false;
        });

    DenseSet<Operation *> prevUnconvertedOps;
    DenseSet<Operation *> unconvertedOps;

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    FrozenRewritePatternSet frozenCanonicalizePatterns(
        std::move(canonicalizePatterns));
    FrozenRewritePatternSet frozenTfPatterns(std::move(lowerTfPatterns));
    while (true) {
      if (failed(
              applyPatternsAndFoldGreedily(op, frozenCanonicalizePatterns))) {
        return signalPassFailure();
      }

      if (failed(applyPatternsAndFoldGreedily(op, frozenTfPatterns))) {
        return signalPassFailure();
      }

      if (failed(applyPartialConversion(op, target, frozenPatterns,
                                        &unconvertedOps))) {
        return signalPassFailure();
      }

      if (prevUnconvertedOps == unconvertedOps) break;
      prevUnconvertedOps = std::move(unconvertedOps);
    }
  }

 private:
  Option<bool> allow_partial_conversion_{
      *this, "allow-partial-conversion",
      llvm::cl::desc("Allow operations that can't be legalized."),
      llvm::cl::init(false)};
  Option<bool> legalize_chlo_{
      *this, "legalize-chlo",
      llvm::cl::desc(
          "Also legalizes intermediate chlo ops to hlo (default true)"),
      llvm::cl::init(false)};
  Option<bool> use_tf2xla_fallback_{
      *this, "use-tf2xla-fallback",
      llvm::cl::desc(
          "Also use TF2XLA fallback for legalization (default false)"),
      llvm::cl::init(false)};
  Option<std::string> device_type_{
      *this, "device-type",
      llvm::cl::desc(
          "The device type used by TF2XLA fallback. Must be specified if "
          "use-tf2xla-fallback is true, otherwise not used."),
      llvm::cl::init("INVALID_DEVICE_TYPE")};
};

std::unique_ptr<Pass> createConvertToMHLOPass() {
  return std::make_unique<ConvertToMHLOPass>();
}

static PassRegistration<ConvertToMHLOPass> pass;

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
