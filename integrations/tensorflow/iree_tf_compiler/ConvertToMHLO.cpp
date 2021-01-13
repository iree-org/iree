// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree_tf_compiler/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

// This is a customized version of the TF to XLA lowering in:
//    tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc
// It does not require the same number of options as we can hardcode as the pass
// the IREE requires.
class ConvertToMHLOPass : public PassWrapper<ConvertToMHLOPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TF::TensorFlowDialect,
                    mlir::tf_executor::TensorFlowExecutorDialect,
                    mlir::tf_device::TensorFlowDeviceDialect,
                    mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    chlo::HloClientDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, StandardOpsDialect>();
  }

 public:
  ConvertToMHLOPass() = default;
  ConvertToMHLOPass(const ConvertToMHLOPass &) {}

  void runOnFunction() override {
    auto op = getFunction();
    MLIRContext *context = op.getContext();

    // Lower TF Patterns must be separate from canonocalization patterns as
    // they are sometimes inversions of eachother.
    OwningRewritePatternList lowerTfPatterns;
    mlir::TF::PopulateLoweringTFPatterns(context, &lowerTfPatterns);

    OwningRewritePatternList canonicalizePatterns;
    for (auto *op : context->getRegisteredOperations()) {
      op->getCanonicalizationPatterns(canonicalizePatterns, context);
    }

    OwningRewritePatternList patterns;
    // Note that the `OperationConverter` orders patterns lexicographically by:
    // 1) Ascending legalization depth (i.e., minimum number of patterns
    // necessary to arrive at conversion target).
    // 2) Descending pattern benefit.
    // 3) Order of patterns in `OwningRewritePatternList`.

    // Add TF->HLO legalization patterns.
    mhlo::PopulateLegalizeTfPatterns(context, &patterns);

    // TF::PopulateLoweringTFPatterns(context, &patterns);

    // Populate with CHLO->HLO lowerings to account for TF ops legalized to
    // CHLO first.
    chlo::PopulateLegalizeChloToHloPatterns(context, &patterns);

    // ConstantLike op is convenient to create splat constants, but is
    // canonicalized to plain HLO constant if statically shaped. Add the
    // canonicalization pattern to pattern list to enable multi-hop lowering.
    chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addIllegalDialect<chlo::HloClientDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalOp<mlir::CallOp>();
    target.addLegalOp<mlir::tensor::CastOp>();

    DenseSet<Operation *> prevUnconvertedOps;
    DenseSet<Operation *> unconvertedOps;

    FrozenRewritePatternList frozenPatterns(std::move(patterns));
    FrozenRewritePatternList frozenCanonicalizePatterns(
        std::move(canonicalizePatterns));
    FrozenRewritePatternList frozenTfPatterns(std::move(lowerTfPatterns));
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
      llvm::cl::init(true)};
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

std::unique_ptr<FunctionPass> createConvertToMHLOPass() {
  return std::make_unique<ConvertToMHLOPass>();
}

static PassRegistration<ConvertToMHLOPass> pass(
    "iree-tf-convert-to-mhlo",
    "Converts from TensorFlow to the XLA MHLO dialect");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
