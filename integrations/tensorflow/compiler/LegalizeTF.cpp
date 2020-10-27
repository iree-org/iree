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

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace mhlo {
namespace {

// This is a customizer version of the TF to XLA lowering in:
//    tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc
// It does not require the same number of options as we can hardcode as the pass
// the IREE requires.
class LegalizeTF : public PassWrapper<LegalizeTF, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<chlo::HloClientDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, StandardOpsDialect>();
  }

 public:
  LegalizeTF() = default;
  LegalizeTF(const LegalizeTF &) {}

  /// Performs the lowering to XLA dialect.
  void runOnFunction() override {
    auto op = getFunction();
    MLIRContext *context = op.getContext();
    OwningRewritePatternList canonicalizePatterns;
    for (auto *op : context->getRegisteredOperations())
      op->getCanonicalizationPatterns(canonicalizePatterns, context);

    OwningRewritePatternList patterns;
    // Note that the `OperationConverter` orders patterns lexicographically by:
    // 1) Ascending legalization depth (i.e., minimum number of patterns
    // necessary to arrive at conversion target).
    // 2) Descending pattern benefit.
    // 3) Order of patterns in `OwningRewritePatternList`.

    // Add TF->HLO legalization patterns.
    PopulateLegalizeTfPatterns(context, &patterns);

    // Add TF->TF lowering patterns.
    TF::PopulateLoweringTFPatterns(context, &patterns);

    // Populate with CHLO->HLO lowerings to account for TF ops legalized to
    // CHLO first.
    chlo::PopulateLegalizeChloToHloPatterns(context, &patterns);

    // ConstantLike op is convenient to create splat constants, but is
    // canonicalized to plain HLO constant if statically shaped. Add the
    // canonicalization pattern to pattern list to enable multi-hop lowering.
    chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addIllegalDialect<chlo::HloClientDialect>();
    target.addLegalDialect<MhloDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalOp<CallOp>();
    target.addLegalOp<TensorCastOp>();

    DenseSet<Operation *> prevUnconvertedOps;
    DenseSet<Operation *> unconvertedOps;

    FrozenRewritePatternList frozenPatterns(std::move(patterns));
    FrozenRewritePatternList frozenCanonicalizePatterns(
        std::move(canonicalizePatterns));
    while (true) {
      if (failed(applyPartialConversion(op, target, frozenPatterns,
                                        &unconvertedOps))) {
        return signalPassFailure();
      }

      if (prevUnconvertedOps == unconvertedOps) break;

      prevUnconvertedOps = std::move(unconvertedOps);
      if (failed(
              applyPatternsAndFoldGreedily(op, frozenCanonicalizePatterns))) {
        return signalPassFailure();
      }
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

static PassRegistration<LegalizeTF> pass(
    "iree-xla-legalize-tf", "Legalize from TensorFlow to the XLA dialect");

}  // namespace
}  // namespace mhlo
}  // namespace mlir
