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

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {
namespace {

// Convert mhlo.is_finite to linalg.generic for float32 values ranked inputs.
class LowerHLOIsFiniteF32OpToLinalg
    : public OpRewritePattern<mhlo::IsFiniteOp> {
 public:
  using OpRewritePattern<mhlo::IsFiniteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::IsFiniteOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = op.x().getType().dyn_cast_or_null<RankedTensorType>();
    if (!inputType || inputType.getElementType() != rewriter.getF32Type()) {
      return failure();
    }
    auto loc = op.getLoc();
    auto rank = inputType.getRank();
    auto identityMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    SmallVector<StringRef, 3> loopAttributeTypes(rank, "parallel");

    auto result = rewriter.create<linalg::GenericOp>(
        loc, op.y().getType(), /*inputs*/ op.x(),
        /*outputs*/ ValueRange{},
        /*intTensors*/ ValueRange{},
        ArrayRef<AffineMap>{identityMap, identityMap}, loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto constOp = nestedBuilder.create<ConstantOp>(
              nestedLoc,
              rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity()));
          Value cmp = nestedBuilder.create<CmpFOp>(
              nestedLoc, CmpFPredicate::ONE, args[0], constOp);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, cmp);
        });
    rewriter.replaceOp(op, result.getResults()[0]);
    return success();
  }
};

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, mhlo::MhloDialect>();
  }

  void runOnFunction() override {
    MLIRContext *context = &getContext();

    OwningRewritePatternList patterns;
    populateHLOToLinalgOnTensorsConversionPatterns(context, patterns);

    ConversionTarget target(getContext());
    // Allow constant to appear in Linalg op regions.
    target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) -> bool {
      return isa<linalg::LinalgOp>(op.getOperation()->getParentOp());
    });
    // Don't convert the body of reduction ops.
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            [](Operation *op) {
              auto parentOp = op->getParentRegion()->getParentOp();
              return isa<mhlo::ReduceOp>(parentOp) ||
                     isa<mhlo::ReduceWindowOp>(parentOp);
            }));
    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<LowerHLOIsFiniteF32OpToLinalg>(context);
  mhlo::populateHLOToLinalgConversionPattern(context, &patterns);
}

std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertHLOToLinalgOnTensorsPass>();
}

static PassRegistration<ConvertHLOToLinalgOnTensorsPass> legalize_pass(
    "iree-codegen-hlo-to-linalg-on-tensors",
    "Convert from XLA-HLO ops to Linalg ops on tensors");

}  // namespace iree_compiler
}  // namespace mlir
