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

struct ConvertHLOToLinalgOnTensorsPass
    : public PassWrapper<ConvertHLOToLinalgOnTensorsPass, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateHLOToLinalgOnTensorsConversionPatterns(&getContext(), patterns);

    ConversionTarget target(getContext());
    // Allow constant to appear in Linalg op regions.
    target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) -> bool {
      return isa<linalg::LinalgOp>(op.getOperation()->getParentOp());
    });
    // Don't convert the body of reduction ops.
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>(
        Optional<ConversionTarget::DynamicLegalityCallbackFn>(
            [](Operation* op) {
              auto parentOp = op->getParentRegion()->getParentOp();
              return isa<mhlo::ReduceOp>(parentOp) ||
                     isa<mhlo::ReduceWindowOp>(parentOp);
            }));
    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyPartialConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext* context, OwningRewritePatternList& patterns) {
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
