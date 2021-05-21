// Copyright 2021 Google LLC
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

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

struct VerifyCompilerInputLegalityPass
    : public VerifyCompilerInputLegalityBase<VerifyCompilerInputLegalityPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(*context);
    OwningRewritePatternList conversionPatterns(&getContext());

    // Note that we would prefer allow-lists of what we positively support.
    // However, it is so common to sneak input-level ops into the pipeline
    // that we explicitly deny the dialects we know about.
    conversionTarget.addIllegalDialect<tosa::TosaDialect>();
    conversionTarget.addIllegalDialect<mhlo::MhloDialect>();
    conversionTarget.addIllegalDialect<chlo::HloClientDialect>();
    conversionTarget.addIllegalDialect<lmhlo::LmhloDialect>();

    // Exception: ApplyScaleOp is actually a lowered op on par with standard
    // dialect.
    conversionTarget.addLegalOp<tosa::ApplyScaleOp>();

    // NOTE: It is not fully illegal to tunnel input dialect ops through to
    // backends that expect them. When such situations arise, the container
    // op should be marked recursively legal here.

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      emitError(getOperation().getLoc())
          << "one or more illegal operations were found in the compiler input "
             "(are you missing an --iree-input-type= flag, or did you mean to "
             "pre-process through an IREE importer frontend?)";
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createVerifyCompilerInputLegality() {
  return std::make_unique<VerifyCompilerInputLegalityPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
