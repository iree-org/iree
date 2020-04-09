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

#include <memory>

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
#include "iree/compiler/Translation/CodegenPasses/LinalgVectorTransformPatterns.h.inc"

namespace {
struct IREELinalgVectorTransformPass
    : public PassWrapper<IREELinalgVectorTransformPass, FunctionPass> {
  void runOnFunction() override;
};

void IREELinalgVectorTransformPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto funcOp = getFunction();
  populateWithGenerated(&getContext(), &patterns);
  applyPatternsGreedily(funcOp, patterns);
}

static PassRegistration<IREELinalgVectorTransformPass> pass(
    "iree-linalg-vector-transforms", "Lower linalg to vector dialect");

}  // namespace
}  // namespace iree_compiler
}  // namespace mlir
