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

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(OwningRewritePatternList &patterns) {
  linalg::insertVectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                                      linalg::ContractionOpInterface>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), patterns.getContext())));
}

static Optional<SmallVector<int64_t, 4>> getNativeVectorSize(Operation *op) {
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // Map elementwise ops to vec4.
      SmallVector<int64_t, 4> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = 4;
      return nativeSize;
    }
  } else if (auto vt = dyn_cast<VectorTransferOpInterface>(op)) {
    auto rank = vt.getVectorType().getRank();
    SmallVector<int64_t, 4> nativeSize(rank, 1);
    nativeSize.back() = 4;
    return nativeSize;
  } else if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    unsigned lastParalleldim = 0;
    for (auto it : llvm::enumerate(contract.iterator_types())) {
      if (isParallelIterator(it.value())) lastParalleldim = it.index();
    }
    SmallVector<int64_t, 4> nativeSize(contract.iterator_types().size(), 1);
    nativeSize[lastParalleldim] = 4;
    return nativeSize;
  }
  return llvm::None;
}

static void populateVectorUnrollPatterns(OwningRewritePatternList &patterns) {
  patterns.add<vector::UnrollVectorPattern>(
      patterns.getContext(),
      vector::UnrollVectorOptions().setNativeShapeFn(getNativeVectorSize));
}

namespace {
struct VectorizationPass
    : public PassWrapper<VectorizationPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    {
      // Step 1. Vectorize
      OwningRewritePatternList vectorizationPatterns(context);
      populateVectorizationPatterns(vectorizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorizationPatterns));
    }
    // TODO: This should be a folding of Add into Contract in core but while
    // they live in different dialects, it is not possible without unnatural
    // dependencies.
    funcOp.walk([&](Operation *op) {
      if (auto contract = canonicalizeContractionAdd(op))
        op->replaceAllUsesWith(contract);
    });

    {
      // Step 2. Unroll the vetors to native size and canonicalize.
      OwningRewritePatternList vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));

      OwningRewritePatternList canonicalizationPatterns1(funcOp.getContext());
      vector::populateVectorToVectorCanonicalizationPatterns(
          canonicalizationPatterns1);
      vector::populateVectorToVectorTransformationPatterns(
          canonicalizationPatterns1);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns1));

      OwningRewritePatternList canonicalizationPatterns2(funcOp.getContext());
      vector::populateVectorSlicesLoweringPatterns(canonicalizationPatterns2);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns2));

      linalg::hoistRedundantVectorTransfers(funcOp);
    }
    {
      // Step 3. Canonicalize the transfer ops generated.
      RewritePatternSet vectorToLoopsPatterns(context);
      VectorTransferToSCFOptions vectorToSCFOptions;
      vectorToSCFOptions.setUnroll(true);
      populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                            vectorToSCFOptions);
      populateStdLegalizationPatternsForSPIRVLowering(vectorToLoopsPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorToLoopsPatterns));
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}

static PassRegistration<VectorizationPass> pass(
    "iree-codegen-cuda-vectorization", "Pass to convert linalg into Vector.");

}  // namespace iree_compiler
}  // namespace mlir
