// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-vectorization"

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  linalg::LinalgVectorizationOptions opt;
  linalg::LinalgTransformationFilter f(
      StringAttr::get(ctx, getVectorizeMarker()));
  f.setMatchByDefault();
  linalg::VectorizationPatterns<linalg::FillOp, linalg::GenericOp>::insert(
      patterns, opt, f);
  patterns.add<linalg::CopyVectorizationPattern>(ctx);
  patterns.add<linalg::LinalgVectorizationPattern>(
      ctx, f.addOpFilter<linalg::ContractionOpInterface>(), opt);
}

static Optional<SmallVector<int64_t, 4>> getGPUNativeVectorSize(
    Operation *op, int64_t nativeVector) {
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // Map elementwise ops to vec4.
      SmallVector<int64_t, 4> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = nativeVector;
      return nativeSize;
    }
  } else if (auto vt = dyn_cast<VectorTransferOpInterface>(op)) {
    auto rank = vt.getVectorType().getRank();
    SmallVector<int64_t, 4> nativeSize(rank, 1);
    // Load 4 elements on the most inner dimension.
    for (auto dim : llvm::enumerate(vt.permutation_map().getResults())) {
      if (auto dimExpr = dim.value().dyn_cast<AffineDimExpr>()) {
        if (dimExpr.getPosition() == vt.permutation_map().getNumDims() - 1) {
          nativeSize[dim.index()] = nativeVector;
        }
      }
    }
    return nativeSize;
  } else if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    unsigned lastParalleldim = 0;
    for (auto it : llvm::enumerate(contract.getIteratorTypes())) {
      if (isParallelIterator(it.value())) lastParalleldim = it.index();
    }
    SmallVector<int64_t, 4> nativeSize(contract.getIteratorTypes().size(), 1);
    nativeSize[lastParalleldim] = nativeVector;
    return nativeSize;
  }
  return llvm::None;
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                         int64_t nativeVector) {
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions().setNativeShapeFn(
                    [nativeVector](Operation *op) {
                      return getGPUNativeVectorSize(op, nativeVector);
                    }));
}

namespace {
struct LLVMGPUVectorizationPass
    : public LLVMGPUVectorizationBase<LLVMGPUVectorizationPass> {
  LLVMGPUVectorizationPass(int64_t nativeVector, bool generateContract) {
    this->nativeVector = nativeVector;
    this->generateContract = generateContract;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    {
      // Step 1. Vectorize
      RewritePatternSet vectorizationPatterns(context);
      populateVectorizationPatterns(vectorizationPatterns);
      if (generateContract) {
        vector::populateVectorTransferPermutationMapLoweringPatterns(
            vectorizationPatterns);
        vector::populateVectorReductionToContractPatterns(
            vectorizationPatterns);
      }
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(vectorizationPatterns)))) {
        return signalPassFailure();
      }

      // Fold consumer add ops into the contraction op itself.
      RewritePatternSet canonicalizationPatterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }

      if (nativeVector > 0) {
        RewritePatternSet vectorUnrollPatterns(context);
        populateVectorUnrollPatterns(vectorUnrollPatterns, nativeVector);
        if (failed(applyPatternsAndFoldGreedily(
                funcOp, std::move(vectorUnrollPatterns)))) {
          return signalPassFailure();
        }
      }
      DEBUG_WITH_TYPE(DEBUG_TYPE, {
        llvm::dbgs() << "\n--- After Step 1: Vectorization ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }
    {
      // Step 2. Lower transfer op to canonical form.
      RewritePatternSet lowerTransferOpPatterns(funcOp.getContext());
      vector::populateVectorToVectorCanonicalizationPatterns(
          lowerTransferOpPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(lowerTransferOpPatterns)))) {
        return signalPassFailure();
      }
      DEBUG_WITH_TYPE(DEBUG_TYPE, {
        llvm::dbgs()
            << "\n--- After Step 2: Lower transfer op to canonical form. ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      // Step 3. Canonicalize.
      RewritePatternSet canonicalizationPatterns(funcOp.getContext());
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(
          canonicalizationPatterns, canonicalizationPatterns.getContext());
      vector::populateVectorToVectorCanonicalizationPatterns(
          canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
        return signalPassFailure();
      }
      DEBUG_WITH_TYPE(DEBUG_TYPE, {
        llvm::dbgs() << "\n--- After Step 3: Canonicalize. ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorizationPass(
    int64_t nativeVector, bool generateContract) {
  return std::make_unique<LLVMGPUVectorizationPass>(nativeVector,
                                                    generateContract);
}

}  // namespace iree_compiler
}  // namespace mlir
