// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVVectorize.cpp -------------------------------------------------===//
//
// This pass vectorizes Linalg ops with buffer semantics.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-vectorize"

namespace mlir {
namespace iree_compiler {
namespace {

Optional<SmallVector<int64_t, 4>> getSPIRVNativeVectorSize(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // Use 4-element vectors for elementwise ops.
      SmallVector<int64_t, 4> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = 4;
      return nativeSize;
    }
  } else if (auto vtOp = dyn_cast<VectorTransferOpInterface>(op)) {
    auto rank = vtOp.getVectorType().getRank();
    SmallVector<int64_t, 4> nativeSize(rank, 1);
    for (auto dim : llvm::enumerate(vtOp.permutation_map().getResults())) {
      if (auto dimExpr = dim.value().dyn_cast<AffineDimExpr>()) {
        if (dimExpr.getPosition() == vtOp.permutation_map().getNumDims() - 1)
          nativeSize[dim.index()] = 4;
      }
    }
    return nativeSize;
  } else if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    unsigned lastParalleldim = 0;
    for (auto it : llvm::enumerate(contractOp.iterator_types())) {
      if (isParallelIterator(it.value())) lastParalleldim = it.index();
    }
    SmallVector<int64_t, 4> nativeSize(contractOp.iterator_types().size(), 1);
    nativeSize[lastParalleldim] = 4;
    // Map to vec4 fma operations.
    return nativeSize;
  }
  return llvm::None;
}

/// Add patterns to vectorize Linalg ops with vectorization marker.
void populateVectorizationPatterns(MLIRContext *context,
                                   RewritePatternSet &patterns) {
  linalg::insertVectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                                      linalg::ContractionOpInterface>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), context)));
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(MLIRContext *context,
                                  RewritePatternSet &patterns) {
  vector::populateVectorUnrollPatterns(
      patterns,
      vector::UnrollVectorOptions().setNativeShapeFn(getSPIRVNativeVectorSize));
}

/// Vectorizes Linalg ops on buffer semantics.
class SPIRVVectorizePass : public SPIRVVectorizeBase<SPIRVVectorizePass> {
 public:
  SPIRVVectorizePass() = default;
  SPIRVVectorizePass(const SPIRVVectorizePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FuncOp funcOp = getOperation();

    {
      RewritePatternSet vectorizationPatterns(&getContext());
      populateVectorizationPatterns(context, vectorizationPatterns);
      populateLinalgToVectorVectorizeConvPatterns(context,
                                                  vectorizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorizationPatterns));

      // Fold consumer add ops into the contraction op itself.
      RewritePatternSet canonicalizationPatterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After vectorization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet vectorUnrollPatterns(funcOp.getContext());
      populateVectorUnrollPatterns(funcOp.getContext(), vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    linalg::hoistRedundantVectorTransfersOnTensor(funcOp);
    linalg::hoistRedundantVectorTransfers(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After hoisting vector transfers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet canonicalizationPatterns(funcOp.getContext());
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          canonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After canonicalizing vectors ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet contractLoweringPatterns(funcOp.getContext());
      vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorContractLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(contractLoweringPatterns));
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After handling contraction ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSPIRVVectorizePass() {
  return std::make_unique<SPIRVVectorizePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
