// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVFinalVectorLowering.cpp ---------------------------------------===//
//
// This pass hosts final steps towards lowering vectors ops to meet SPIR-V
// requirements--it applies vector lowering patterns to convert vector ops
// to more basic forms.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-final-vector-lowering"

namespace mlir {
namespace iree_compiler {
namespace {

void debugPrint(func::FuncOp funcOp, const char *message) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << message << " ---//\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

class SPIRVFinalVectorLoweringPass
    : public SPIRVFinalVectorLoweringBase<SPIRVFinalVectorLoweringPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // vector.gather lowering patterns target scf ops.
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();

    // Lower vector transfer permutation map.
    {
      RewritePatternSet patterns(context);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns,
                                                                 context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after lowering transfer ops");

    // Lower vector broadcast/transpose and contraction.
    {
      RewritePatternSet patterns(context);
      auto options = vector::VectorTransformsOptions()
                         .setVectorTransformsOptions(
                             vector::VectorContractLowering::OuterProduct)
                         .setVectorTransposeLowering(
                             vector::VectorTransposeLowering::EltWise);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(patterns, options);
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      vector::populateVectorTransposeLoweringPatterns(patterns, options);
      vector::populateVectorGatherLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    debugPrint(funcOp, "after lowering various vector ops");

    // Run all sorts of canonicalization patterns to clean up again.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      vector::InsertOp::getCanonicalizationPatterns(patterns, context);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
      populateVectorTransferTensorSliceTransforms(patterns);
      vector::ReductionOp::getCanonicalizationPatterns(patterns, context);
      scf::IfOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVFinalVectorLoweringPass() {
  return std::make_unique<SPIRVFinalVectorLoweringPass>();
}

} // namespace iree_compiler
} // namespace mlir
