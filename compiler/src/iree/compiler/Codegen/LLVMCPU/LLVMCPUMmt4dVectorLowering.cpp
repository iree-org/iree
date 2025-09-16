// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-mmt4d-vector-lowering"

// A flag to switch between inline asm and intrinsics while we develop these two
// parallel paths.
static llvm::cl::opt<bool> clMmt4dUseIntrinsics(
    "iree-codegen-mmt4d-use-intrinsics",
    llvm::cl::desc("Whether to use instrinsics when lowering vector contracts "
                   "generated from mmt4d matmuls (as opposed to inline asm). "
                   "Not for production use."),
    llvm::cl::init(false));

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUMMT4DVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
struct LLVMCPUMmt4dVectorLoweringPass
    : public impl::LLVMCPUMmt4dVectorLoweringPassBase<
          LLVMCPUMmt4dVectorLoweringPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void LLVMCPUMmt4dVectorLoweringPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  std::optional<int64_t> numLoops;
  funcOp.walk([&](vector::ContractionOp op) {
    if (numLoops)
      return signalPassFailure();
    numLoops = op.getIndexingMapsArray()[0].getNumDims();
  });
  // No vector.contract op to optimize.
  if (!numLoops)
    return;

  {
    // Fold consumer add ops into the contraction op itself.
    RewritePatternSet canonicalizationPatterns(context);
    vector::ContractionOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                       context);
    if (failed(applyPatternsGreedily(funcOp,
                                     std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs()
          << "\n--- After folding consumer add ops into contraction op "
             "iteself ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    RewritePatternSet castAwayUnitDimPatterns(&getContext());
    vector::populateCastAwayVectorLeadingOneDimPatterns(
        castAwayUnitDimPatterns);
    if (failed(applyPatternsGreedily(funcOp,
                                     std::move(castAwayUnitDimPatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet reductionToContractPatterns(&getContext());
    vector::populateVectorReductionToContractPatterns(
        reductionToContractPatterns);
    vector::ExtractOp::getCanonicalizationPatterns(reductionToContractPatterns,
                                                   context);
    if (failed(applyPatternsGreedily(funcOp,
                                     std::move(reductionToContractPatterns)))) {
      return signalPassFailure();
    }
  }

  if (enableVectorContractCustomKernels) {
    // Special-case vector.contract codegen paths. This needs to happen
    // just before the generic vector ops lowerings.
    RewritePatternSet patterns(context);
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    populateVectorContractCustomKernelsPatterns(target, patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}
} // namespace mlir::iree_compiler
