// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

/// Currently, IREE requires `lowering_config`s to be propagated to all compute
/// ops within a dispatch region. This can be problematic for SME, as it does
/// not make sense to use 2D-scalability for all ops within a dispatch region.
/// The Arm SME target only supports 2D scalable outer products, so if it's not
/// an outer product, we can only scalably vectorize in one dimension.
///
/// The solution here is this pass (`2d-scalable-to-1d-scalable`) that runs just
/// before vectorization, that drops unsupported scalable tile/vector sizes,
/// producing loops of ops that will only be vectorized scalably in one
/// dimension. This allows earlier passes like `tile-and-fuse` to still function
/// correctly.
///
/// Take this simple example:
///
/// ```mlir
/// // Lowering configs propagated (from matmul):
/// linalg.fill {lowering_config = [[4], [4]]
/// linalg.matmul {lowering_config = [[4], [4], 1]
/// linalg.generic {lowering_config = [[4], [4]]
/// ```
/// Here the `linalg.generic` cannot be vectorized with 2D scalable vectors.
///
/// After `tile-and-fuse` (which requires consistent lowering configs):
/// ```mlir
/// scf.for i in range(0, 1000) step 4 x vscale {
///   scf.for j in range(0, 2000) step 4 x vscale {
///     linalg.fill {lowering_config = [[4], [4]]
///     for k in range(0, 100) step 1 {
///       linalg.matmul {lowering_config = [[4], [4], 1]
///     }
///     // 2D scalable vectorization unsupported here:
///     linalg.generic {lowering_config = [[4], [4]]
///   }
/// }
/// ```
///
/// Unsupported scalability removed (by `2d-scalable-to-1d-scalable`):
/// ```mlir
/// scf.for i in range(0, 1000) step 4 x vscale {
///   scf.for j in range(0, 2000) step 4 x vscale {
///     linalg.fill {lowering_config = [[4], [4]]
///     for k in range(0, 100) step 1 {
///       linalg.matmul {lowering_config = [[4], [4], 1]
///     }
///     // Insert a new loop:
///     for n in range(0, 4 x vscale) step 4 {
///        // Drop a scalable dim:
///        linalg.generic {lowering_config = [4, [4]]
///     }
///   }
/// }
/// ```
///
/// This can now be vectorized and lowered successfully, which produces a
/// dispatch that mixes SME and SVE.
class LLVMCPU2DScalableTo1DScalablePass
    : public LLVMCPU2DScalableTo1DScalableBase<
          LLVMCPU2DScalableTo1DScalablePass> {
public:
  using LLVMCPU2DScalableTo1DScalableBase::LLVMCPU2DScalableTo1DScalableBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

static bool opKnownToSupport2DScalableVectorizationWithArmSME(Operation *op) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeAOp, linalg::FillOp>(op);
}

// Note: It would be easy to parameterize this rewrite to convert N-D scalable
// operations to M-D scalable ones (where M < N). However this is currently not
// needed.
struct DropUnsupportedScalableDimsFromTilingInterfaceOps
    : public OpInterfaceRewritePattern<TilingInterface> {
  DropUnsupportedScalableDimsFromTilingInterfaceOps(MLIRContext *context,
                                                    bool assumeArmSME)
      : OpInterfaceRewritePattern(context), assumeArmSME(assumeArmSME) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    // Note: Which operations should have scalability dropped is specific to
    // ArmSME. The rest of this rewrite could be generic (though currently
    // there's no other targets that support > 1D scalability).
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
    bool isArmSME = assumeArmSME || hasSMEFeature(targetAttr);
    if (!isArmSME || opKnownToSupport2DScalableVectorizationWithArmSME(op))
      return failure();

    auto loweringConfigAttr = getLoweringConfig(op);
    if (!loweringConfigAttr)
      return failure();

    auto tileSizes = loweringConfigAttr.getTileSizeVals();
    auto scalableFlags = loweringConfigAttr.getScalableTileFlagVals();

    // 1. Drop scalable dimensions from leading tiling levels first (and leading
    // dimensions within tiling levels). This works out as dropping scalability
    // from leading dimensions of the vector type.
    int64_t firstTilingLevelWithScalableDims = -1;
    int64_t numScalableDims = 0;
    for (auto [level, levelScalableFlags] : llvm::enumerate(scalableFlags)) {
      numScalableDims += llvm::count(levelScalableFlags, true);
      if (numScalableDims > 0 && firstTilingLevelWithScalableDims == -1)
        firstTilingLevelWithScalableDims = level;
    }

    if (numScalableDims <= 1)
      return failure();

    auto levelTileSizes = tileSizes[firstTilingLevelWithScalableDims];
    auto levelScalableFlags = scalableFlags[firstTilingLevelWithScalableDims];

    SmallVector<int64_t> loopTileSizes;
    SmallVector<bool> newScalableFlags;
    for (auto [flag, size] : llvm::zip(levelScalableFlags, levelTileSizes)) {
      if (flag && numScalableDims >= 2) {
        --numScalableDims;
        loopTileSizes.push_back(size);
        newScalableFlags.push_back(false);
      } else {
        loopTileSizes.push_back(0);
        newScalableFlags.push_back(flag);
      }
    }

    // 2. Re-tile the operation with some scalability dropped. This introduces
    // loops for previously scalable vector/tile sizes.
    scf::SCFTilingOptions options{};
    setSCFTileSizes(options, op, loopTileSizes, {});
    auto tilingResult =
        scf::tileUsingSCF(rewriter, cast<TilingInterface>(op), options);
    if (failed(tilingResult))
      return failure();

    // 3. Update the lowering config of the new tiled operations.
    scalableFlags[firstTilingLevelWithScalableDims] = newScalableFlags;
    auto newLoweringConfig = IREE::Codegen::LoweringConfigAttr::get(
        getContext(), tileSizes, scalableFlags);
    for (auto *newOp : tilingResult->tiledOps) {
      if (isa<TilingInterface>(newOp))
        setLoweringConfig(newOp, newLoweringConfig);
    }

    rewriter.replaceOp(op, tilingResult->replacements);
    return success();
  };

private:
  bool assumeArmSME{false};
};

void LLVMCPU2DScalableTo1DScalablePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<DropUnsupportedScalableDimsFromTilingInterfaceOps>(
      patterns.getContext(), assumeArmSME);

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPU2DScalableTo1DScalablePass() {
  return std::make_unique<LLVMCPU2DScalableTo1DScalablePass>();
}

} // namespace mlir::iree_compiler
