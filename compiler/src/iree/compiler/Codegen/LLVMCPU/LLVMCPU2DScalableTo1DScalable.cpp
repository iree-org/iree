// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPU2DSCALABLETO1DSCALABLEPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

/// Currently, IREE requires `lowering_config`s to be propagated to all compute
/// ops within a dispatch region. This can be problematic for SME which only
/// supports 2D scalable outer product operations -- if an operation cannot be
/// lowered to an outer product, we can only scalably vectorize it in one
/// dimension.
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
    : public impl::LLVMCPU2DScalableTo1DScalablePassBase<
          LLVMCPU2DScalableTo1DScalablePass> {
public:
  using impl::LLVMCPU2DScalableTo1DScalablePassBase<
      LLVMCPU2DScalableTo1DScalablePass>::LLVMCPU2DScalableTo1DScalablePassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }

  void runOnOperation() override;
};

static bool opKnownToSupport2DScalableVectorizationWithArmSME(Operation *op) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
    return isLinalgGeneric2DTranspose(genericOp);
  return isa<linalg::MatmulOp, linalg::MatmulTransposeAOp, linalg::FillOp>(op);
}

/// Returns a new `LoweringConfigAttr`, with the tile sizes of vector
/// dimensions, set to `sizes`, and the corresponding scalability set to
/// `scalableFlags`.
static IREE::CPU::LoweringConfigAttr getLoweringConfigWithNewVectorSizes(
    IREE::CPU::LoweringConfigAttr loweringConfig, ArrayRef<int64_t> sizes,
    ArrayRef<bool> scalableFlags) {
  using TilingLevel = IREE::CPU::TilingLevel;
  MLIRContext *ctx = loweringConfig.getContext();
  SmallVector<NamedAttribute> items;
  for (unsigned i = 0, e = TilingLevel::MaxNumTileLevels; i < e; ++i) {
    auto level = static_cast<TilingLevel>(i);
    if (!loweringConfig.hasTilingLevel(level)) {
      continue;
    }
    switch (level) {
    case TilingLevel::DistributionTiles:
    case TilingLevel::CacheParallelTiles:
    case TilingLevel::CacheReductionTiles: {
      items.emplace_back(IREE::CPU::getTilingLevelName(level),
                         loweringConfig.getTilingLevelAttr(i));
      break;
    }
    case TilingLevel::VectorCommonParallelTiles:
    case TilingLevel::VectorReductionTiles:
    case TilingLevel::VectorInnerParallelTiles: {
      auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
          loweringConfig.getTilingLevelAttr(i));
      SmallVector<int64_t> newSizes(attr.getSizes());
      SmallVector<bool> newScalableFlags(attr.getScalableFlags());
      newScalableFlags.resize(newSizes.size(), false);
      for (auto [idx, size] : llvm::enumerate(newSizes)) {
        if (size == 0) {
          continue;
        }
        newSizes[idx] = sizes[idx];
        newScalableFlags[idx] = scalableFlags[idx];
      }
      auto newLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
          ctx, newSizes, attr.getInterchange(), newScalableFlags);
      items.emplace_back(IREE::CPU::getTilingLevelName(level), newLevel);
      break;
    }
    case TilingLevel::MaxNumTileLevels:
    case TilingLevel::InvalidLevel:
      break;
    };
  }
  return IREE::CPU::LoweringConfigAttr::get(ctx, items);
}

// Note: It would be easy to parameterize this rewrite to convert N-D
// scalable operations to M-D scalable ones (where M < N). However this is
// currently not needed.
static LogicalResult
dropScalabilityFromUnsupportedOperations(mlir::FunctionOpInterface funcOp,
                                         bool assumeArmSME = false) {
  // Note: Which operations should have scalability dropped is specific to
  // ArmSME. The rest of this rewrite could be generic (though currently
  // there's no other targets that support > 1D scalability).
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  bool isArmSME = assumeArmSME ||
                  (targetAttr && hasSMEFeature(targetAttr.getConfiguration()));
  if (!isArmSME) {
    return success();
  }

  SmallVector<TilingInterface> computeOps;
  funcOp.walk([&](TilingInterface op) {
    if (!opKnownToSupport2DScalableVectorizationWithArmSME(op)) {
      computeOps.push_back(op);
    }
  });

  for (TilingInterface tilingOp : computeOps) {
    auto loweringConfigAttr =
        getLoweringConfig<IREE::CPU::LoweringConfigAttr>(tilingOp);
    if (!loweringConfigAttr) {
      continue;
    }

    std::optional<SmallVector<int64_t>> vectorSizes =
        loweringConfigAttr.getVectorSizes();
    SmallVector<bool> scalableFlags =
        loweringConfigAttr.getVectorScalableFlags();
    int64_t numScalableDims = llvm::count(scalableFlags, true);
    if (numScalableDims <= 1) {
      continue;
    }

    SmallVector<int64_t> loopTileSizes;
    SmallVector<bool> newScalableFlags;
    for (auto [flag, size] : llvm::zip_equal(scalableFlags, *vectorSizes)) {
      if (flag && numScalableDims >= 2) {
        --numScalableDims;
        loopTileSizes.push_back(size);
        newScalableFlags.push_back(false);
      } else {
        loopTileSizes.push_back(0);
        newScalableFlags.push_back(flag);
      }
    }

    IRRewriter rewriter(tilingOp->getContext());
    rewriter.setInsertionPoint(tilingOp);

    // Re-tile the operation with some scalability dropped. This introduces
    // loops for previously scalable vector/tile sizes.
    scf::SCFTilingOptions options;
    setSCFTileSizes(options, tilingOp, loopTileSizes, /*tileScalableFlags=*/{});
    auto tilingResult = scf::tileUsingSCF(rewriter, tilingOp, options);
    if (failed(tilingResult))
      return failure();

    // Update the lowering config of the new tiled operations.
    IREE::CPU::LoweringConfigAttr newLoweringConfig =
        getLoweringConfigWithNewVectorSizes(loweringConfigAttr, *vectorSizes,
                                            newScalableFlags);
    for (auto *newOp : tilingResult->tiledOps) {
      if (isa<TilingInterface>(newOp))
        setLoweringConfig(newOp, newLoweringConfig);
    }

    rewriter.replaceOp(tilingOp, tilingResult->replacements);
  }
  return success();
}

void LLVMCPU2DScalableTo1DScalablePass::runOnOperation() {
  if (failed(dropScalabilityFromUnsupportedOperations(getOperation(),
                                                      assumeArmSME)))
    signalPassFailure();
}

} // namespace
} // namespace mlir::iree_compiler
