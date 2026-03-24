// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-codegen-gpu-pack-partial-reductions"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPACKPARTIALREDUCTIONSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct GPUPackPartialReductionsPass final
    : impl::GPUPackPartialReductionsPassBase<GPUPackPartialReductionsPass> {
  using Base::Base;
  void runOnOperation() override;
};

/// For each reduction dimension d where partial_reduction[d] > 1 and
/// thread[d] > 1, pack with inner_tile = thread[d]. This splits the
/// reduction dimension into [outer, inner] via linalg::pack.
///
/// After packing, update the lowering config to account for the new
/// inner dimensions appended at the end of the iteration space.
static LogicalResult packPartialReduction(IRRewriter &rewriter,
                                          linalg::LinalgOp linalgOp) {
  auto loweringConfig =
      getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
  if (!loweringConfig) {
    return success();
  }

  SmallVector<int64_t> partialReduction =
      loweringConfig.getStaticTilingLevelSizes(
          llvm::to_underlying(IREE::GPU::TilingLevel::PartialReduction),
          linalgOp);
  SmallVector<int64_t> threadSizes = loweringConfig.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Thread), linalgOp);

  if (partialReduction.empty() || threadSizes.empty()) {
    return success();
  }

  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  if (reductionDims.empty()) {
    return success();
  }

  unsigned numLoops = linalgOp.getNumLoops();

  // Find reduction dims that need packing: partial_reduction[d] > 1 and
  // thread[d] > 1.
  SmallVector<unsigned> dimsToPackVec;
  for (unsigned d : reductionDims) {
    if (d < partialReduction.size() && d < threadSizes.size() &&
        partialReduction[d] > 1 && threadSizes[d] > 1) {
      dimsToPackVec.push_back(d);
    }
  }

  if (dimsToPackVec.empty()) {
    return success();
  }

  // Build packed sizes: 0 for dims we don't pack, thread[d] for dims we pack.
  SmallVector<OpFoldResult> packedSizes(numLoops, rewriter.getIndexAttr(0));
  for (unsigned d : dimsToPackVec) {
    packedSizes[d] = rewriter.getIndexAttr(threadSizes[d]);
  }

  LDBG() << "Packing op at " << linalgOp.getLoc();

  rewriter.setInsertionPoint(linalgOp);
  FailureOr<linalg::PackResult> maybeResult =
      linalg::pack(rewriter, linalgOp, packedSizes);
  if (failed(maybeResult)) {
    return linalgOp.emitError(
        "failed to pack reduction dimensions for partial reduction");
  }

  linalg::LinalgOp packedOp = maybeResult->packedLinalgOp;

  // TODO: linalg::pack was probably only written for matmuls and simple
  // elementwise ops. That's not the only class of stuff we handle though.
  // To bypass this for now, this pass expects that padding should be done
  // properly before this pass. This pass is generally placed after
  // GPUApplyPaddingLevel, which is the only reason it works, otherwise it's
  // wrong in general. The other problem with the transformation is it does
  // not account for linalg.index in the body of the op. These commonly come
  // up because general masking of reductions requires this. We fix up the index
  // ops after the fact, but it would be better if linalg::pack just handled
  // this correctly in the first place. The ideal solution is to implement a
  // more general version of linalg::pack that works on IndexingOpInterface and
  // handles things correctly.
  //
  // Fix linalg.index ops in the body. After packing, dimension d is split into
  // [outer=d, inner=numLoops+idx]. The original linalg.index d now refers to
  // the outer dimension. Replace it with: outer * T + inner.
  SmallVector<linalg::IndexOp> indexOps;
  packedOp.getBlock()->walk(
      [&](linalg::IndexOp op) { indexOps.push_back(op); });
  for (linalg::IndexOp indexOp : indexOps) {
    uint64_t dim = indexOp.getDim();
    auto it = llvm::find(dimsToPackVec, dim);
    if (it == dimsToPackVec.end()) {
      continue;
    }
    unsigned idx = std::distance(dimsToPackVec.begin(), it);
    unsigned innerDim = numLoops + idx;
    int64_t tileSize = threadSizes[dim];

    rewriter.setInsertionPointAfter(indexOp);
    Location loc = indexOp.getLoc();
    Value inner = linalg::IndexOp::create(rewriter, loc, innerDim);
    AffineExpr outerExpr, innerExpr;
    bindDims(rewriter.getContext(), outerExpr, innerExpr);
    OpFoldResult combined = affine::makeComposedFoldedAffineApply(
        rewriter, loc, outerExpr * tileSize + innerExpr,
        {getAsOpFoldResult(indexOp.getResult()), getAsOpFoldResult(inner)});
    Value combinedVal =
        getValueOrCreateConstantIndexOp(rewriter, loc, combined);
    rewriter.replaceUsesWithIf(indexOp, combinedVal, [&](OpOperand &use) {
      return use.getOwner() != combinedVal.getDefiningOp();
    });
  }

  // Now build the updated lowering config for the packed op.
  // The packed op has numLoops + dimsToPackVec.size() dimensions.
  // For each packed dim d, it splits into [outer=d, inner=newDim].
  //
  // Tile size splitting:
  //   partial_reduction[d] = P, thread[d] = T
  //   -> outer: partial_reduction = P/T, thread = 1
  //   -> inner: partial_reduction = 0,   thread = T

  SmallVector<int64_t> workgroup = loweringConfig.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Workgroup), linalgOp);

  FailureOr<IREE::GPU::Basis> laneBasisResult =
      IREE::GPU::getBasis(loweringConfig, IREE::GPU::TilingLevel::Thread);
  FailureOr<IREE::GPU::Basis> subgroupBasisResult =
      IREE::GPU::getBasis(loweringConfig, IREE::GPU::TilingLevel::Subgroup);

  unsigned numNewDims = dimsToPackVec.size();
  unsigned newNumLoops = numLoops + numNewDims;

  // Resize existing arrays to numLoops in case they were shorter.
  workgroup.resize(numLoops, 0);
  partialReduction.resize(numLoops, 0);
  threadSizes.resize(numLoops, 0);

  SmallVector<int64_t> newWorkgroup(newNumLoops, 0);
  SmallVector<int64_t> newPartialReduction(newNumLoops, 0);
  SmallVector<int64_t> newThread(newNumLoops, 0);

  IREE::GPU::Basis newLaneBasis;
  newLaneBasis.counts.resize(newNumLoops, 1);
  newLaneBasis.mapping.resize(newNumLoops);

  IREE::GPU::Basis newSubgroupBasis;
  newSubgroupBasis.counts.resize(newNumLoops, 1);
  newSubgroupBasis.mapping.resize(newNumLoops);

  // Copy existing outer dims.
  for (unsigned d = 0; d < numLoops; ++d) {
    newWorkgroup[d] = workgroup[d];
    bool isPacked = llvm::is_contained(dimsToPackVec, d);
    if (isPacked) {
      // Outer part of packed dim: thread count moves to inner dim.
      int64_t T = threadSizes[d];
      assert(partialReduction[d] % T == 0 &&
             "partial reduction size must be divisible by thread tile size");
      newPartialReduction[d] = partialReduction[d] / T;
      newThread[d] = 1;
    } else {
      newPartialReduction[d] = partialReduction[d];
      newThread[d] = threadSizes[d];
    }

    // Basis counts stay on outer dim (distribution happens at outer level).
    if (succeeded(laneBasisResult)) {
      newLaneBasis.counts[d] = laneBasisResult->counts[d];
      newLaneBasis.mapping[d] = laneBasisResult->mapping[d];
    } else {
      newLaneBasis.mapping[d] = d;
    }

    if (succeeded(subgroupBasisResult)) {
      newSubgroupBasis.counts[d] = subgroupBasisResult->counts[d];
      newSubgroupBasis.mapping[d] = subgroupBasisResult->mapping[d];
    } else {
      newSubgroupBasis.mapping[d] = d;
    }
  }

  // Append inner dims (one per packed dim, appended in order).
  for (auto [idx, d] : llvm::enumerate(dimsToPackVec)) {
    unsigned innerDim = numLoops + idx;
    newThread[innerDim] = threadSizes[d];
    newLaneBasis.mapping[innerDim] = innerDim;
    newSubgroupBasis.mapping[innerDim] = innerDim;
  }

  // Build new config, preserving any attributes we don't modify.
  MLIRContext *context = rewriter.getContext();
  Builder b(context);
  NamedAttrList configAttrs(loweringConfig.getAttributes());
  configAttrs.set("workgroup", b.getI64ArrayAttr(newWorkgroup));
  configAttrs.set("partial_reduction", b.getI64ArrayAttr(newPartialReduction));
  configAttrs.set("thread", b.getI64ArrayAttr(newThread));

  // setBasis appends, so erase old entries first to avoid duplicates.
  if (succeeded(laneBasisResult)) {
    configAttrs.erase(b.getStringAttr("lane_basis"));
  }
  if (succeeded(subgroupBasisResult)) {
    configAttrs.erase(b.getStringAttr("subgroup_basis"));
  }

  SmallVector<NamedAttribute> attrsVec(configAttrs.begin(), configAttrs.end());
  if (succeeded(laneBasisResult)) {
    IREE::GPU::setBasis(context, attrsVec, IREE::GPU::TilingLevel::Thread,
                        newLaneBasis);
  }
  if (succeeded(subgroupBasisResult)) {
    IREE::GPU::setBasis(context, attrsVec, IREE::GPU::TilingLevel::Subgroup,
                        newSubgroupBasis);
  }

  auto configDict = b.getDictionaryAttr(attrsVec);
  auto newConfig = IREE::GPU::LoweringConfigAttr::get(context, configDict);
  setLoweringConfig(packedOp, newConfig);

  LDBG() << "After packing: " << *packedOp;

  return success();
}

void GPUPackPartialReductionsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // Collect worklist before mutating IR.
  SmallVector<linalg::LinalgOp> worklist;
  getOperation()->walk([&](linalg::LinalgOp op) {
    if (getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op)) {
      worklist.push_back(op);
    }
  });

  for (linalg::LinalgOp op : worklist) {
    if (failed(packPartialReduction(rewriter, op))) {
      return signalPassFailure();
    }
  }
}

} // namespace
} // namespace mlir::iree_compiler
