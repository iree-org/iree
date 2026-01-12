// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-distribute"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORDISTRIBUTEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

ContractionVectorLayoutOptions::ContractionVectorLayoutOptions(
    Operation *root, Value laneId, int64_t subgroupSize,
    ArrayRef<int64_t> workgroupSize)
    : VectorLayoutOptions(root), patterns(root->getContext()) {
  populateGPUDistributionPatterns(patterns);
  populateGPUDistributeNestedLayoutAttrPatterns(patterns, laneId, subgroupSize,
                                                workgroupSize);
  populateGPUDistributeNestedLayoutContractAMDGPUPatterns(patterns);
}

RewritePatternSet &ContractionVectorLayoutOptions::getPatterns() {
  return patterns;
}

VectorLayoutInterface
ContractionVectorLayoutOptions::getDefaultLayout(VectorType type) const {
  // We only allow a default layout for 0-d vectors for now.
  if (type.getRank() > 0) {
    return VectorLayoutInterface();
  }
  ArrayRef<int64_t> empty = {};
  return IREE::VectorExt::NestedLayoutAttr::get(
      type.getContext(), empty, empty, empty, empty, empty, empty, empty);
}

namespace {

struct RemoveUnitDimStrechingBroadcast final
    : OpRewritePattern<vector::BroadcastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    SetVector<int64_t> strechedDims = broadcastOp.computeBroadcastedUnitDims();
    if (strechedDims.empty()) {
      return failure();
    }
    VectorType srcTy = cast<VectorType>(broadcastOp.getSource().getType());
    VectorType dstTy = broadcastOp.getResultVectorType();
    int64_t numLeadingBroadcastDims = dstTy.getRank() - srcTy.getRank();
    SmallVector<int64_t> broadcastedUnitDims;
    for (auto i : llvm::seq<int64_t>(numLeadingBroadcastDims)) {
      if (dstTy.getShape()[i] == 1) {
        broadcastedUnitDims.push_back(i);
      }
    }
    if (strechedDims.size() > broadcastedUnitDims.size()) {
      return rewriter.notifyMatchFailure(
          broadcastOp,
          "number of streched dims is greater than broadcasted unit dims");
    }
    // Build a permutation that swaps the broadcasted unit dims with a leading
    // unit dim.
    // Note that the way this permutation is built, makes it involutory, i.e.,
    // P = P^{-1}.
    auto perm = llvm::to_vector(llvm::seq<int64_t>(dstTy.getRank()));
    // We use zip instead of zip_equal because strechedDims.size() <=
    // broadcastedUnitDims.size().
    for (auto [strechedDim, broadcastedUnitDim] :
         llvm::zip(strechedDims.getArrayRef(), broadcastedUnitDims)) {
      std::swap(perm[strechedDim], perm[broadcastedUnitDim]);
    }
    VectorType permutedBroadcastTy = VectorType::get(
        applyPermutation(dstTy.getShape(), perm), dstTy.getElementType());
    Value permutedBroadcast = vector::BroadcastOp::create(
        rewriter, broadcastOp.getLoc(), permutedBroadcastTy,
        broadcastOp.getSource());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(broadcastOp,
                                                     permutedBroadcast, perm);
    return success();
  }
};

static LogicalResult
preVectorDistributionNormalizations(mlir::FunctionOpInterface funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  patterns.add<RemoveUnitDimStrechingBroadcast>(funcOp.getContext());
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

struct LLVMGPUVectorDistributePass final
    : impl::LLVMGPUVectorDistributePassBase<LLVMGPUVectorDistributePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Pre-distribution normalizations\n");
    if (failed(preVectorDistributionNormalizations(funcOp))) {
      funcOp->emitOpError()
          << "failed to apply pre-distribution normalization patterns";
      return signalPassFailure();
    }
    std::array<int64_t, 3> workgroupSize;
    if (funcOp->hasAttr("workgroup_size")) {
      auto tmpSizes =
          cast<ArrayAttr>(funcOp->getAttr("workgroup_size")).getValue();
      for (auto [i, size] : llvm::enumerate(tmpSizes)) {
        workgroupSize[i] = cast<IntegerAttr>(size).getInt();
      }
    } else {
      std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
          getWorkgroupSize(funcOp);
      if (!maybeWorkgroupSize) {
        funcOp->emitOpError()
            << "unable to query workgroup_size information from entry point";
        return signalPassFailure();
      }
      for (auto [index, value] : llvm::enumerate(maybeWorkgroupSize.value())) {
        workgroupSize[index] = value;
      }
      for (auto index : llvm::seq<size_t>(maybeWorkgroupSize->size(), 3)) {
        workgroupSize[index] = 1;
      }
    }

    IRRewriter rewriter(funcOp);
    rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    SmallVector<Value> threadGrid = {rewriter.createOrFold<gpu::ThreadIdOp>(
                                         funcOp.getLoc(), gpu::Dimension::z),
                                     rewriter.createOrFold<gpu::ThreadIdOp>(
                                         funcOp.getLoc(), gpu::Dimension::y),
                                     rewriter.createOrFold<gpu::ThreadIdOp>(
                                         funcOp.getLoc(), gpu::Dimension::x)};
    std::reverse(workgroupSize.begin(), workgroupSize.end());

    Value linearThreadIdVal = affine::AffineLinearizeIndexOp::create(
        rewriter, funcOp.getLoc(), threadGrid, workgroupSize,
        /*disjoint=*/true);

    std::optional<int64_t> subgroupSize = getSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp->emitOpError()
          << "unable to query subgroup size information from entry point";
      return signalPassFailure();
    }

    ContractionVectorLayoutOptions options(funcOp, linearThreadIdVal,
                                           subgroupSize.value(), workgroupSize);

    if (failed(distributeVectorOps(funcOp, options.getPatterns(), options))) {
      funcOp->emitOpError() << "failed to distribute";
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
