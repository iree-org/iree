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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-distribute"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORDISTRIBUTEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

ContractionVectorLayoutOptions::ContractionVectorLayoutOptions(
    Operation *root, Value laneId, int64_t subgroupSize)
    : VectorLayoutOptions(root), patterns(root->getContext()) {
  populateGPUDistributionPatterns(patterns);
  populateGPUDistributeNestedLayoutAttrPatterns(patterns, laneId, subgroupSize);
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
struct LLVMGPUVectorDistributePass final
    : impl::LLVMGPUVectorDistributePassBase<LLVMGPUVectorDistributePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    std::array<int64_t, 3> workgroupSize;
    if (func->hasAttr("workgroup_size")) {
      auto tmpSizes =
          llvm::cast<ArrayAttr>(func->getAttr("workgroup_size")).getValue();
      for (auto [i, size] : llvm::enumerate(tmpSizes)) {
        workgroupSize[i] = llvm::cast<IntegerAttr>(size).getInt();
      }
    } else {
      std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
          getWorkgroupSize(func);
      if (!maybeWorkgroupSize) {
        func->emitOpError()
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

    IRRewriter rewriter(func);
    rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
    SmallVector<Value> threadGrid = {rewriter.createOrFold<gpu::ThreadIdOp>(
                                         func.getLoc(), gpu::Dimension::z),
                                     rewriter.createOrFold<gpu::ThreadIdOp>(
                                         func.getLoc(), gpu::Dimension::y),
                                     rewriter.createOrFold<gpu::ThreadIdOp>(
                                         func.getLoc(), gpu::Dimension::x)};
    std::reverse(workgroupSize.begin(), workgroupSize.end());

    Value linearThreadIdVal = rewriter.create<affine::AffineLinearizeIndexOp>(
        func.getLoc(), threadGrid, workgroupSize, /*disjoint=*/true);

    std::optional<int64_t> subgroupSize = getSubgroupSize(func);
    if (!subgroupSize) {
      func->emitOpError()
          << "unable to query subgroup size information from entry point";
      return signalPassFailure();
    }

    ContractionVectorLayoutOptions options(func, linearThreadIdVal,
                                           subgroupSize.value());

    if (failed(distributeVectorOps(func, options.getPatterns(), options))) {
      func->emitOpError() << "failed to distribute";
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
