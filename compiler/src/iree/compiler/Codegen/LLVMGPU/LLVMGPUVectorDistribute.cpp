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

namespace {

class ContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  ContractionVectorLayoutOptions(Operation *root, Value laneId,
                                 int64_t subgroupSize)
      : VectorLayoutOptions(root), patterns(root->getContext()) {
    populateGPUDistributionPatterns(patterns);
    populateGPUDistributionLayoutAttrPatterns(laneId, patterns);
    populateGPUDistributeNestedLayoutAttrPatterns(patterns, laneId,
                                                  subgroupSize);
    populateGPUDistributeNestedLayoutContractAMDGPUPatterns(patterns);
  }

  RewritePatternSet &getPatterns() { return patterns; }

private:
  RewritePatternSet patterns;
};

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

    AffineExpr x, y, z;
    bindSymbols(func.getContext(), x, y, z);
    // Construct the expression for linearizing the thread indices.
    AffineExpr linearId =
        x + workgroupSize[0] * y + workgroupSize[1] * workgroupSize[0] * z;

    IRRewriter rewriter(func);
    rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
    SmallVector<OpFoldResult> threadGrid = {
        rewriter.createOrFold<gpu::ThreadIdOp>(func.getLoc(),
                                               gpu::Dimension::x),
        rewriter.createOrFold<gpu::ThreadIdOp>(func.getLoc(),
                                               gpu::Dimension::y),
        rewriter.createOrFold<gpu::ThreadIdOp>(func.getLoc(),
                                               gpu::Dimension::z)};

    Value linearThreadIdVal = affine::makeComposedAffineApply(
        rewriter, func.getLoc(), linearId, threadGrid);

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
