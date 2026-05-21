// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_CONVERTFORALLTOGENERICNESTGPUPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {

/// Returns true if the forall op has gpu::GPUThreadMappingAttr mapping
/// attributes.
static bool hasThreadMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping->empty()) {
    return false;
  }
  return llvm::all_of(mapping.value(),
                      llvm::IsaPred<gpu::GPUThreadMappingAttr>);
}

/// Runs the conversion for forall ops with gpu.thread mapping to nested
/// pcf.generic ops.
static LogicalResult runConversion(Operation *op) {
  MLIRContext *ctx = op->getContext();

  // Create nested scopes: subgroup (outer) + lane (inner).
  // Interface is implemented via external models hence the casts.
  auto subgroupScope =
      cast<PCF::ScopeAttrInterface>(SubgroupScopeAttr::get(ctx));
  auto laneScope = cast<PCF::ScopeAttrInterface>(LaneScopeAttr::get(ctx));

  SmallVector<PCF::ScopeAttrInterface> scopes = {subgroupScope, laneScope};

  IRRewriter rewriter(ctx);
  SmallVector<scf::ForallOp> forallOps;
  op->walk([&](scf::ForallOp forallOp) {
    // Only convert foralls with gpu.thread mapping attributes.
    if (hasThreadMapping(forallOp)) {
      forallOps.push_back(forallOp);
    }
  });

  for (scf::ForallOp forallOp : forallOps) {
    rewriter.setInsertionPoint(forallOp);
    FailureOr<PCF::GenericOp> result =
        PCF::convertForallToGenericNest(rewriter, forallOp, scopes);
    if (failed(result)) {
      forallOp.emitError("failed to convert forall to generic nest");
      return failure();
    }
    // Replace forall results with generic results.
    rewriter.replaceOp(forallOp, result->getResults());
  }
  return success();
}

struct ConvertForallToGenericNestGPUPass final
    : public impl::ConvertForallToGenericNestGPUPassBase<
          ConvertForallToGenericNestGPUPass> {
  using Base::Base;

  void runOnOperation() override {
    if (failed(runConversion(getOperation()))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::GPU
