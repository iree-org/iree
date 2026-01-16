// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTFORALLTOGENERICNESTWORKGROUPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Returns true if the forall op has WorkgroupMappingAttr mapping attributes.
static bool hasWorkgroupMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping->empty()) {
    return false;
  }
  return llvm::all_of(mapping.value(),
                      llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>);
}

struct ConvertForallToGenericNestWorkgroupPass final
    : public impl::ConvertForallToGenericNestWorkgroupPassBase<
          ConvertForallToGenericNestWorkgroupPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Always use linearized workgroup scope (1 id).
    // Interface is implemented via external models hence the cast.
    auto scope = cast<IREE::PCF::ScopeAttrInterface>(
        IREE::Codegen::WorkgroupScopeAttr::get(ctx, /*linearize=*/true));

    SmallVector<IREE::PCF::ScopeAttrInterface> scopes = {scope};

    IRRewriter rewriter(ctx);
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Only convert foralls with workgroup mapping attributes.
      if (hasWorkgroupMapping(forallOp)) {
        forallOps.push_back(forallOp);
      }
    });

    for (scf::ForallOp forallOp : forallOps) {
      rewriter.setInsertionPoint(forallOp);
      FailureOr<IREE::PCF::GenericOp> result =
          IREE::PCF::convertForallToGenericNest(rewriter, forallOp, scopes);
      if (failed(result)) {
        forallOp.emitError("failed to convert forall to generic nest");
        return signalPassFailure();
      }
      // Replace forall results with generic results.
      rewriter.replaceOp(forallOp, result->getResults());
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
