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

#define GEN_PASS_DEF_TESTCONVERTFORALLTOGENERICNESTWORKGROUPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Returns true if the forall op has LocalMappingAttr mapping attributes,
/// or the mapping is empty/not present.
static bool hasEmptyOrLocalMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping->empty()) {
    return true;
  }
  return llvm::all_of(mapping.value(),
                      llvm::IsaPred<IREE::Codegen::LocalMappingAttr>);
}

struct TestConvertForallToGenericNestWorkgroupPass final
    : public impl::TestConvertForallToGenericNestWorkgroupPassBase<
          TestConvertForallToGenericNestWorkgroupPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    // Create workgroup scope with the specified linearization setting.
    // Interface is implemented via external models hence the cast.
    auto scope = cast<IREE::PCF::ScopeAttrInterface>(
        IREE::Codegen::WorkgroupScopeAttr::get(ctx, linearize));

    SmallVector<IREE::PCF::ScopeAttrInterface> scopes = {scope};

    IRRewriter rewriter(ctx);
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Only convert foralls with empty mapping or local_mapping attributes.
      if (hasEmptyOrLocalMapping(forallOp)) {
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
