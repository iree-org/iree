// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTWORKGROUPFORALLTOPCFPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ConvertWorkgroupForall : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override;
};

struct ConvertWorkgroupForallToPCFPass final
    : public impl::ConvertWorkgroupForallToPCFPassBase<
          ConvertWorkgroupForallToPCFPass> {
  void runOnOperation() override;
  using Base::Base;
};

} // namespace

// DO NOT SUBMIT: Write tests.
LogicalResult
ConvertWorkgroupForall::matchAndRewrite(scf::ForallOp op,
                                        PatternRewriter &rewriter) const {
  std::optional<ArrayAttr> maybeMapping = op.getMappingAttr();
  if (!maybeMapping || maybeMapping.value().empty() ||
      !llvm::all_of(maybeMapping.value(),
                    llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>)) {
    return failure();
  }
  // Interface is implemented via external models hence the cast.
  auto scope = cast<IREE::PCF::ScopeAttr>(
      IREE::Codegen::WorkgroupAttr::get(rewriter.getContext()));
  return convertForallToPCF(rewriter, op, scope, 3);
}

void ConvertWorkgroupForallToPCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertWorkgroupForall>(&getContext());
  walkAndApplyPatterns(getOperation(), std::move(patterns));
}

} // namespace mlir::iree_compiler
