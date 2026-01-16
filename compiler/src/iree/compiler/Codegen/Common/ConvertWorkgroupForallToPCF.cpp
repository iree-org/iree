// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTWORKGROUPFORALLTOPCFPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ConvertWorkgroupForall : OpRewritePattern<scf::ForallOp> {
  using Base::Base;
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

LogicalResult
ConvertWorkgroupForall::matchAndRewrite(scf::ForallOp op,
                                        PatternRewriter &rewriter) const {
  ArrayAttr mappingAttr = op.getMappingAttr();
  if (!mappingAttr || mappingAttr.empty() ||
      !llvm::all_of(mappingAttr,
                    llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>)) {
    return failure();
  }
  // Linearize all ids down to 1 so that in cases when there are multiple
  // scf.foralls with incompatible delinearization bases. This technically
  // may be a small pessimization in very specific static cases, so if someone
  // ever finds they care they can try doing the analysis here to figure out
  // when it's ok not to linearize.
  //
  // Interface is implemented via external models hence the cast.
  auto scope = cast<IREE::PCF::ScopeAttrInterface>(
      IREE::Codegen::WorkgroupScopeAttr::get(rewriter.getContext(),
                                             /*linearize=*/true));
  FailureOr<IREE::PCF::LoopOp> res =
      convertForallToPCFLoop(rewriter, op, scope, 1);
  if (failed(res)) {
    return failure();
  }

  // Create a workgroup count hint to launch all workgroups along x.
  auto counts = llvm::to_vector_of<OpFoldResult>(res->getCount());
  rewriter.setInsertionPoint(*res);
  [[maybe_unused]] LogicalResult hintRes = createWorkgroupCountHint(
      rewriter, res->getLoc(), counts, /*maxWorkgroupParallelDims=*/1,
      /*reverse=*/false);
  assert(succeeded(hintRes) &&
         "Unexpected failure to construct workgroup count hint");
  return success();
}

void ConvertWorkgroupForallToPCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertWorkgroupForall>(&getContext());
  walkAndApplyPatterns(getOperation(), std::move(patterns));
}

} // namespace mlir::iree_compiler
