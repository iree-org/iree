// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fold-memref-alias-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_IREEFOLDMEMREFALIASOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

struct IREEFoldMemRefAliasOpsPass final
    : impl::IREEFoldMemRefAliasOpsPassBase<IREEFoldMemRefAliasOpsPass> {
  using IREEFoldMemRefAliasOpsPassBase::IREEFoldMemRefAliasOpsPassBase;

  void runOnOperation() override;
};

void IREEFoldMemRefAliasOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  memref::populateFoldMemRefAliasOpPatterns(patterns);
  IREE::VectorExt::populateVectorExtFoldMemRefAliasOpPatterns(patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler
