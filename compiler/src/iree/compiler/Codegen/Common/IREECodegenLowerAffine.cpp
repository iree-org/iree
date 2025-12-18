// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-lower-affine"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_IREECODEGENLOWERAFFINEPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct IREECodegenLowerAffinePass final
    : impl::IREECodegenLowerAffinePassBase<IREECodegenLowerAffinePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Normal);

    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateAffineToVectorConversionPatterns(patterns);
    affine::populateAffineExpandIndexOpsPatterns(patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                           scf::SCFDialect, vector::VectorDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
