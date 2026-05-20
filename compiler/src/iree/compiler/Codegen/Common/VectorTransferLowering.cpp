// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vector-transfer-lowering"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VECTORTRANSFERLOWERINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
class VectorTransferLoweringPass
    : public impl::VectorTransferLoweringPassBase<VectorTransferLoweringPass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void VectorTransferLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  // First, flatten the inner contiguous dims of multi-dim transfers so that
  // contiguous trailing chunks (e.g. a packed `<16x2xbf16>` RHS tile whose
  // 32 elements sit in 32 contiguous bytes) become a single 1-D transfer
  // and lower to one wide vector load. If we skipped this and went straight
  // to the rank-reduction unrolling below, the multi-dim transfer would
  // unroll into one rank-1 load per outer iteration (e.g. 16 separate
  // `<2 x bfloat>` loads), which then have to be reassembled into the wide
  // vector via a chain of `shufflevector`s/`vpermt2` in the inner loop.
  {
    RewritePatternSet patterns(ctx);
    vector::populateFlattenVectorTransferPatterns(patterns);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  RewritePatternSet patterns(ctx);
  // Explicitly materialize the mask on transfer_read/transfer_write.
  // Assume we don't have 4 GB vectors.
  vector::populateVectorMaskMaterializationPatterns(
      patterns, /*force32BitVectorIndices=*/true);
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);
  auto vectorTransferToSCFOptions =
      VectorTransferToSCFOptions().enableFullUnroll();
  if (enableScalableLowerings) {
    vectorTransferToSCFOptions.enableLowerScalable();
  }

  populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}
} // namespace
} // namespace mlir::iree_compiler
