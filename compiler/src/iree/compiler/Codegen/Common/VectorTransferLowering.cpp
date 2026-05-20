// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/IR/DataLayout.h"
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

  // Flatten contiguous trailing dims of multi-dim transfers when the trailing
  // dim is narrower than the target's natural word (the pointer size), so a
  // packed `<16x2xbf16>` (32-bit innermost) lowers to one wide load instead
  // of 16 narrow loads the rank reduction below would reassemble with a
  // chain of shuffles. Sub-word loads in bulk are uniformly pathological;
  // word-and-up loads (`<2xf32>` ... `<16xf32>`) are already fine and
  // flattening *them* fuses register-sized rows into an oversized 1-D
  // transfer + a `vector.shape_cast` re-split (extracts), regressing whole-
  // model .vmfb size for no benefit. This is *not* `native_vector_size`:
  // that is the *widest* useful vector, not the smallest non-pathological
  // load.
  unsigned pointerBits = 64;
  if (auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp)) {
    if (auto attr =
            targetAttr.getConfiguration().getAs<StringAttr>("data_layout")) {
      if (!attr.getValue().empty()) {
        pointerBits = llvm::DataLayout(attr.getValue()).getPointerSizeInBits();
      }
    }
  }
  {
    RewritePatternSet patterns(ctx);
    vector::populateFlattenVectorTransferPatterns(patterns, pointerBits);
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
