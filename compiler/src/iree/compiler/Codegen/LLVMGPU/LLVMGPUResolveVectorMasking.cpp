// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPURESOLVEVECTORMASKINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// Creates a mask for an operand by projecting the iterator-space mask to the
// operand's space using the indexing map.
static FailureOr<Value> projectMaskForOperand(ImplicitLocOpBuilder &builder,
                                              vector::CreateMaskOp createMask,
                                              VectorType operandType,
                                              AffineMap indexingMap) {
  // The mask covers the full iterator domain. For each operand dimension,
  // extract the corresponding bound from the create_mask operation.
  SmallVector<Value> operandBounds;
  for (int64_t i = 0; i < indexingMap.getNumResults(); ++i) {
    int64_t iterDimPos = indexingMap.getDimPosition(i);
    operandBounds.push_back(createMask.getOperand(iterDimPos));
  }

  // Create new mask for this operand
  auto maskType = VectorType::get(operandType.getShape(), builder.getI1Type());
  auto maskOp = vector::CreateMaskOp::create(builder, maskType, operandBounds);
  return maskOp.getResult();
}

// Unwraps a masked vector.contract by padding operands with identity values
// and using arith.select to mask them.
struct UnwrapMaskedContractPattern final : OpRewritePattern<vector::MaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskOp maskOp,
                                PatternRewriter &rewriter) const override {
    // Match vector.mask { vector.contract }
    auto contractOp = dyn_cast<vector::ContractionOp>(maskOp.getMaskableOp());
    if (!contractOp) {
      return failure();
    }

    // Reject masks with passthrough values
    if (maskOp.getPassthru()) {
      return rewriter.notifyMatchFailure(maskOp, "passthru not supported");
    }

    Value mask = maskOp.getMask();
    auto maskType = dyn_cast<VectorType>(mask.getType());
    if (!maskType) {
      return failure();
    }

    // For now, only handle masks coming from vector.create_mask
    auto createMask = mask.getDefiningOp<vector::CreateMaskOp>();
    if (!createMask) {
      return rewriter.notifyMatchFailure(maskOp,
                                         "mask is not from vector.create_mask");
    }

    ImplicitLocOpBuilder builder(maskOp.getLoc(), rewriter);

    // Extract combining kind from contract
    vector::CombiningKind kind = contractOp.getKind();

    // Get operand types
    auto lhsType = contractOp.getLhsType();
    auto rhsType = contractOp.getRhsType();

    // Get indexing maps
    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    if (indexingMaps.size() != 3) {
      return failure();
    }

    // Generate identity values for LHS and RHS
    Value identityLhs =
        getCombiningIdentityValue(builder.getLoc(), builder, kind, lhsType);
    Value identityRhs =
        getCombiningIdentityValue(builder.getLoc(), builder, kind, rhsType);

    AffineMap lhsMap = indexingMaps[0];
    AffineMap rhsMap = indexingMaps[1];

    auto lhsMask = projectMaskForOperand(builder, createMask, lhsType, lhsMap);
    if (failed(lhsMask)) {
      return failure();
    }

    auto rhsMask = projectMaskForOperand(builder, createMask, rhsType, rhsMap);
    if (failed(rhsMask)) {
      return failure();
    }

    // Create masked operands using arith.select
    Value lhsMasked = arith::SelectOp::create(builder, *lhsMask,
                                              contractOp.getLhs(), identityLhs);
    Value rhsMasked = arith::SelectOp::create(builder, *rhsMask,
                                              contractOp.getRhs(), identityRhs);

    // Create unmasked contract with masked operands
    auto newContract = vector::ContractionOp::create(
        builder, lhsMasked, rhsMasked, contractOp.getAcc(),
        contractOp.getIndexingMaps(), contractOp.getIteratorTypes(), kind);
    // Preserve all discardable attributes (especially iree.gpu.mma)
    newContract->setDiscardableAttrs(
        contractOp->getDiscardableAttrDictionary());

    rewriter.replaceOp(maskOp, newContract);
    return success();
  }
};

struct LLVMGPUResolveVectorMaskingPass final
    : impl::LLVMGPUResolveVectorMaskingPassBase<
          LLVMGPUResolveVectorMaskingPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UnwrapMaskedContractPattern>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler
