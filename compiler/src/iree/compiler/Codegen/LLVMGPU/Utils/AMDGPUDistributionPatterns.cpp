// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

namespace {

struct DistributeContractions final
    : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = dyn_cast<VectorValue>(contractOp.getResult());
    if (!result) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "result should be of type vector");
    }

    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(signature[result]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          contractOp, "result layout should be of type LayoutAttr");
    }

    auto mmaAttr =
        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
    if (!mmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing iree.amdgpu.mma intrinsic attribute");
    }

    constexpr int LHS = 0;
    constexpr int RHS = 1;
    constexpr int ACC = 2;
    SmallVector<VectorValue> operands;
    SmallVector<LayoutAttr> layouts;
    for (Value operand : contractOp->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        auto layout = signature[vectorOperand];
        if (auto vectorLayout = dyn_cast<LayoutAttr>(layout)) {
          operands.push_back(vectorOperand);
          layouts.push_back(vectorLayout);
        }
      }
    }

    Type elementType =
        llvm::cast<ShapedType>(operands[ACC].getType()).getElementType();
    SmallVector<int64_t> vectorShape = resultLayout.getDistributedShape();
    auto vectorType = VectorType::get(vectorShape, elementType);
    Location loc = contractOp.getLoc();
    Value vector = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));

    VectorContractOpInfo opInfo(contractOp);
    auto [lhsK, rhsK] = opInfo.getOperandKIndex();

    std::optional<int64_t> kBatch = layouts[LHS].getBatchDim(lhsK);
    if (!kBatch) {
      return failure();
    }

    auto contractFn = [&](const LayoutIterator::State &state) {
      auto [lhsM, rhsN] = opInfo.getOperandMNIndex();
      auto [lhsK, rhsK] = opInfo.getOperandKIndex();
      SmallVector<int64_t> indices = state.computeIteratorProjectedSIMTIndex();
      Value dMatrix = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, operands[ACC], layouts[ACC]), indices);
      for (int k = 0; k < kBatch; ++k) {
        SmallVector<int64_t> lhsIndices(2);
        SmallVector<int64_t> rhsIndices(2);
        lhsIndices[lhsM] = indices[0];
        lhsIndices[lhsK] = k;
        rhsIndices[rhsN] = indices[1];
        rhsIndices[rhsK] = k;

        Value aMatrix = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, operands[LHS], layouts[LHS]),
            lhsIndices);

        Value bMatrix = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, operands[RHS], layouts[RHS]),
            rhsIndices);

        dMatrix = mmaAttr
                      .buildMmaOperation(rewriter, loc, dMatrix.getType(),
                                         aMatrix, bMatrix, dMatrix)
                      .value();
      }
      vector = rewriter.create<vector::InsertOp>(loc, dMatrix, vector, indices);
      return success();
    };

    LayoutIterator iterator(resultLayout);
    LayoutIterator batchIterator = iterator.getBatchIterator();
    batchIterator.apply(contractFn);
    replaceOpWithDistributedValues(rewriter, contractOp, vector);
    return success();
  }
};
} // namespace

void populateAMDGPUDistributionPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributeContractions>(patterns.getContext());
}

} // namespace mlir::iree_compiler
