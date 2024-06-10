// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

enum class ContractMatrixType { A, B, C, D };
enum class ContractType { MM, MMT, MTM, MTMT, UNSUPPORTED };

namespace {

static bool isOperandATransposed(ContractType contractType) {
  return (contractType == ContractType::MTM) ||
         (contractType == ContractType::MTMT);
}

static bool isOperandBTransposed(ContractType contractType) {
  return (contractType == ContractType::MMT) ||
         (contractType == ContractType::MTMT);
}
struct DistributeContractions final
    : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  // For a MM contraction, we compute C(i, k) += A(i, j) * B(j, k).
  // If we have an MMT contraction, we compute C(i, k) += A(i, j) * B(k, j).
  // This function returns the appropriate indices for the A and B matrices.
  // Given incoming indices (i, j), it either returns the same or swaps them,
  // depending on the type of contraction and type of matrix.
  SmallVector<int64_t> getIndices(ContractType contractType,
                                  ContractMatrixType matrixType, int i,
                                  int j) const {
    SmallVector<int64_t> originalIndices{i, j};
    SmallVector<int64_t> swappedIndices{j, i};
    if (contractType == ContractType::MTMT)
      return swappedIndices;
    if ((contractType == ContractType::MTM) &&
        (matrixType == ContractMatrixType::A))
      return swappedIndices;
    if ((contractType == ContractType::MMT) &&
        (matrixType == ContractMatrixType::B))
      return swappedIndices;
    return originalIndices;
  }

  int64_t getReductionDimensionShape(int64_t rowBatch, int64_t colBatch,
                                     ContractType contractType) const {
    if (isOperandATransposed(contractType)) {
      return rowBatch;
    }
    return colBatch;
  }

  ContractType inferContractType(MLIRContext *ctx,
                                 SmallVector<AffineMap> maps) const {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, ctx);
    };
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    if ((maps == infer({{m, k}, {k, n}, {m, n}})) ||
        (maps == infer({{n, k}, {k, m}, {n, m}}))) {
      return ContractType::MM;
    }
    if ((maps == infer({{m, k}, {n, k}, {m, n}})) ||
        (maps == infer({{n, k}, {m, k}, {n, m}}))) {
      return ContractType::MMT;
    }
    if ((maps == infer({{k, m}, {k, n}, {m, n}})) ||
        (maps == infer({{k, n}, {k, m}, {n, m}}))) {
      return ContractType::MTM;
    }
    if ((maps == infer({{k, m}, {n, k}, {m, n}})) ||
        (maps == infer({{k, n}, {m, k}, {n, m}}))) {
      return ContractType::MTMT;
    }
    return ContractType::UNSUPPORTED;
  }

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue result = dyn_cast<VectorValue>(contractOp.getResult());
    if (!result) {
      return failure();
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

    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(signature[result]);
    if (!resultLayout)
      return failure();

    Type elementType =
        llvm::cast<ShapedType>(operands[ACC].getType()).getElementType();
    SmallVector<int64_t> vectorShape = resultLayout.getDistributedShape();
    auto vectorType = VectorType::get(vectorShape, elementType);
    Location loc = contractOp.getLoc();
    Value vector = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));

    ContractType contractType = inferContractType(
        contractOp.getContext(), contractOp.getIndexingMapsArray());
    if (contractType == ContractType::UNSUPPORTED)
      return failure();

    auto mmaAttr =
        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
    if (!mmaAttr) {
      return rewriter.notifyMatchFailure(
          contractOp, "missing iree.amdgpu.mma intrinsic attribute");
    }

    std::optional<int64_t> rowBatch = layouts[LHS].getBatchDim(0);
    if (!rowBatch)
      return failure();
    std::optional<int64_t> colBatch = layouts[LHS].getBatchDim(1);
    if (!colBatch)
      return failure();

    int K = getReductionDimensionShape(rowBatch.value(), colBatch.value(),
                                       contractType);

    auto contractFn = [&](const LayoutIterator::State &state) {
      SmallVector<int64_t> indices = state.computeIteratorProjectedSIMTIndex();
      Value dMatrix = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, operands[ACC], layouts[ACC]), indices);
      for (int k = 0; k < K; k++) {
        Value aMatrix = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, operands[LHS], layouts[LHS]),
            getIndices(contractType, ContractMatrixType::A, indices[0], k));
        Value bMatrix = rewriter.create<vector::ExtractOp>(
            loc, getDistributed(rewriter, operands[RHS], layouts[RHS]),
            getIndices(contractType, ContractMatrixType::B, k, indices[1]));
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
