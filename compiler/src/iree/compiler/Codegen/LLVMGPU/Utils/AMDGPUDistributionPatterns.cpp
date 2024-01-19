// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

enum class ContractMatrixType { A, B, C, D };
enum class ContractType { MM, MMT, MTM, MTMT, UNSUPPORTED };

namespace {

struct DistributeContractions final
    : OpDistributionPattern<vector::ContractionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeContractions(MLIRContext *ctx, MFMAType mfmaType)
      : OpDistributionPattern(ctx), mfmaType(mfmaType) {}

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
    ;
  }

  int64_t getReductionDimensionShape(int64_t rowBatch, int64_t colBatch,
                                     ContractType contractType) const {
    if ((contractType == ContractType::MTM) ||
        (contractType == ContractType::MTMT)) {
      return rowBatch;
    }
    return colBatch;
  }

  ContractType inferContractType(MLIRContext *ctx,
                                 SmallVector<AffineMap> maps) const {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    if ((maps == infer({{m, k}, {k, n}, {m, n}})) ||
        (maps == infer({{n, k}, {k, m}, {n, m}}))) {
      return ContractType::MM;
    } else if ((maps == infer({{m, k}, {n, k}, {m, n}})) ||
               (maps == infer({{n, k}, {m, k}, {n, m}}))) {
      return ContractType::MMT;
    } else if ((maps == infer({{k, m}, {k, n}, {m, n}})) ||
               (maps == infer({{k, n}, {k, m}, {n, m}}))) {
      return ContractType::MTM;
    } else if ((maps == infer({{k, m}, {n, k}, {m, n}})) ||
               (maps == infer({{k, n}, {m, k}, {n, m}}))) {
      return ContractType::MTMT;
    }
    return ContractType::UNSUPPORTED;
  }

  Value computeMMA(Value a, Value b, Value c, Location loc,
                   OpBuilder &rewriter) const {
    uint32_t m, n, k, blks;
    if (mfmaType == MFMAType::F16_16x16x16_F32) {
      m = n = k = 16;
    } else if (mfmaType == MFMAType::F16_32x32x8_F32) {
      m = n = 32;
      k = 8;
    }
    blks = 1;
    return rewriter.create<amdgpu::MFMAOp>(loc, c.getType(), m, n, k, blks, a,
                                           b, c);
  }

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    constexpr int LHS = 0;
    constexpr int RHS = 1;
    constexpr int ACC = 2;
    SmallVector<VectorValue> operands;
    SmallVector<LayoutAttr> layouts;
    for (auto [operand, layout] :
         llvm::zip(contractOp->getOperands(), signature.operands)) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        if (auto vectorLayout = dyn_cast<LayoutAttr>(layout)) {
          operands.push_back(vectorOperand);
          layouts.push_back(vectorLayout);
        }
      }
    }
    LayoutAttr resultLayout = dyn_cast<LayoutAttr>(signature.results[0]);
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
        dMatrix = computeMMA(aMatrix, bMatrix, dMatrix, loc, rewriter);
      }
      vector = rewriter.create<vector::InsertOp>(loc, dMatrix, vector, indices);
    };

    LayoutIterator iterator(resultLayout);
    LayoutIterator batchIterator = iterator.getBatchIterator();
    batchIterator.apply(contractFn);
    replaceOpWithDistributedValues(rewriter, contractOp, vector);
    return success();
  }

  MFMAType mfmaType;
};
} // namespace

void populateAMDGPUDistributionPatterns(RewritePatternSet &patterns,
                                        MFMAType mfmaType) {
  patterns.add<DistributeContractions>(patterns.getContext(), mfmaType);
}

} // namespace mlir::iree_compiler
