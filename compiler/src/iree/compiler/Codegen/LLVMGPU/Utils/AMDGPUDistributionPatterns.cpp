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

// The naming scheme for these operators is:
// InputType_MxNxK_OutputType.
enum class MFMAType {
  F16_16x16x16_F32,
  F16_32x32x8_F32,
};

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
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
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

  Value computeMMA(Value a, Value b, Value c, Location loc, OpBuilder &rewriter,
                   MFMAType mfmaType) const {
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

  PerDimLayoutAttr createPerDimLayout(MLIRContext *ctx,
                                      ArrayRef<LayoutDimension> dims,
                                      ArrayRef<int64_t> shapes) const {
    SmallVector<LayoutDimensionAttr> dimAttrs;
    for (auto dim : dims)
      dimAttrs.push_back(LayoutDimensionAttr::get(ctx, dim));
    return PerDimLayoutAttr::get(ctx, dimAttrs, shapes);
  }

  std::tuple<PerDimLayoutAttr, PerDimLayoutAttr> createCanonicalLayouts16x16x16(
      LayoutDimension batchRowLabel, int64_t batchRow,
      LayoutDimension batchColLabel, int64_t batchCol) const {
    MLIRContext *ctx = getContext();
    PerDimLayoutAttr rowLayout = createPerDimLayout(
        ctx, {batchRowLabel, LayoutDimension::LANEX}, {batchRow, 16});
    PerDimLayoutAttr colLayout = createPerDimLayout(
        ctx, {batchColLabel, LayoutDimension::LANEY, LayoutDimension::VECTORX},
        {batchCol, 4, 4});
    return {rowLayout, colLayout};
  }

  bool isCompatible16x16x16A(LayoutAttr layout, int64_t batchRow,
                             int64_t batchCol) const {
    auto [rowLayout, colLayout] = createCanonicalLayouts16x16x16(
        LayoutDimension::BATCHX, batchRow, LayoutDimension::BATCHY, batchCol);
    LayoutAttr canonicalLayout =
        LayoutAttr::get(getContext(), {rowLayout, colLayout});
    return layout == canonicalLayout;
  }

  bool isCompatible16x16x16B(LayoutAttr layout, int64_t batchRow,
                             int64_t batchCol) const {
    auto [colLayout, rowLayout] = createCanonicalLayouts16x16x16(
        LayoutDimension::BATCHY, batchCol, LayoutDimension::BATCHX, batchRow);
    LayoutAttr canonicalLayout =
        LayoutAttr::get(getContext(), {rowLayout, colLayout});
    return layout == canonicalLayout;
  }

  bool isCompatible16x16x16C(LayoutAttr layout, int64_t batchRow,
                             int64_t batchCol) const {
    return isCompatible16x16x16B(layout, batchRow, batchCol);
  }

  std::tuple<PerDimLayoutAttr, PerDimLayoutAttr>
  createCanonicalLayouts32x32x8(LayoutDimension batchRowLabel, int64_t batchRow,
                                LayoutDimension batchColLabel, int64_t batchCol,
                                ContractMatrixType matrixType) const {
    MLIRContext *ctx = getContext();
    PerDimLayoutAttr rowLayout = createPerDimLayout(
        ctx, {batchRowLabel, LayoutDimension::LANEX}, {batchRow, 32});
    PerDimLayoutAttr colLayout;
    if (matrixType == ContractMatrixType::C) {
      colLayout =
          createPerDimLayout(ctx,
                             {batchColLabel, LayoutDimension::VECTORY,
                              LayoutDimension::LANEY, LayoutDimension::VECTORX},
                             {batchCol, 4, 2, 4});
    } else {
      colLayout = createPerDimLayout(
          ctx,
          {batchColLabel, LayoutDimension::LANEY, LayoutDimension::VECTORX},
          {batchCol, 2, 4});
    }
    return {rowLayout, colLayout};
  }

  bool isCompatible32x32x8A(LayoutAttr layout, int64_t batchRow,
                            int64_t batchCol) const {
    auto [rowLayout, colLayout] = createCanonicalLayouts32x32x8(
        LayoutDimension::BATCHX, batchRow, LayoutDimension::BATCHY, batchCol,
        ContractMatrixType::A);
    LayoutAttr canonicalLayout =
        LayoutAttr::get(getContext(), {rowLayout, colLayout});
    return layout == canonicalLayout;
  }

  bool isCompatible32x32x8B(LayoutAttr layout, int64_t batchRow,
                            int64_t batchCol) const {
    auto [colLayout, rowLayout] = createCanonicalLayouts32x32x8(
        LayoutDimension::BATCHY, batchCol, LayoutDimension::BATCHX, batchRow,
        ContractMatrixType::B);
    LayoutAttr canonicalLayout =
        LayoutAttr::get(getContext(), {rowLayout, colLayout});
    return layout == canonicalLayout;
  }

  bool isCompatible32x32x8C(LayoutAttr layout, int64_t batchRow,
                            int64_t batchCol) const {
    auto [colLayout, rowLayout] = createCanonicalLayouts32x32x8(
        LayoutDimension::BATCHY, batchCol, LayoutDimension::BATCHX, batchRow,
        ContractMatrixType::C);
    LayoutAttr canonicalLayout =
        LayoutAttr::get(getContext(), {rowLayout, colLayout});
    return layout == canonicalLayout;
  }

  bool isCompatible16x16x16(LayoutAttr layout, ContractMatrixType matrixType,
                            int64_t batchRow, int64_t batchCol) const {
    switch (matrixType) {
    case ContractMatrixType::A:
      return isCompatible16x16x16A(layout, batchRow, batchCol);
    case ContractMatrixType::B:
      return isCompatible16x16x16B(layout, batchRow, batchCol);
    default:
      return isCompatible16x16x16C(layout, batchRow, batchCol);
    }
    return false;
  }

  bool isCompatible32x32x8(LayoutAttr layout, ContractMatrixType matrixType,
                           int64_t batchRow, int64_t batchCol) const {
    switch (matrixType) {
    case ContractMatrixType::A:
      return isCompatible32x32x8A(layout, batchRow, batchCol);
    case ContractMatrixType::B:
      return isCompatible32x32x8B(layout, batchRow, batchCol);
    default:
      return isCompatible32x32x8C(layout, batchRow, batchCol);
    }
    return false;
  }

  bool isCompatible(LayoutAttr layout, ContractMatrixType matrixType,
                    MFMAType mfmaType) const {
    std::optional<int64_t> batchRow = layout.getBatchDim(0);
    if (!batchRow)
      return false;
    std::optional<int64_t> batchCol = layout.getBatchDim(1);
    if (!batchCol)
      return false;
    switch (mfmaType) {
    case MFMAType::F16_16x16x16_F32:
      return isCompatible16x16x16(layout, matrixType, batchRow.value(),
                                  batchCol.value());
    case MFMAType::F16_32x32x8_F32:
      return isCompatible32x32x8(layout, matrixType, batchRow.value(),
                                 batchCol.value());
    default:
      return false;
    }
    return false;
  }

  // If we have a prior guess of the MFMA type, only evaluate that type.
  // Otherwise, evaluate all types to find a match.
  std::optional<MFMAType> inferMFMAType(LayoutAttr layout,
                                        ContractMatrixType matrixType,
                                        std::optional<MFMAType> prior) const {
    SmallVector<MFMAType> mfmaTypes;
    if (prior) {
      mfmaTypes.push_back(prior.value());
    } else {
      mfmaTypes = {MFMAType::F16_16x16x16_F32, MFMAType::F16_32x32x8_F32};
    }
    for (MFMAType mfmaType : mfmaTypes) {
      if (isCompatible(layout, matrixType, mfmaType))
        return mfmaType;
    }
    return std::nullopt;
  }

  // Inputs are LHS, RHS and ACC operands and corresponding layouts.
  // Output is inferred MFMAType or none (if layout is not compatible with any
  // MFMA layout).
  std::optional<MFMAType>
  inferCompatibleMFMAType(ArrayRef<LayoutAttr> layouts,
                          ContractType contractType) const {
    std::optional<MFMAType> mfmaType{std::nullopt};
    SmallVector<ContractMatrixType> matrixTypes{
        ContractMatrixType::A, ContractMatrixType::B, ContractMatrixType::C};

    if (isOperandATransposed(contractType)) {
      matrixTypes[0] = ContractMatrixType::B;
    }

    if (isOperandBTransposed(contractType)) {
      matrixTypes[1] = ContractMatrixType::A;
    }

    for (auto [layout, matrixType] : llvm::zip(layouts, matrixTypes)) {
      mfmaType = inferMFMAType(layout, matrixType, mfmaType);
      if (!mfmaType)
        return std::nullopt;
    }
    return mfmaType;
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

    std::optional<MFMAType> mfmaType =
        inferCompatibleMFMAType(layouts, contractType);
    if (!mfmaType)
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
        dMatrix = computeMMA(aMatrix, bMatrix, dMatrix, loc, rewriter,
                             mfmaType.value());
      }
      vector = rewriter.create<vector::InsertOp>(loc, dMatrix, vector, indices);
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
