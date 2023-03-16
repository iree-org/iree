// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-llvmgpu-layout-analysis-and-distribution"

namespace mlir::iree_compiler {

namespace {

static constexpr int maxTensorDims = 2;
namespace DimType {
static constexpr int Batch0 = 0;  // Batch dimension for tensor dim 0
static constexpr int Batch1 = 1;  // Batch dimension for tensor dim 1
static constexpr int LaneIdZ = 2;
static constexpr int LaneIdY = 3;
static constexpr int LaneIdX = 4;
static constexpr int VecIdZ = 5;
static constexpr int VecIdY = 6;
static constexpr int VecIdX = 7;
static constexpr int NumDims = 8;
}  // namespace DimType

static std::string typeToString(int i) {
  switch (i) {
    case DimType::Batch0:
      return "Batch0";
    case DimType::Batch1:
      return "Batch1";
    case DimType::LaneIdZ:
      return "LaneIdZ";
    case DimType::LaneIdY:
      return "LaneIdY";
    case DimType::LaneIdX:
      return "LaneIdX";
    case DimType::VecIdZ:
      return "VecIdZ";
    case DimType::VecIdY:
      return "VecIdY";
    case DimType::VecIdX:
      return "VecIdX";
    default:
      return "";
  }
}

struct Dimension {
  int type;
  int value;
};

using DimOrderArray = std::array<std::array<Dimension, 3>, maxTensorDims>;
using OrderArray = std::array<std::array<int, 4>, maxTensorDims>;
using DimArray = std::array<int, DimType::NumDims>;

struct Layout {
  Layout(const DimOrderArray &orders,
         const std::array<int, maxTensorDims> &canonicalShape);
  // Reduces the layout along the tensor dim i
  // Updates the batch dims of the layout given the tensor dims
  void updateBatchDims(int dim0, int dim1);
  // Computes the ith dimension expression for a given state
  AffineExpr computeDim(int i, const DimArray &state, OpBuilder &builder);
  bool operator==(const Layout &layout) const { return shape == layout.shape; }
  bool operator!=(const Layout &layout) const { return shape != layout.shape; }
  void debugPrint(llvm::StringRef str) const;

  // Contains the shape of the layout
  DimArray shape;
  // Contains the order of layout dims for each of the tensor dims
  OrderArray order;
  // Shape of the tensor when used in mma
  std::array<int, maxTensorDims> canonicalShape;
  int rank;
};

// MMA Layout Utilities
enum class MMAType {
  M16N8K16,
  NONE,
};

enum class MMAMatrixType { AMatrix, BMatrix, CMatrix };

static std::array<Dimension, 3> getMMADimensions(MMAType mmaType,
                                                 MMAMatrixType matrixType,
                                                 int dim) {
  switch (mmaType) {
    case MMAType::M16N8K16:
      switch (matrixType) {
        case MMAMatrixType::AMatrix:
          if (dim == 0)
            return {{{DimType::LaneIdY, 8},
                     {DimType::VecIdZ, 2},
                     {DimType::LaneIdZ, 1}}};
          return {{{DimType::VecIdX, 2},
                   {DimType::LaneIdX, 4},
                   {DimType::VecIdY, 2}}};
        case MMAMatrixType::BMatrix:
          if (dim == 0)
            return {{{DimType::VecIdX, 2},
                     {DimType::LaneIdX, 4},
                     {DimType::VecIdY, 2}}};
          return {{{DimType::LaneIdY, 8},
                   {DimType::LaneIdZ, 1},
                   {DimType::VecIdZ, 1}}};
        case MMAMatrixType::CMatrix:
          if (dim == 0)
            return {{{DimType::LaneIdY, 8},
                     {DimType::VecIdY, 2},
                     {DimType::LaneIdZ, 1}}};
          return {{{DimType::VecIdX, 2},
                   {DimType::LaneIdX, 4},
                   {DimType::VecIdZ, 1}}};
      }
      break;
    default:
      return {};
  }
}

static std::array<int, 2> getMMACanonicalShape(MMAType mmaType,
                                               MMAMatrixType matrixType) {
  switch (mmaType) {
    case MMAType::M16N8K16:
      switch (matrixType) {
        case MMAMatrixType::AMatrix:
          return {16, 16};
        case MMAMatrixType::BMatrix:
        case MMAMatrixType::CMatrix:
          return {16, 8};
      }
      break;
    default:
      return {};
  }
}

Layout::Layout(const DimOrderArray &dimOrder,
               const std::array<int, maxTensorDims> &canonicalShape) {
  assert((dimOrder.size() > 0) && (dimOrder.size() <= maxTensorDims));
  for (int i = 0; i < maxTensorDims; i++) {
    int j;
    for (j = 0; j < dimOrder[i].size(); j++) {
      Dimension dim = dimOrder[i][j];
      order[i][j] = dim.type;
      shape[dim.type] = dim.value;
    }
    // Add batch dimension to the end
    if (i == 0) {
      order[i][j] = DimType::Batch0;
      shape[DimType::Batch0] = 1;
    } else {
      order[i][j] = DimType::Batch1;
      shape[DimType::Batch1] = 1;
    }
    this->canonicalShape[i] = canonicalShape[i];
  }
  rank = dimOrder.size();
}

void Layout::updateBatchDims(int dim0, int dim1) {
  shape[DimType::Batch0] = dim0 / canonicalShape[0];
  shape[DimType::Batch1] = dim1 / canonicalShape[1];
}

AffineExpr Layout::computeDim(int i, const DimArray &state,
                              OpBuilder &builder) {
  AffineExpr d0, d1, d2;
  bindDims(builder.getContext(), d0, d1, d2);
  AffineExpr dim = builder.getAffineConstantExpr(0);
  AffineExpr dimScale = builder.getAffineConstantExpr(1.0);
  for (const auto &dimType : order[i]) {
    switch (dimType) {
      case DimType::LaneIdX:
        dim = dim + dimScale * d0;
        break;
      case DimType::LaneIdY:
        dim = dim + dimScale * d1;
        break;
      case DimType::LaneIdZ:
        dim = dim + dimScale * d2;
        break;
      default:
        dim = dim + dimScale * builder.getAffineConstantExpr(state[dimType]);
        break;
    }
    dimScale = dimScale * builder.getAffineConstantExpr(shape[dimType]);
  }
  return dim;
}

void Layout::debugPrint(llvm::StringRef str) const {
  LLVM_DEBUG({
    llvm::dbgs() << str << " = \n";
    for (int i = 0; i < DimType::NumDims; i++) {
      llvm::dbgs() << "   " << typeToString(i) << ": " << shape[i] << " ";
      bool isRow{false};
      for (int k = 0; k < order[0].size(); k++) {
        if (order[0][k] == i) {
          isRow = true;
          break;
        }
      }
      if (isRow)
        llvm::dbgs() << "(R)";
      else
        llvm::dbgs() << "(C)";
      llvm::dbgs() << "  ";
    }
    llvm::dbgs() << "\n";
  });
}

static MMAType getMMAType(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                          ArrayRef<int64_t> cShape) {
  if ((aShape[0] % 16 == 0) && (aShape[1] % 16 == 0) && (cShape[0] % 16 == 0) &&
      (cShape[1] % 8 == 0)) {
    if ((bShape[0] % 16 == 0) && (bShape[1] % 8 == 0)) return MMAType::M16N8K16;
  }
  return MMAType::NONE;
}

void setMMALayout(Value aMatrix, Value bMatrix, Value cMatrix,
                  DenseMap<Value, Layout> &layoutMap) {
  // First determine which variant of MMA this op is most suitable for
  auto aType = aMatrix.getType().cast<ShapedType>();
  auto bType = aMatrix.getType().cast<ShapedType>();
  auto cType = aMatrix.getType().cast<ShapedType>();
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  ArrayRef<int64_t> cShape = cType.getShape();
  MMAType mmaType = getMMAType(aShape, bShape, cShape);
  if (mmaType == MMAType::NONE) return;
  // Set layouts for A, B and C
  auto setLayout = [&](Value matrix, MMAMatrixType matrixType,
                       llvm::StringRef name) {
    DimOrderArray dimOrder;
    for (int i = 0; i < 2; i++) {
      dimOrder[i] = getMMADimensions(mmaType, matrixType, i);
    }
    std::array<int, 2> canonicalShape =
        getMMACanonicalShape(mmaType, matrixType);
    Layout layout(dimOrder, canonicalShape);
    ArrayRef<int64_t> shape = matrix.getType().cast<ShapedType>().getShape();
    layout.updateBatchDims(shape[0], shape[1]);
    layoutMap.try_emplace(matrix, layout);
    layout.debugPrint(name);
  };
  setLayout(aMatrix, MMAMatrixType::AMatrix, "aMatrix");
  setLayout(bMatrix, MMAMatrixType::BMatrix, "bMatrix");
  setLayout(cMatrix, MMAMatrixType::CMatrix, "cMatrix");
}

void propagateLayout(Operation *op, DenseMap<Value, Layout> &layoutMap) {
  // TODO: Not Implemented
}

void distributeTransferReads(vector::TransferReadOp readOp,
                             DenseMap<Value, Layout> &layoutMap,
                             DenseMap<Value, Value> &simdToSimtMap,
                             OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(readOp);
  Value result = readOp.getResult();
  if (!layoutMap.count(result)) return;
  Value source = readOp.getSource();
  Location loc = readOp.getLoc();
  SmallVector<Value> indices = readOp.getIndices();
  Type elementType = source.getType().cast<ShapedType>().getElementType();
  Value threadIdX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value threadIdY = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
  Value threadIdZ = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
  Layout layout = layoutMap.at(result);
  auto vecType = VectorType::get(
      {layout.shape[DimType::Batch0], layout.shape[DimType::Batch1],
       layout.shape[DimType::VecIdZ] * layout.shape[DimType::VecIdY],
       layout.shape[DimType::VecIdX]},
      elementType);
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));
  std::array<int, DimType::NumDims> state;
  for (int b0 = 0; b0 < layout.shape[DimType::Batch0]; b0++) {
    state[DimType::Batch0] = b0;
    for (int b1 = 0; b1 < layout.shape[DimType::Batch1]; b1++) {
      state[DimType::Batch1] = b1;
      for (int i = 0; i < layout.shape[DimType::VecIdZ]; i++) {
        state[DimType::VecIdZ] = i;
        for (int j = 0; j < layout.shape[DimType::VecIdY]; j++) {
          state[DimType::VecIdY] = j;
          for (int k = 0; k < layout.shape[DimType::VecIdX]; k++) {
            state[DimType::VecIdX] = k;
            AffineExpr row = layout.computeDim(0, state, rewriter);
            AffineMap rowMap = AffineMap::get(3, 0, row, rewriter.getContext());
            Value rowIndex = rewriter.create<AffineApplyOp>(
                loc, rowMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            AffineExpr col = layout.computeDim(1, state, rewriter);
            AffineMap colMap = AffineMap::get(3, 0, col, rewriter.getContext());
            Value colIndex = rewriter.create<AffineApplyOp>(
                loc, colMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            if (layout.rank == 1) indices.back() = rowIndex;
            if (layout.rank == 2) {
              assert(indices.size() >= 2);
              indices[indices.size() - 2] = rowIndex;
              indices[indices.size() - 1] = colIndex;
            }
            Value el = rewriter.create<memref::LoadOp>(loc, source, indices);
            auto vectorType = VectorType::get({1}, elementType);
            Value v = rewriter.create<vector::BroadcastOp>(loc, vectorType, el);
            SmallVector<int64_t> offsets{
                b0, b1, j * layout.shape[DimType::VecIdZ] + i, k};
            SmallVector<int64_t> strides{1};
            vector = rewriter.create<vector::InsertStridedSliceOp>(
                loc, v, vector, offsets, strides);
          }
        }
      }
    }
  }
  simdToSimtMap.try_emplace(result, vector);
}

void distributeContracts(vector::ContractionOp contractOp,
                         DenseMap<Value, Layout> &layoutMap,
                         DenseMap<Value, Value> &simdToSimtMap,
                         OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(contractOp);
  Value lhs = contractOp.getLhs();
  if (!layoutMap.count(lhs)) return;
  if (!simdToSimtMap.count(lhs)) return;
  Type elementType = lhs.getType().cast<ShapedType>().getElementType();
  Value rhs = contractOp.getRhs();
  if (!layoutMap.count(rhs)) return;
  if (!simdToSimtMap.count(rhs)) return;
  Location loc = contractOp.getLoc();
  Value contractResult = contractOp.getResult();
  Layout lhsLayout = layoutMap.at(lhs);
  Layout resultLayout = layoutMap.at(contractResult);
  SmallVector<int64_t> vecShape{
      resultLayout.shape[DimType::Batch0], resultLayout.shape[DimType::Batch1],
      resultLayout.shape[DimType::VecIdZ] * resultLayout.shape[DimType::VecIdY],
      resultLayout.shape[DimType::VecIdX]};
  auto vecType = VectorType::get(vecShape, elementType);
  Value result = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));
  int M = resultLayout.shape[DimType::Batch0];
  int N = resultLayout.shape[DimType::Batch1];
  int canonicalM = resultLayout.canonicalShape[0];
  int canonicalN = resultLayout.canonicalShape[1];
  int K = lhsLayout.shape[DimType::Batch1];
  int canonicalK = lhsLayout.canonicalShape[1];
  auto cType = VectorType::get({vecShape[2], vecShape[3]}, elementType);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      Value cMatrix = rewriter.create<arith::ConstantOp>(
          loc, cType, rewriter.getZeroAttr(cType));
      for (int k = 0; k < K; k++) {
        Value aMatrix = rewriter.create<vector::ExtractOp>(
            loc, simdToSimtMap.at(lhs), SmallVector<int64_t>{i, k});
        Value bMatrix = rewriter.create<vector::ExtractOp>(
            loc, simdToSimtMap.at(rhs), SmallVector<int64_t>{k, j});
        cMatrix = rewriter.create<nvgpu::MmaSyncOp>(
            loc, aMatrix, bMatrix, cMatrix,
            rewriter.getI64ArrayAttr({canonicalM, canonicalN, canonicalK}));
      }
      result = rewriter.create<vector::InsertOp>(loc, cMatrix, result,
                                                 SmallVector<int64_t>{i, j});
    }
  }
  simdToSimtMap.try_emplace(contractResult, result);
}

void distributeTransferWrites(vector::TransferWriteOp writeOp,
                              DenseMap<Value, Layout> &layoutMap,
                              DenseMap<Value, Value> &simdToSimtMap,
                              OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(writeOp);
  Value vector = writeOp.getVector();
  Value source = writeOp.getSource();
  Location loc = writeOp.getLoc();
  SmallVector<Value> indices = writeOp.getIndices();
  Value threadIdX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value threadIdY = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
  Value threadIdZ = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);
  if (!layoutMap.count(vector)) return;
  if (!simdToSimtMap.count(vector)) return;
  Layout layout = layoutMap.at(vector);
  std::array<int, DimType::NumDims> state;
  for (int b0 = 0; b0 < layout.shape[DimType::Batch0]; b0++) {
    state[DimType::Batch0] = b0;
    for (int b1 = 0; b1 < layout.shape[DimType::Batch1]; b1++) {
      state[DimType::Batch1] = b1;
      for (int i = 0; i < layout.shape[DimType::VecIdZ]; i++) {
        state[DimType::VecIdZ] = i;
        for (int j = 0; j < layout.shape[DimType::VecIdY]; j++) {
          state[DimType::VecIdY] = j;
          for (int k = 0; k < layout.shape[DimType::VecIdX]; k++) {
            state[DimType::VecIdX] = k;
            Value v = rewriter.create<vector::ExtractOp>(
                loc, simdToSimtMap.at(vector),
                SmallVector<int64_t>{b0, b1,
                                     j * layout.shape[DimType::VecIdZ] + i, k});
            AffineExpr row = layout.computeDim(0, state, rewriter);
            AffineMap rowMap = AffineMap::get(3, 0, row, rewriter.getContext());
            Value rowIndex = rewriter.create<AffineApplyOp>(
                loc, rowMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            AffineExpr col = layout.computeDim(1, state, rewriter);
            AffineMap colMap = AffineMap::get(3, 0, col, rewriter.getContext());
            Value colIndex = rewriter.create<AffineApplyOp>(
                loc, colMap,
                SmallVector<Value>{threadIdX, threadIdY, threadIdZ});
            if (layout.rank == 1) indices.back() = rowIndex;
            if (layout.rank == 2) {
              assert(indices.size() >= 2);
              indices[indices.size() - 2] = rowIndex;
              indices[indices.size() - 1] = colIndex;
            }
            rewriter.create<memref::StoreOp>(loc, v, source, indices);
          }
        }
      }
    }
  }
}

template <typename T>
static void eraseOps(func::FuncOp funcOp, IRRewriter &rewriter) {
  funcOp.walk([&](T op) {
    rewriter.eraseOp(op);
    return WalkResult::advance();
  });
}

}  // namespace

void doLayoutAnalysisAndDistribution(IRRewriter &rewriter,
                                     func::FuncOp funcOp) {
  // First compute the layouts (set MMA layouts and propagate to rest)
  DenseMap<Value, Layout> layoutMap;
  funcOp.walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      Value lhs = contractOp.getLhs();
      Value rhs = contractOp.getRhs();
      Value result = contractOp.getResult();
      setMMALayout(lhs, rhs, result, layoutMap);
    } else {
      propagateLayout(op, layoutMap);
    }
    return WalkResult::advance();
  });

  // Apply SIMD to SIMT conversion
  DenseMap<Value, Value> simdToSimtMap;
  funcOp.walk([&](Operation *op) {
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      distributeTransferReads(readOp, layoutMap, simdToSimtMap, rewriter);
    }
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      distributeContracts(contractOp, layoutMap, simdToSimtMap, rewriter);
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      distributeTransferWrites(writeOp, layoutMap, simdToSimtMap, rewriter);
    }
    return WalkResult::advance();
  });

  // Erase old ops
  eraseOps<vector::TransferWriteOp>(funcOp, rewriter);
  eraseOps<vector::ContractionOp>(funcOp, rewriter);
  eraseOps<vector::TransferReadOp>(funcOp, rewriter);
}

}  // namespace mlir::iree_compiler
