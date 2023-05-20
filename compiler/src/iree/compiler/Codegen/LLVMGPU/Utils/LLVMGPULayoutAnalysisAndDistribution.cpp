// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-llvmgpu-layout-analysis-and-distribution"

namespace mlir::iree_compiler {

namespace {

static constexpr int warpSize = 32;
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
            return {{{DimType::LaneIdY, 8},
                     {DimType::LaneIdZ, 1},
                     {DimType::VecIdZ, 1}}};
          return {{{DimType::VecIdX, 2},
                   {DimType::LaneIdX, 4},
                   {DimType::VecIdY, 2}}};
        case MMAMatrixType::CMatrix:
          if (dim == 0)
            return {{{DimType::LaneIdY, 8},
                     {DimType::VecIdY, 2},
                     {DimType::LaneIdZ, 1}}};
          return {{{DimType::VecIdX, 2},
                   {DimType::LaneIdX, 4},
                   {DimType::VecIdZ, 1}}};
      }
      return {};
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
          return {8, 16};
        case MMAMatrixType::CMatrix:
          return {16, 8};
      }
      return {};
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
  AffineExpr dimScale = builder.getAffineConstantExpr(1);
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

// The value needs to have the target layout before op, so
// we create a layout conflict op that resolves the layout differences
// before the op.
static void createLayoutConflictOp(Value value, Layout targetLayout,
                                   DenseMap<Value, Layout> &layoutMap,
                                   Operation *op, IRRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  vector::ShapeCastOp conflictOp = rewriter.create<vector::ShapeCastOp>(
      op->getLoc(), value.getType(), value);
  Value resolvedValue = conflictOp.getResult();
  layoutMap.try_emplace(resolvedValue, targetLayout);
  layoutMap.at(resolvedValue).debugPrint("layout conflict resolved");
  rewriter.replaceAllUsesExcept(value, resolvedValue, conflictOp);
}

static void setMMALayout(Value aMatrix, Value bMatrix, Value cMatrix,
                         Value dMatrix, DenseMap<Value, Layout> &layoutMap,
                         Operation *op, IRRewriter &rewriter) {
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
    if (layoutMap.count(matrix) && (layout != layoutMap.at(matrix))) {
      createLayoutConflictOp(matrix, layout, layoutMap, op, rewriter);
    } else {
      layoutMap.try_emplace(matrix, layout);
      layout.debugPrint(name);
    }
  };
  setLayout(aMatrix, MMAMatrixType::AMatrix, "aMatrix");
  setLayout(bMatrix, MMAMatrixType::BMatrix, "bMatrix");
  setLayout(cMatrix, MMAMatrixType::CMatrix, "cMatrix");
  setLayout(dMatrix, MMAMatrixType::CMatrix, "dMatrix");
}

static void propagateLayoutToReduceBroadcastTranspose(
    vector::MultiDimReductionOp reductionOp, vector::BroadcastOp broadcastOp,
    vector::TransposeOp transposeOp, DenseMap<Value, Layout> &layoutMap) {
  if (!broadcastOp) return;
  if (!transposeOp) return;
  Value reductionSrc = reductionOp.getSource();
  if (!layoutMap.count(reductionSrc)) return;
  // Get the reduction dims
  auto reductionDims = llvm::to_vector<4>(
      reductionOp.getReductionDims().getAsRange<IntegerAttr>());
  // Get the transpose permutation
  SmallVector<int64_t> perm;
  transposeOp.getTransp(perm);
  // Don't support dim-1 broadcasted dims
  llvm::SetVector<int64_t> dimOneBroadcastedDims =
      broadcastOp.computeBroadcastedUnitDims();
  if (dimOneBroadcastedDims.size() > 0) return;
  Value broadcastSource = broadcastOp.getSource();
  Value broadcastResult = broadcastOp.getResult();
  int64_t broadcastSourceRank =
      broadcastSource.getType().cast<VectorType>().getRank();
  int64_t broadcastResultRank =
      broadcastResult.getType().cast<VectorType>().getRank();
  int64_t rankDiff = broadcastResultRank - broadcastSourceRank;
  llvm::SetVector<int64_t> broadcastedDims;
  for (int64_t i = 0; i < rankDiff; i++) broadcastedDims.insert(i);
  ArrayRef<int64_t> broadcastShape =
      broadcastResult.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> srcShape =
      reductionSrc.getType().cast<ShapedType>().getShape();
  // Check that the same number of dims are reduced and broadcasted
  if (reductionDims.size() != broadcastedDims.size()) return;
  // Check that transpose(reductionDim) == broadcastDim
  // and that the shapes match
  for (IntegerAttr dimAttr : reductionDims) {
    int64_t dim = dimAttr.getInt();
    int64_t transposedDim = perm[dim];
    if (!broadcastedDims.contains(transposedDim)) return;
    if (srcShape[dim] != broadcastShape[transposedDim]) return;
  }
  Value transposedResult = transposeOp.getResult();
  layoutMap.try_emplace(transposedResult, layoutMap.at(reductionSrc));
  layoutMap.at(transposedResult).debugPrint("transposed");
  // Propagate 2D layout to 1D accumulator
  Value acc = reductionOp.getAcc();
  if (layoutMap.count(acc)) return;
  Layout accLayout = layoutMap.at(reductionSrc);
  accLayout.rank = 1;
  layoutMap.try_emplace(acc, accLayout);
}

static std::tuple<vector::BroadcastOp, vector::TransposeOp>
checkForReduceBroadcastTranspose(vector::MultiDimReductionOp reductionOp) {
  vector::BroadcastOp broadcastOp{nullptr};
  vector::TransposeOp transposeOp{nullptr};
  for (Operation *user : reductionOp.getResult().getUsers()) {
    if (auto broadcast = dyn_cast<vector::BroadcastOp>(user)) {
      for (Operation *bUser : broadcast.getResult().getUsers()) {
        if (auto transpose = dyn_cast<vector::TransposeOp>(bUser)) {
          transposeOp = transpose;
          break;
        }
      }
      broadcastOp = broadcast;
      break;
    }
  }
  return std::make_tuple(broadcastOp, transposeOp);
}

static void propagateLayoutToFor(scf::ForOp forOp,
                                 DenseMap<Value, Layout> &layoutMap) {
  for (auto argIndex : llvm::enumerate(forOp.getRegionIterArgs())) {
    BlockArgument &arg = argIndex.value();
    if (!layoutMap.count(arg)) continue;
    OpOperand &operand = forOp.getOpOperandForRegionIterArg(arg);
    Value result = forOp.getResult(argIndex.index());
    Layout newLayout = layoutMap.at(arg);
    layoutMap.try_emplace(operand.get(), newLayout);
    layoutMap.try_emplace(result, newLayout);
    layoutMap.at(operand.get()).debugPrint("for operand");
    layoutMap.at(result).debugPrint("for result");
  }
}

static void propagateLayoutToOthers(SmallVectorImpl<Value> &operands,
                                    DenseMap<Value, Layout> &layoutMap) {
  int numOperands = operands.size();
  // Find an operand with a layout
  int i;
  for (i = 0; i < numOperands; i++) {
    if (layoutMap.count(operands[i])) break;
  }
  // Propagate layout to others
  for (int j = 0; j < numOperands; j++) {
    if (j == i) continue;
    if (!layoutMap.count(operands[j])) {
      layoutMap.try_emplace(operands[j], layoutMap.at(operands[i]));
      layoutMap.at(operands[j]).debugPrint("binary/unary operand");
    } else {
      assert(layoutMap.at(operands[i]) == layoutMap.at(operands[j]));
    }
  }
}

static void propagateLayoutToElementwiseOp(Operation *op,
                                           DenseMap<Value, Layout> &layoutMap) {
  if (!OpTrait::hasElementwiseMappableTraits(op) || op->getNumResults() != 1)
    return;
  auto operands = llvm::to_vector(op->getOperands());
  operands.push_back(op->getResult(0));
  propagateLayoutToOthers(operands, layoutMap);
}

static void propagateLayout(Operation *op, DenseMap<Value, Layout> &layoutMap) {
  if (auto reductionOp = dyn_cast<vector::MultiDimReductionOp>(op)) {
    auto [broadcastOp, transposeOp] =
        checkForReduceBroadcastTranspose(reductionOp);
    propagateLayoutToReduceBroadcastTranspose(reductionOp, broadcastOp,
                                              transposeOp, layoutMap);
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    propagateLayoutToFor(forOp, layoutMap);
  }
  propagateLayoutToElementwiseOp(op, layoutMap);
}

/// Get indices of transfer op after distribution.
static SmallVector<Value> getDistributedIndices(
    OpBuilder &rewriter, Location loc, Layout &layout,
    std::array<int, DimType::NumDims> &state, ArrayRef<Value> indices,
    AffineMap permutationMap, const std::array<Value, 3> &threadIds) {
  AffineExpr row = layout.computeDim(0, state, rewriter);
  AffineMap rowMap = AffineMap::get(3, 0, row, rewriter.getContext());
  std::array<Value, 2> laneOffsets;
  laneOffsets[0] =
      rewriter.create<affine::AffineApplyOp>(loc, rowMap, threadIds);
  AffineExpr col = layout.computeDim(1, state, rewriter);
  AffineMap colMap = AffineMap::get(3, 0, col, rewriter.getContext());
  laneOffsets[1] =
      rewriter.create<affine::AffineApplyOp>(loc, colMap, threadIds);
  SmallVector<Value> newIndices{indices.begin(), indices.end()};
  int64_t laneDim = 0;
  for (AffineExpr expr : permutationMap.getResults()) {
    auto dimExpr = expr.dyn_cast<AffineDimExpr>();
    if (!dimExpr) continue;
    unsigned pos = dimExpr.getPosition();
    newIndices[pos] = rewriter.create<arith::AddIOp>(
        loc, laneOffsets[laneDim++], newIndices[pos]);
  }
  return newIndices;
}

/// Return true if the read op with the given layout can be implemented using
/// ldmatrix op.
static bool isLdMatrixCompatible(vector::TransferReadOp readOp,
                                 const Layout &layout) {
  // ldMatrix requires loading from shared memory.
  if (!hasSharedMemoryAddressSpace(
          readOp.getSource().getType().cast<MemRefType>()))
    return false;
  // TODO: Can be any 16bits type.
  if (!readOp.getVectorType().getElementType().isF16()) return false;
  bool compatibleLayout = layout.order.back()[0] == DimType::VecIdX &&
                          layout.shape[DimType::VecIdX] == 2 &&
                          layout.order.back()[1] == DimType::LaneIdX &&
                          layout.shape[DimType::LaneIdX] == 4 &&
                          layout.shape[DimType::LaneIdY] == 8;
  return compatibleLayout;
}

/// Return true if the permutation map requires using a transposed ldmatrix op.
static bool isTransposedLdMatrix(AffineMap map) {
  if (map.getNumResults() != 2) {
    return false;
  }
  auto exprX = map.getResult(0).dyn_cast<AffineDimExpr>();
  auto exprY = map.getResult(1).dyn_cast<AffineDimExpr>();
  if (!exprX || !exprY) return false;
  return exprX.getPosition() > exprY.getPosition();
}

/// Emit nvgpu.ldmatrix code sequence at the given offset.
/// ldmatrix loads 8 blocks of 8 contiguous 16bits elements. The start address
/// of each blocks are packed in 8 lanes. Therefore we first calculate the
/// common base offset for all the lanes and we add the block offsets using
/// `blockid = flatid %8`
static Value emitLdMatrix(OpBuilder &rewriter, Location loc, Layout &layout,
                          std::array<int, DimType::NumDims> &state,
                          ArrayRef<Value> indices, AffineMap permutationMap,
                          const std::array<Value, 3> &threadIds,
                          Value memrefValue) {
  bool transpose = isTransposedLdMatrix(permutationMap);
  // First compute the vector part of the offset.
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  std::array<Value, 3> threadIdsLdMatrix = {zero, zero, zero};
  AffineExpr row = layout.computeDim(0, state, rewriter);
  AffineMap rowMap = AffineMap::get(3, 0, row, rewriter.getContext());
  std::array<Value, 2> vectorOffsets;
  vectorOffsets[0] =
      rewriter.create<affine::AffineApplyOp>(loc, rowMap, threadIdsLdMatrix);
  AffineExpr col = layout.computeDim(1, state, rewriter);
  AffineMap colMap = AffineMap::get(3, 0, col, rewriter.getContext());
  vectorOffsets[1] =
      rewriter.create<affine::AffineApplyOp>(loc, colMap, threadIdsLdMatrix);

  // Then compute the offset for each lane.
  AffineExpr d0, d1, d2;
  bindDims(rewriter.getContext(), d0, d1, d2);
  AffineExpr laneIdModuloEight = (d0 + d1 * 4) % 8;
  Value laneId = rewriter.create<affine::AffineApplyOp>(
      loc, laneIdModuloEight, ArrayRef<Value>({threadIds[0], threadIds[1]}));
  std::array<int, DimType::NumDims> emptyState = {0};
  threadIdsLdMatrix = {zero, laneId, zero};
  AffineExpr row2 = layout.computeDim(0, emptyState, rewriter);
  AffineMap rowMap2 = AffineMap::get(3, 0, row2, rewriter.getContext());
  std::array<Value, 2> laneOffsets;
  laneOffsets[0] =
      rewriter.create<affine::AffineApplyOp>(loc, rowMap2, threadIdsLdMatrix);
  AffineExpr col2 = layout.computeDim(1, emptyState, rewriter);
  AffineMap colMap2 = AffineMap::get(3, 0, col2, rewriter.getContext());
  laneOffsets[1] =
      rewriter.create<affine::AffineApplyOp>(loc, colMap2, threadIdsLdMatrix);
  SmallVector<Value> newIndices{indices.begin(), indices.end()};
  int64_t laneDim = 0;
  // When transposing we need to swap the lane offsets.
  if (transpose) {
    std::swap(laneOffsets[0], laneOffsets[1]);
  }
  for (AffineExpr expr : permutationMap.getResults()) {
    auto dimExpr = expr.dyn_cast<AffineDimExpr>();
    if (!dimExpr) continue;
    unsigned pos = dimExpr.getPosition();
    newIndices[pos] = rewriter.create<arith::AddIOp>(
        loc, vectorOffsets[laneDim], newIndices[pos]);
    newIndices[pos] = rewriter.create<arith::AddIOp>(
        loc, laneOffsets[laneDim++], newIndices[pos]);
  }
  Type vecType = VectorType::get(
      {1, 2}, memrefValue.getType().cast<MemRefType>().getElementType());
  Value el = rewriter.create<nvgpu::LdMatrixOp>(loc, vecType, memrefValue,
                                                newIndices, transpose, 1);
  return el;
}

static void distributeTransferReads(vector::TransferReadOp readOp,
                                    DenseMap<Value, Layout> &layoutMap,
                                    DenseMap<Value, Value> &simdToSimtMap,
                                    OpBuilder &rewriter,
                                    llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(readOp);
  Value result = readOp.getResult();
  if (!layoutMap.count(result)) return;
  Value source = readOp.getSource();
  Location loc = readOp.getLoc();
  SmallVector<Value> indices = readOp.getIndices();
  Type elementType = source.getType().cast<ShapedType>().getElementType();
  std::array<Value, 3> threadIds = {
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y),
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z)};
  Layout layout = layoutMap.at(result);
  auto vecType = VectorType::get(
      {layout.shape[DimType::Batch0], layout.shape[DimType::Batch1],
       layout.shape[DimType::VecIdZ] * layout.shape[DimType::VecIdY],
       layout.shape[DimType::VecIdX]},
      elementType);
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));
  std::array<int, DimType::NumDims> state;
  bool useLdMatrix = isLdMatrixCompatible(readOp, layout);
  for (int b0 = 0; b0 < layout.shape[DimType::Batch0]; b0++) {
    state[DimType::Batch0] = b0;
    for (int b1 = 0; b1 < layout.shape[DimType::Batch1]; b1++) {
      state[DimType::Batch1] = b1;
      for (int i = 0; i < layout.shape[DimType::VecIdZ]; i++) {
        state[DimType::VecIdZ] = i;
        for (int j = 0; j < layout.shape[DimType::VecIdY]; j++) {
          state[DimType::VecIdY] = j;
          if (useLdMatrix) {
            state[DimType::VecIdX] = 0;
            Value ld =
                emitLdMatrix(rewriter, loc, layout, state, indices,
                             readOp.getPermutationMap(), threadIds, source);
            SmallVector<int64_t> offsets{
                b0, b1, j * layout.shape[DimType::VecIdZ] + i, 0};
            SmallVector<int64_t> strides{1, 1};
            vector = rewriter.create<vector::InsertStridedSliceOp>(
                loc, ld, vector, offsets, strides);
            continue;
          }
          for (int k = 0; k < layout.shape[DimType::VecIdX]; k++) {
            state[DimType::VecIdX] = k;
            SmallVector<Value> newIndices =
                getDistributedIndices(rewriter, loc, layout, state, indices,
                                      readOp.getPermutationMap(), threadIds);
            Value el = rewriter.create<memref::LoadOp>(loc, source, newIndices);
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
  ops.insert(readOp);
}

static void distributeContracts(vector::ContractionOp contractOp,
                                DenseMap<Value, Layout> &layoutMap,
                                DenseMap<Value, Value> &simdToSimtMap,
                                OpBuilder &rewriter,
                                llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(contractOp);
  Value lhs = contractOp.getLhs();
  if (!layoutMap.count(lhs)) return;
  if (!simdToSimtMap.count(lhs)) return;
  Type elementType = lhs.getType().cast<ShapedType>().getElementType();
  Value rhs = contractOp.getRhs();
  if (!layoutMap.count(rhs)) return;
  if (!simdToSimtMap.count(rhs)) return;
  Value acc = contractOp.getAcc();
  if (!simdToSimtMap.count(acc)) return;
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
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      Value cMatrix = rewriter.create<vector::ExtractOp>(
          loc, simdToSimtMap.at(acc), SmallVector<int64_t>{i, j});
      for (int k = 0; k < K; k++) {
        Value aMatrix = rewriter.create<vector::ExtractOp>(
            loc, simdToSimtMap.at(lhs), SmallVector<int64_t>{i, k});
        Value bMatrix = rewriter.create<vector::ExtractOp>(
            loc, simdToSimtMap.at(rhs), SmallVector<int64_t>{j, k});
        cMatrix = rewriter.create<nvgpu::MmaSyncOp>(
            loc, aMatrix, bMatrix, cMatrix,
            rewriter.getI64ArrayAttr({canonicalM, canonicalN, canonicalK}));
      }
      result = rewriter.create<vector::InsertOp>(loc, cMatrix, result,
                                                 SmallVector<int64_t>{i, j});
    }
  }
  simdToSimtMap.try_emplace(contractResult, result);
  ops.insert(contractOp);
}

static void distributeTransferWrites(vector::TransferWriteOp writeOp,
                                     DenseMap<Value, Layout> &layoutMap,
                                     DenseMap<Value, Value> &simdToSimtMap,
                                     OpBuilder &rewriter,
                                     llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(writeOp);
  Value vector = writeOp.getVector();
  Value source = writeOp.getSource();
  Location loc = writeOp.getLoc();
  SmallVector<Value> indices = writeOp.getIndices();
  std::array<Value, 3> threadIds = {
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x),
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y),
      rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z)};
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
            SmallVector<Value> newIndices =
                getDistributedIndices(rewriter, loc, layout, state, indices,
                                      writeOp.getPermutationMap(), threadIds);
            rewriter.create<memref::StoreOp>(loc, v, source, newIndices);
          }
        }
      }
    }
  }
  ops.insert(writeOp);
}

static bool isBatchId(int dimType) {
  return ((dimType == DimType::Batch0) || (dimType == DimType::Batch1));
}

static bool isLaneId(int dimType) {
  return ((dimType == DimType::LaneIdX) || (dimType == DimType::LaneIdY) ||
          (dimType == DimType::LaneIdZ));
}

static bool isVectorId(int dimType) {
  return ((dimType == DimType::VecIdX) || (dimType == DimType::VecIdY) ||
          (dimType == DimType::VecIdZ));
}

static int getLaneIdIndex(std::array<int, 4> &order) {
  for (int i = 0; i < 4; i++) {
    if (isLaneId(order[i])) return i;
  }
  return -1;
}

static int isSingleLaneIdReduced(std::array<int, 4> &order) {
  int count{0};
  for (int i = 0; i < 4; i++) {
    if (isLaneId(order[i])) count++;
  }
  return count == 1;
}

static int getVecSizes(std::array<int, 4> &order, const Layout &layout) {
  int size = 1;
  for (int i = 0; i < 4; i++) {
    if (isVectorId(i)) size *= layout.shape[i];
  }
  return size;
}

using bodyType = std::function<void(std::array<int, DimType::NumDims> &)>;

/// This function iterates over the dimensions of a given column/row order
/// that are not LaneIdX, LaneIdY or LaneIdZ and executes the function body
/// inside the innermost loop. It keeps track of the induction variables
/// in the state array and passes them to the body function.
static void iterate(int dimType, ArrayRef<int> order,
                    std::array<int, DimType::NumDims> &state,
                    const Layout &layout, bodyType body) {
  if (dimType == DimType::NumDims) {
    body(state);
    return;
  }
  if ((std::find(order.begin(), order.end(), dimType) != order.end()) &&
      (!isLaneId(dimType))) {
    for (int i = 0; i < layout.shape[dimType]; i++) {
      state[dimType] = i;
      iterate(dimType + 1, order, state, layout, body);
    }
  } else {
    iterate(dimType + 1, order, state, layout, body);
  }
}

/// Computes the 4D SIMT vector index using the current value
/// of the induction variables of the loops being iterated
/// (state) [b0, b1, lz, ly, lx, vz, vy, vx]
/// and the layout shape.
/// Dim 0 of the SIMT vector maps to b0
/// Dim 1 of the SIMT vector maps to b1
/// Dim 2 of the SIMT vector maps to vy * VZ + vz
/// Dim 3 of the SIMT vector maps to vx
/// where VZ is the shape of the VectorZ dimension of the layout.
static SmallVector<int64_t> getIndicesFromState(
    std::array<int, DimType::NumDims> &state, Layout &layout) {
  SmallVector<int64_t> indices{
      state[DimType::Batch0], state[DimType::Batch1],
      state[DimType::VecIdY] * layout.shape[DimType::VecIdZ] +
          state[DimType::VecIdZ],
      state[DimType::VecIdX]};
  return indices;
}

static void distributeReductionBroadcastTranspose(
    vector::MultiDimReductionOp reductionOp, vector::BroadcastOp broadcastOp,
    vector::TransposeOp transposeOp, DenseMap<Value, Layout> &layoutMap,
    DenseMap<Value, Value> &simdToSimtMap, OpBuilder &rewriter,
    llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(reductionOp);
  Value source = reductionOp.getSource();
  Type elementType = source.getType().cast<ShapedType>().getElementType();
  if (!layoutMap.count(source)) return;
  if (!simdToSimtMap.count(source)) return;
  if (!broadcastOp) return;
  if (!transposeOp) return;
  Location loc = reductionOp.getLoc();
  Layout layout = layoutMap.at(source);
  auto reductionDims = llvm::to_vector<4>(
      reductionOp.getReductionDims().getAsRange<IntegerAttr>());
  vector::CombiningKind combiningKind = reductionOp.getKind();
  // Only support reduction on one dimension
  if (reductionDims.size() > 1) return;
  int reductionDim = reductionDims[0].getInt();
  std::array<int, 4> reductionOrder = layout.order[reductionDim];
  std::array<int, 4> parallelOrder = layout.order[!reductionDim];
  Value acc = reductionOp.getAcc();
  if (!simdToSimtMap.count(acc)) return;
  SmallVector<int64_t> vecShape{
      layout.shape[DimType::Batch0], layout.shape[DimType::Batch1],
      layout.shape[DimType::VecIdZ] * layout.shape[DimType::VecIdY],
      layout.shape[DimType::VecIdX]};
  auto vecType = VectorType::get(vecShape, elementType);
  Value output = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));

  if (!isSingleLaneIdReduced(reductionOrder)) return;
  int dimIndex = getLaneIdIndex(reductionOrder);
  int dimType = reductionOrder[dimIndex];
  int offset{0};
  switch (dimType) {
    case DimType::LaneIdX:
      offset = 1;
      break;
    case DimType::LaneIdY:
      offset = layout.shape[DimType::LaneIdX];
      break;
    case DimType::LaneIdZ:
      offset = layout.shape[DimType::LaneIdX] * layout.shape[DimType::LaneIdY];
      break;
  }

  bodyType loopBody = [&](std::array<int, DimType::NumDims> &state) {
    Value vector = simdToSimtMap.at(source);
    VectorType vectorType = vector.getType().cast<VectorType>();
    Type elementType = vectorType.getElementType();
    bool isFP32 = elementType.isF32();
    Value mask;

    Value accValue = rewriter.create<vector::ExtractOp>(
        loc, simdToSimtMap.at(acc), getIndicesFromState(state, layout));

    int index{0};
    auto zero =
        rewriter.getZeroAttr(VectorType::get({isFP32 ? 1 : 2}, elementType));
    Value tmp = rewriter.create<arith::ConstantOp>(loc, zero);
    Value result;

    // Returns vector<1xf32> or vector<2xf16>
    auto reduceLocal = [&](std::array<int, DimType::NumDims> &state) {
      Value current = rewriter.create<vector::ExtractOp>(
          loc, vector, getIndicesFromState(state, layout));
      tmp = rewriter.create<vector::InsertOp>(loc, current, tmp,
                                              SmallVector<int64_t>{index});
      if (!isFP32) {
        index = !index;
        if (index) return;
      }
      result = !result ? tmp
                       : makeArithReduction(rewriter, loc, combiningKind,
                                            result, tmp, mask);
    };
    iterate(0, reductionOrder, state, layout, reduceLocal);

    auto reduceGlobal = [&]() {
      for (uint64_t i = offset; i < offset * layout.shape[dimType]; i <<= 1) {
        Value packed = packVectorToSupportedWidth(loc, rewriter, result);
        auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
            loc, packed, i, warpSize, gpu::ShuffleMode::XOR);
        Value unpacked =
            unpackToVector(loc, rewriter, shuffleOp.getShuffleResult(),
                           result.getType().cast<VectorType>());
        result = makeArithReduction(rewriter, loc, combiningKind, unpacked,
                                    result, mask);
      }

      // Convert to f16 or f32
      Value v0 = rewriter.create<vector::ExtractOp>(loc, result,
                                                    SmallVector<int64_t>{0});
      if (isFP32) {
        result = makeArithReduction(rewriter, loc, combiningKind, v0, accValue,
                                    mask);
      } else {
        Value v1 = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{1});
        result = makeArithReduction(rewriter, loc, combiningKind, v0, v1, mask);
        result = makeArithReduction(rewriter, loc, combiningKind, result,
                                    accValue, mask);
      }
    };
    reduceGlobal();

    auto broadcastResult = [&](std::array<int, DimType::NumDims> &state) {
      output = rewriter.create<vector::InsertOp>(
          loc, result, output, getIndicesFromState(state, layout));
    };
    // Broadcast result to same shape as original
    iterate(0, reductionOrder, state, layout, broadcastResult);

    // Reset reduction state
    for (int type : reductionOrder) state[type] = 0;
  };

  std::array<int, DimType::NumDims> state;
  state.fill(0);
  iterate(0, parallelOrder, state, layout, loopBody);

  simdToSimtMap.try_emplace(transposeOp.getResult(), output);

  ops.insert(reductionOp);
  ops.insert(broadcastOp);
  ops.insert(transposeOp);
}

static void replaceForOpWithNewSignature(RewriterBase &rewriter,
                                         scf::ForOp loop,
                                         ValueRange newIterOperands,
                                         DenseMap<Value, Layout> &layoutMap,
                                         DenseMap<Value, Value> &simdToSimtMap,
                                         llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop);

  // Create a new loop before the existing one, with the extra operands.
  // We will be using dummy values instead of the old operands
  // only for those operands that are being distributed
  SmallVector<Value> newOperands;
  auto operands = llvm::to_vector<4>(loop.getIterOperands());
  for (auto operand : operands) {
    if (!layoutMap.count(operand)) {
      newOperands.push_back(operand);
      continue;
    }
    Value zero = rewriter.create<arith::ConstantOp>(
        loop.getLoc(), rewriter.getZeroAttr(operand.getType()));
    newOperands.push_back(zero);
  }

  newOperands.append(newIterOperands.begin(), newIterOperands.end());
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      newOperands);
  newLoop.getBody()->erase();

  newLoop.getLoopBody().getBlocks().splice(
      newLoop.getLoopBody().getBlocks().begin(),
      loop.getLoopBody().getBlocks());
  for (Value operand : newIterOperands)
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());

  // Replace old results and propagate layouts
  int numOldResults = loop.getNumResults();
  for (auto it : llvm::zip(loop.getResults(),
                           newLoop.getResults().take_front(numOldResults))) {
    if (layoutMap.count(std::get<0>(it)))
      layoutMap.try_emplace(std::get<1>(it), layoutMap.at(std::get<0>(it)));
    rewriter.replaceAllUsesWith(std::get<0>(it), std::get<1>(it));
  }

  // Propagate layout + mapping from old to new block args and results
  auto bbArgs = newLoop.getRegionIterArgs();
  auto results = newLoop.getResults();
  for (int i = 0; i < numOldResults; i++) {
    if (layoutMap.count(bbArgs[i]))
      layoutMap.try_emplace(bbArgs[i + numOldResults], layoutMap.at(bbArgs[i]));
    simdToSimtMap.try_emplace(bbArgs[i], bbArgs[i + numOldResults]);
    if (layoutMap.count(results[i]))
      layoutMap.try_emplace(results[i + numOldResults],
                            layoutMap.at(results[i]));
    simdToSimtMap.try_emplace(results[i], results[i + numOldResults]);
  }

  ops.insert(loop);
  return;
}

static void distributeFor(scf::ForOp forOp, DenseMap<Value, Layout> &layoutMap,
                          DenseMap<Value, Value> &simdToSimtMap,
                          IRRewriter &rewriter,
                          llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);

  SmallVector<Value> newOperands;
  for (const auto &operand : llvm::enumerate(forOp.getIterOperands())) {
    if (!simdToSimtMap.count(operand.value())) {
      continue;
    }
    newOperands.push_back(simdToSimtMap.at(operand.value()));
  }
  replaceForOpWithNewSignature(rewriter, forOp, newOperands, layoutMap,
                               simdToSimtMap, ops);
}

static void distributeYield(scf::YieldOp yieldOp,
                            DenseMap<Value, Layout> &layoutMap,
                            DenseMap<Value, Value> &simdToSimtMap,
                            IRRewriter &rewriter,
                            llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(yieldOp);

  // Update yield op with additional operand
  auto loop = cast<scf::ForOp>(yieldOp->getParentOp());
  auto yieldOperands = llvm::to_vector<4>(yieldOp.getOperands());
  for (const auto &operand : llvm::enumerate(yieldOp.getOperands())) {
    if (!simdToSimtMap.count(operand.value())) continue;
    // Replace the yield of old value with the for op argument to make it easier
    // to remove the dead code.
    yieldOperands[operand.index()] = loop.getIterOperands()[operand.index()];
    yieldOperands.push_back(simdToSimtMap.at(operand.value()));
  }
  rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldOperands);
  ops.insert(yieldOp);
}

static void distributeConstants(arith::ConstantOp constantOp,
                                DenseMap<Value, Layout> &layoutMap,
                                DenseMap<Value, Value> &simdToSimtMap,
                                IRRewriter &rewriter,
                                llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(constantOp);
  Value constant = constantOp.getResult();
  if (!layoutMap.count(constant)) return;
  auto attr = constantOp.getValue().cast<DenseElementsAttr>();
  // Only handle splat values for now
  if (!attr.isSplat()) return;
  Layout layout = layoutMap.at(constant);
  Type elementType = constant.getType().cast<VectorType>().getElementType();
  auto vType = VectorType::get(
      {layout.shape[DimType::Batch0], layout.shape[DimType::Batch1],
       layout.shape[DimType::VecIdZ] * layout.shape[DimType::VecIdY],
       layout.shape[DimType::VecIdX]},
      elementType);
  Value result = rewriter.create<arith::ConstantOp>(
      constantOp.getLoc(), vType,
      DenseElementsAttr::get(vType, attr.getSplatValue<APFloat>()));
  simdToSimtMap.try_emplace(constant, result);
}

static void distributeElementwise(Operation *op,
                                  DenseMap<Value, Layout> &layoutMap,
                                  DenseMap<Value, Value> &simdToSimtMap,
                                  IRRewriter &rewriter,
                                  llvm::SetVector<Operation *> &ops) {
  if (!OpTrait::hasElementwiseMappableTraits(op)) return;
  if (op->getNumResults() != 1) return;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  SmallVector<Value> newOperands;
  for (auto operand : op->getOperands()) {
    if (!simdToSimtMap.count(operand)) return;
    newOperands.push_back(simdToSimtMap.at(operand));
  }
  SmallVector<Type> resultTypes{newOperands.front().getType()};
  Operation *newOp =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), newOperands,
                      resultTypes, op->getAttrs());
  simdToSimtMap.try_emplace(op->getResult(0), newOp->getResult(0));
  ops.insert(op);
}

static Value resolveBatchConflict(SmallVectorImpl<int> &mismatchedDims,
                                  Value vector, const Layout &targetLayout,
                                  const Layout &currentLayout,
                                  IRRewriter &rewriter, Location loc) {
  assert(mismatchedDims.size() == 1);
  int batchDim = mismatchedDims[0];
  VectorType vectorType = vector.getType().cast<VectorType>();
  ArrayRef<int64_t> vectorShape = vectorType.getShape();
  SmallVector<int64_t> offsets(vectorShape.size(), 0);
  SmallVector<int64_t> strides(vectorShape.size(), 1);
  SmallVector<int64_t> shape(vectorShape);
  shape[batchDim] = targetLayout.shape[batchDim];

  Value newVector;
  // If target layout shape is less than current, then extract a slice,
  // otherwise broadcast
  if (currentLayout.shape[batchDim] > targetLayout.shape[batchDim]) {
    newVector = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, vector, offsets, shape, strides);
  } else {
    Value transposedVector = vector;
    if (batchDim == DimType::Batch1) {
      transposedVector = rewriter.create<vector::TransposeOp>(
          loc, transposedVector, ArrayRef<int64_t>{1, 0, 2, 3});
      std::swap(shape[DimType::Batch0], shape[DimType::Batch1]);
    }
    transposedVector = rewriter.create<vector::ExtractOp>(loc, transposedVector,
                                                          ArrayRef<int64_t>{0});
    Type elementType = vectorType.getElementType();
    newVector = rewriter.create<vector::BroadcastOp>(
        loc, VectorType::get(shape, elementType), transposedVector);
    if (batchDim == DimType::Batch1) {
      newVector = rewriter.create<vector::TransposeOp>(
          loc, newVector, ArrayRef<int64_t>{1, 0, 2, 3});
    }
  }
  return newVector;
}

static Value resolveBatchVectorConflict(SmallVectorImpl<int> &mismatchedDims,
                                        Value vector,
                                        const Layout &targetLayout,
                                        const Layout &currentLayout,
                                        IRRewriter &rewriter, Location loc) {
  int numMismatchedVecDims{0};
  int vecDim, batchDim;
  for (auto dimType : mismatchedDims) {
    if (isVectorId(dimType)) {
      numMismatchedVecDims++;
      vecDim = dimType;
    }
    if (isBatchId(dimType)) batchDim = dimType;
  }
  // Only support single vector mismatched dim
  if (numMismatchedVecDims > 1) return Value{};
  // Assumes target layout vector dim > current layout vector dim
  int ratio = ((float)targetLayout.shape[vecDim] / currentLayout.shape[vecDim]);

  // Check that the batch can be used to compensate for vector layout
  // differences
  if (currentLayout.shape[batchDim] != targetLayout.shape[batchDim] * ratio)
    return Value{};

  SmallVector<int64_t> vecShape{
      targetLayout.shape[DimType::Batch0], targetLayout.shape[DimType::Batch1],
      targetLayout.shape[DimType::VecIdZ] * targetLayout.shape[DimType::VecIdY],
      targetLayout.shape[DimType::VecIdX]};
  Type elementType = vector.getType().cast<VectorType>().getElementType();
  auto vecType = VectorType::get(vecShape, elementType);
  Value newVector = rewriter.create<arith::ConstantOp>(
      loc, vecType, rewriter.getZeroAttr(vecType));
  for (int i = 0; i < currentLayout.shape[DimType::Batch0]; i++) {
    for (int j = 0, offset = 0; j < currentLayout.shape[DimType::Batch1]; j++) {
      Value slice = rewriter.create<vector::ExtractOp>(
          loc, vector, SmallVector<int64_t>{i, j});
      int newI = batchDim == DimType::Batch0 ? i / ratio : i;
      int newJ = batchDim == DimType::Batch0 ? j : j / ratio;
      int newK = vecDim == DimType::VecIdX ? 0 : ratio * offset;
      int newL = vecDim == DimType::VecIdX ? ratio * offset : 0;
      SmallVector<int64_t> offsets{newI, newJ, newK, newL};
      SmallVector<int64_t> strides{1, 1};
      newVector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, slice, newVector, offsets, strides);
      offset = (offset + 1) % ratio;
    }
  }
  return newVector;
}

static void distributeLayoutConflicts(vector::ShapeCastOp op,
                                      DenseMap<Value, Layout> &layoutMap,
                                      DenseMap<Value, Value> &simdToSimtMap,
                                      IRRewriter &rewriter,
                                      llvm::SetVector<Operation *> &ops) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Value source = op.getSource();
  if (!layoutMap.count(source) || !simdToSimtMap.count(source)) return;
  Layout currentLayout = layoutMap.at(source);
  Value result = op.getResult();
  if (!layoutMap.count(result)) return;
  Layout targetLayout = layoutMap.at(result);

  Value resolvedResult = simdToSimtMap.at(source);
  // For row and col of the vector, resolve layout differences
  for (int i = 0; i < currentLayout.order.size(); i++) {
    if (!resolvedResult) return;
    // Check which dimension(s) are mismatched.
    SmallVector<int> mismatchedDims;
    for (auto dimType : currentLayout.order[i]) {
      if (currentLayout.shape[dimType] != targetLayout.shape[dimType]) {
        mismatchedDims.push_back(dimType);
      }
    }
    if (mismatchedDims.empty()) continue;
    // If any of the mismatched dims are laneId, this layout conflict cannot be
    // resolved.
    if (llvm::any_of(mismatchedDims,
                     [](int dimType) { return isLaneId(dimType); }))
      return;
    // Pure vector conflicts can be resolved, but not supported yet
    if (llvm::all_of(mismatchedDims,
                     [](int dimType) { return isVectorId(dimType); }))
      return;
    if (llvm::all_of(mismatchedDims,
                     [](int dimType) { return isBatchId(dimType); })) {
      resolvedResult =
          resolveBatchConflict(mismatchedDims, resolvedResult, targetLayout,
                               currentLayout, rewriter, op->getLoc());
      continue;
    }
    resolvedResult =
        resolveBatchVectorConflict(mismatchedDims, resolvedResult, targetLayout,
                                   currentLayout, rewriter, op->getLoc());
  }

  simdToSimtMap.try_emplace(result, resolvedResult);
  ops.insert(op);
}

static void eraseOps(llvm::SetVector<Operation *> &opsToErase,
                     IRRewriter &rewriter) {
  for (int i = opsToErase.size() - 1; i >= 0; i--) {
    assert(opsToErase[i]->getUses().empty());
    rewriter.eraseOp(opsToErase[i]);
  }
}

static void collectOperations(Operation *rootOp,
                              SmallVectorImpl<Operation *> &opsToTraverse) {
  for (Region &region : rootOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        opsToTraverse.push_back(&op);
        collectOperations(&op, opsToTraverse);
      }
    }
  }
}

}  // namespace

static bool isMatmulTransposeB(vector::ContractionOp contractOp) {
  // Set up the parallel/reduction structure in right form.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(contractOp.getContext(), m, n, k);
  auto iteratorTypes = contractOp.getIteratorTypes().getValue();
  if (!(vector::isParallelIterator(iteratorTypes[0]) &&
        vector::isParallelIterator(iteratorTypes[1]) &&
        vector::isReductionIterator(iteratorTypes[2])))
    return false;
  SmallVector<AffineMap, 4> maps = contractOp.getIndexingMapsArray();
  return maps == infer({{m, k}, {n, k}, {m, n}});
}

void doLayoutAnalysisAndDistribution(IRRewriter &rewriter,
                                     func::FuncOp funcOp) {
  // First walk through all the MMA ops and set their layouts
  DenseMap<Value, Layout> layoutMap;
  funcOp.walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      if (!isMatmulTransposeB(contractOp)) return WalkResult::advance();
      Value lhs = contractOp.getLhs();
      Value rhs = contractOp.getRhs();
      Value acc = contractOp.getAcc();
      Value result = contractOp.getResult();
      setMMALayout(lhs, rhs, acc, result, layoutMap, op, rewriter);
    }
    return WalkResult::advance();
  });

  // Next propagate the MMA layouts
  funcOp.walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op))
      return WalkResult::advance();
    propagateLayout(op, layoutMap);
    return WalkResult::advance();
  });

  // Apply SIMD to SIMT conversion
  DenseMap<Value, Value> simdToSimtMap;
  llvm::SetVector<Operation *> opsToErase;
  SmallVector<Operation *> opsToTraverse;
  collectOperations(funcOp, opsToTraverse);

  for (Operation *op : opsToTraverse) {
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      distributeTransferReads(readOp, layoutMap, simdToSimtMap, rewriter,
                              opsToErase);
    }
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      distributeContracts(contractOp, layoutMap, simdToSimtMap, rewriter,
                          opsToErase);
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      distributeTransferWrites(writeOp, layoutMap, simdToSimtMap, rewriter,
                               opsToErase);
    }
    if (auto reductionOp = dyn_cast<vector::MultiDimReductionOp>(op)) {
      auto [broadcastOp, transposeOp] =
          checkForReduceBroadcastTranspose(reductionOp);
      distributeReductionBroadcastTranspose(
          reductionOp, broadcastOp, transposeOp, layoutMap, simdToSimtMap,
          rewriter, opsToErase);
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      distributeFor(forOp, layoutMap, simdToSimtMap, rewriter, opsToErase);
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      distributeYield(yieldOp, layoutMap, simdToSimtMap, rewriter, opsToErase);
    }
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      distributeConstants(constantOp, layoutMap, simdToSimtMap, rewriter,
                          opsToErase);
    }
    if (auto conflictOp = dyn_cast<vector::ShapeCastOp>(op)) {
      distributeLayoutConflicts(conflictOp, layoutMap, simdToSimtMap, rewriter,
                                opsToErase);
    }
    distributeElementwise(op, layoutMap, simdToSimtMap, rewriter, opsToErase);
  }

  // Erase old ops
  eraseOps(opsToErase, rewriter);
}

}  // namespace mlir::iree_compiler
