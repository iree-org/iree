// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-rocdl-load-to-transpose-load"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLLOADTOTRANSPOSELOADPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

constexpr int64_t kTransposeLoadLaneGroupSize = 16;

//===----------------------------------------------------------------------===//
// Validation Helpers
//===----------------------------------------------------------------------===//

/// Returns true if the given memory space is workgroup (LDS) memory.
static bool isWorkgroupMemory(MemRefType memrefType) {
  Attribute memSpace = memrefType.getMemorySpace();
  if (!memSpace) {
    return false;
  }
  if (auto intMemSpace = dyn_cast<IntegerAttr>(memSpace)) {
    return intMemSpace.getInt() == 3;
  }
  if (auto gpuMemSpace = dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
    return gpuMemSpace.getValue() == gpu::AddressSpace::Workgroup;
  }
  return false;
}

/// Returns the required vector size for transpose_load given an element type.
/// Returns std::nullopt if the element type is not supported.
static std::optional<int64_t> getTransposeLoadVectorSize(Type elementType) {
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 4:
    return 16;
  case 6:
    return 16;
  case 8:
    return 8;
  case 16:
    return 4;
  default:
    return std::nullopt;
  }
}

/// Given a permutation map, find which memref dimension corresponds to the
/// specified result index in the permutation map.
/// For example, for affine_map<(d0, d1) -> (d0, d1)> with resultIdx=1,
/// returns 1 (d1's position).
/// Asserts that the permutation map is valid and resultIdx is in bounds.
static int64_t getMemrefDimFromMapResult(AffineMap permMap,
                                         unsigned resultIdx) {
  assert(resultIdx < permMap.getNumResults() &&
         "resultIdx out of bounds for permutation map");

  AffineExpr expr = permMap.getResult(resultIdx);
  auto dimExpr = dyn_cast<AffineDimExpr>(expr);
  assert(dimExpr && "permutation map result is not a dimension expression");

  return dimExpr.getPosition();
}

/// Check if the specified memref dimension has stride 1 (is contiguous).
static bool isDimensionContiguous(MemRefType memrefType, int64_t dim) {
  // For default identity layout (row-major), the last dimension has stride 1
  MemRefLayoutAttrInterface layout = memrefType.getLayout();

  if (layout.isIdentity()) {
    return dim == memrefType.getRank() - 1;
  }

  // Handle strided layout
  if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(layout)) {
    ArrayRef<int64_t> strides = stridedLayout.getStrides();
    if (dim < 0 || dim >= static_cast<int64_t>(strides.size())) {
      return false;
    }
    return strides[dim] == 1;
  }

  // For unknown layouts, be conservative
  return false;
}

/// Checks if a row index value is uniform across lanes in a 16-lane group.
/// A row index is uniform if it comes only from:
/// - Constants
/// - index_hint ops with lane_constant attribute (uniform across lanes)
/// - Thread IDs that are constant across lane groups.
/// - Arithmetic/affine operations on these values
static bool isRowIndexUniform(
    Value rowIndex,
    std::optional<SmallVector<int64_t>> workgroupSize = std::nullopt) {
  LLVM_DEBUG(llvm::dbgs() << "isRowIndexUniform: checking value: " << rowIndex
                          << "\n");
  // Verify that the rowIndex is some function of constants and hint ops.
  SmallVector<Value> worklist = {rowIndex};
  while (!worklist.empty()) {
    Value currentVal = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs()
               << "  checking worklist item: " << currentVal << "\n");
    auto opResult = dyn_cast<OpResult>(currentVal);
    if (!opResult) {
      LLVM_DEBUG(llvm::dbgs() << "  FAIL: value is not an OpResult (likely a "
                                 "block argument)\n");
      return false;
    }
    Operation *definingOp = opResult.getOwner();
    LLVM_DEBUG(llvm::dbgs() << "  defining op: " << *definingOp << "\n");
    if (matchPattern(definingOp, m_Constant())) {
      LLVM_DEBUG(llvm::dbgs() << "  OK: is a constant\n");
      continue;
    }
    // Check for new-style index_hint op with lane attributes
    if (auto indexHintOp = dyn_cast<IREE::Codegen::IndexHintOp>(definingOp)) {
      Attribute hint = indexHintOp.getHint();
      if (isa<IREE::GPU::LaneConstantAttr>(hint)) {
        // lane_constant means uniform across lanes - this is OK for row index
        LLVM_DEBUG(llvm::dbgs()
                   << "  OK: index_hint with lane_constant attribute\n");
        continue;
      }
      if (isa<IREE::GPU::LaneIncrementAttr>(hint)) {
        // lane_increment means varying across lanes - NOT OK for row index
        LLVM_DEBUG(llvm::dbgs()
                   << "  FAIL: index_hint with lane_increment attribute "
                      "(column index, not uniform)\n");
        return false;
      }
      // Unknown hint type - be conservative and fail
      LLVM_DEBUG(llvm::dbgs()
                 << "  FAIL: index_hint with unknown hint attribute type\n");
      return false;
    }
    // gpu.thread_id ops are okay as long as the ID is uniform along lanes. For
    // example, `gpu.thread_id y` could be uniform along lanes if the workgroup
    // size along `x` is a multiple of the lane group size.
    if (auto threadIdOp = dyn_cast<gpu::ThreadIdOp>(definingOp)) {
      // Without the workgroup or subgroup size, we cannot prove anything about
      // the thread IDs.
      if (!workgroupSize.has_value()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  FAIL: thread_id op but no workgroup size available\n");
        return false;
      }
      int64_t xWgSize = (*workgroupSize)[0];
      int64_t yWgSize = (*workgroupSize)[1];
      int64_t zWgSize = (*workgroupSize)[2];
      LLVM_DEBUG(llvm::dbgs()
                 << "  thread_id op with workgroup size: [" << xWgSize << ", "
                 << yWgSize << ", " << zWgSize << "]\n");
      switch (threadIdOp.getDimension()) {
      case gpu::Dimension::x:
        if (xWgSize == 1) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  OK: thread_id x with xWgSize=1 (uniform)\n");
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  FAIL: thread_id x with xWgSize=" << xWgSize
                   << " (not uniform, varies across lanes)\n");
        return false;
      case gpu::Dimension::y:
        if (yWgSize == 1 || xWgSize % kTransposeLoadLaneGroupSize == 0) {
          LLVM_DEBUG(llvm::dbgs() << "  OK: thread_id y (yWgSize=" << yWgSize
                                  << ", xWgSize=" << xWgSize << " divisible by "
                                  << kTransposeLoadLaneGroupSize << ")\n");
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  FAIL: thread_id y with yWgSize=" << yWgSize
                   << ", xWgSize=" << xWgSize << " not divisible by "
                   << kTransposeLoadLaneGroupSize << "\n");
        return false;
      case gpu::Dimension::z:
        if (zWgSize == 1 ||
            (xWgSize * yWgSize) % kTransposeLoadLaneGroupSize == 0) {
          LLVM_DEBUG(llvm::dbgs() << "  OK: thread_id z (zWgSize=" << zWgSize
                                  << ", xWgSize*yWgSize=" << (xWgSize * yWgSize)
                                  << " divisible by "
                                  << kTransposeLoadLaneGroupSize << ")\n");
          continue;
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  FAIL: thread_id z with zWgSize=" << zWgSize
                   << ", xWgSize*yWgSize=" << (xWgSize * yWgSize)
                   << " not divisible by " << kTransposeLoadLaneGroupSize
                   << "\n");
        return false;
      }
    }
    // Use a whitelist of arith/affine ops to be conservative for non-leaf ops.
    // Other ops may be valid, but they are not common, and allowing arbitrary
    // ops is unsafe when the semantics of the ops are unknown.
    if (!isa<arith::ArithDialect, affine::AffineDialect>(
            definingOp->getDialect())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  FAIL: op is not from arith or affine dialect (dialect: "
                 << definingOp->getDialect()->getNamespace() << ")\n");
      return false;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  OK: arith/affine op, adding "
               << definingOp->getNumOperands() << " operands to worklist\n");
    for (Value operand : definingOp->getOperands()) {
      worklist.push_back(operand);
    }
  }
  LLVM_DEBUG(
      llvm::dbgs() << "isRowIndexUniform: SUCCESS - all values uniform\n");
  return true;
}

/// Analysis result for a transfer_read that can be transformed.
struct TransposeLoadAnalysis {
  // The column hint op
  IREE::Codegen::IndexHintOp columnHintOp;
  int64_t columnMemrefDim; // Which memref dimension is the column
  SmallVector<int64_t>
      rowMemrefDims;             // Memref dims for rows (ordered by vector dim)
  SmallVector<int64_t> rowSizes; // Vector sizes for row dims (same order)
  int64_t totalRowSize;          // Product of rowSizes
  int64_t intrinsicVectorSize;   // Required size per transpose_load
  int64_t unrollCount;           // Number of transpose_loads needed
};

/// Information about the column hint found in a transfer_read.
struct ColumnHintInfo {
  // index_hint op with lane_increment attribute
  IREE::Codegen::IndexHintOp columnHintOp;
  int64_t columnMemrefDim; // Which memref dim uses the column hint
};

/// Finds if the transfer_read's innermost vector dimension (which must have
/// size 1) uses an index_hint op with lane_increment attribute.
/// Returns the hint op and column memref dimension if found.
static std::optional<ColumnHintInfo>
findColumnHintInfo(vector::TransferReadOp transferOp) {
  VectorType vecType = transferOp.getVectorType();

  // Vector must have at least 1 dimension
  if (vecType.getRank() < 1) {
    return std::nullopt;
  }

  // Innermost vector dimension must have size 1 (this is the column)
  if (vecType.getDimSize(vecType.getRank() - 1) != 1) {
    return std::nullopt;
  }

  AffineMap permMap = transferOp.getPermutationMap();

  // Permutation map must have same number of results as vector rank
  // and must be a projected permutation
  if (permMap.getNumResults() != static_cast<unsigned>(vecType.getRank()) ||
      !permMap.isProjectedPermutation()) {
    return std::nullopt;
  }

  // Column dimension is the last result in the permutation map
  // (corresponds to innermost vector dimension)
  int64_t columnMemrefDim =
      getMemrefDimFromMapResult(permMap, vecType.getRank() - 1);
  Value columnIndex = transferOp.getIndices()[columnMemrefDim];

  // Check if this index comes from a hint op
  auto opResult = dyn_cast<OpResult>(columnIndex);
  if (!opResult) {
    return std::nullopt;
  }

  // Check for index_hint with lane_increment attribute
  if (auto indexHintOp =
          dyn_cast<IREE::Codegen::IndexHintOp>(opResult.getOwner())) {
    if (isa<IREE::GPU::LaneIncrementAttr>(indexHintOp.getHint())) {
      return ColumnHintInfo{indexHintOp, columnMemrefDim};
    }
    // index_hint without lane_increment is not a column hint
    return std::nullopt;
  }

  return std::nullopt;
}

/// Validates that the transfer_read meets all transpose_load requirements.
/// Note: This function is only called after findColumnHintInfo has already
/// validated that the innermost vector dim is 1 and found the column memref
/// dim.
static std::optional<TransposeLoadAnalysis> analyzeTransferReadForTransposeLoad(
    vector::TransferReadOp transferOp, const ColumnHintInfo &columnInfo,
    std::optional<SmallVector<int64_t>> workgroupSize = std::nullopt) {
  TransposeLoadAnalysis analysis;
  analysis.columnHintOp = columnInfo.columnHintOp;
  analysis.columnMemrefDim = columnInfo.columnMemrefDim;

  VectorType vecType = transferOp.getVectorType();
  AffineMap permMap = transferOp.getPermutationMap();

  // These should already be validated by findColumnHintInfo
  assert(vecType.getRank() >= 1 && "Expected at least 1D vector");
  assert(vecType.getDimSize(vecType.getRank() - 1) == 1 &&
         "Expected innermost dimension size 1");
  assert(permMap.isProjectedPermutation() && "Expected valid permutation map");

  // 1. Validate element type and get intrinsic size
  Type elementType = vecType.getElementType();
  std::optional<int64_t> intrinsicSize =
      getTransposeLoadVectorSize(elementType);
  if (!intrinsicSize) {
    LLVM_DEBUG(llvm::dbgs() << "Unsupported element type\n");
    return std::nullopt;
  }
  analysis.intrinsicVectorSize = *intrinsicSize;

  // 2. Collect row dimensions and compute total row size.
  // Row dimensions are all vector dimensions except the innermost (column).
  // We iterate in vector dimension order to maintain correct linearization.
  analysis.totalRowSize = 1;
  for (int64_t vecDim = 0; vecDim < vecType.getRank() - 1; ++vecDim) {
    int64_t size = vecType.getDimSize(vecDim);
    int64_t memrefDim = getMemrefDimFromMapResult(permMap, vecDim);

    // Only include dimensions with size > 1 in row computation.
    // Dimensions with size 1 don't contribute to the row product.
    if (size > 1) {
      analysis.rowMemrefDims.push_back(memrefDim);
      analysis.rowSizes.push_back(size);
      analysis.totalRowSize *= size;
    }
  }

  // 3. Validate total row size is a multiple of intrinsic size.
  if (analysis.totalRowSize % analysis.intrinsicVectorSize != 0) {
    LLVM_DEBUG(llvm::dbgs() << "Total row size " << analysis.totalRowSize
                            << " is not a multiple of intrinsic size "
                            << analysis.intrinsicVectorSize << "\n");
    return std::nullopt;
  }

  analysis.unrollCount = analysis.totalRowSize / analysis.intrinsicVectorSize;

  // 4. Validate column dimension is contiguous (stride 1)
  MemRefType memrefType = cast<MemRefType>(transferOp.getBase().getType());
  if (!isDimensionContiguous(memrefType, analysis.columnMemrefDim)) {
    LLVM_DEBUG(llvm::dbgs() << "Column dimension is not contiguous\n");
    return std::nullopt;
  }

  // 5. Validate all row indices are uniform across lanes
  for (int64_t memrefDim : analysis.rowMemrefDims) {
    Value rowIndex = transferOp.getIndices()[memrefDim];
    if (!isRowIndexUniform(rowIndex, workgroupSize)) {
      LLVM_DEBUG(llvm::dbgs() << "Row index for memref dim " << memrefDim
                              << " is not uniform across lanes\n");
      return std::nullopt;
    }
  }

  return analysis;
}

//===----------------------------------------------------------------------===//
// Transformation Logic
//===----------------------------------------------------------------------===//

/// Delinearize a linear element index into N-D indices.
/// Given sizes [S_0, S_1, ..., S_k] and a linear index, returns indices
/// [Idx_0, Idx_1, ..., Idx_k] such that:
///   linearIdx = Idx_0 * (S_1 * S_2 * ... * S_k) + Idx_1 * (S_2 * ... * S_k) +
///   ... + Idx_k
static SmallVector<Value> delinearizeIndex(Value linearIdx,
                                           ArrayRef<int64_t> sizes,
                                           IRRewriter &rewriter, Location loc) {
  if (sizes.empty()) {
    return {};
  }

  SmallVector<Value> indices(sizes.size());
  Value remaining = linearIdx;

  // Process from innermost to outermost dimension
  for (int64_t i = sizes.size() - 1; i >= 0; --i) {
    Value size = arith::ConstantIndexOp::create(rewriter, loc, sizes[i]);
    indices[i] = arith::RemUIOp::create(rewriter, loc, remaining, size);
    if (i > 0) {
      remaining = arith::DivUIOp::create(rewriter, loc, remaining, size);
    }
  }

  return indices;
}

/// Computes all memref indices for a single transpose_load op during unrolling.
///
/// The transformation converts implicit vector element indices into explicit
/// offset additions. For each unroll iteration:
///   linearElemIdx = unrollIndex * intrinsicSize + rowGroupIdx
///   delinearize to get {Idx_0, Idx_1, ...} for row dimensions
///   newOffset_i = originalOffset_i + Idx_i
///
/// Column index uses lane-based remapping (unchanged from 2D case).
///
/// Returns all memref indices for the transpose_load.
static SmallVector<Value> computeTransposeLoadIndices(
    vector::TransferReadOp transferOp, const TransposeLoadAnalysis &analysis,
    int64_t unrollIndex, Value rowGroupIdx, Value newColIdx,
    IRRewriter &rewriter, Location loc) {
  OperandRange originalIndices = transferOp.getIndices();
  int64_t intrinsicSize = analysis.intrinsicVectorSize;

  // Compute linear element index for this unroll iteration
  // linearElemIdx = unrollIndex * intrinsicSize + rowGroupIdx
  Value cUnrollBase = arith::ConstantIndexOp::create(
      rewriter, loc, unrollIndex * intrinsicSize);
  Value linearElemIdx =
      arith::AddIOp::create(rewriter, loc, cUnrollBase, rowGroupIdx);

  // Delinearize to get indices for each row dimension
  SmallVector<Value> rowIndices =
      delinearizeIndex(linearElemIdx, analysis.rowSizes, rewriter, loc);

  // Build the full index list for the memref
  // Start with original indices, then update row and column dimensions
  SmallVector<Value> newIndices(originalIndices.begin(), originalIndices.end());

  // Update row dimension indices: newOffset = originalOffset + delinearizedIdx
  for (auto [i, memrefDim] : llvm::enumerate(analysis.rowMemrefDims)) {
    Value originalIdx = originalIndices[memrefDim];
    newIndices[memrefDim] =
        arith::AddIOp::create(rewriter, loc, originalIdx, rowIndices[i]);
  }

  // Update column dimension index
  newIndices[analysis.columnMemrefDim] = newColIdx;

  return newIndices;
}

/// Computes the column index for transpose_load using lane-based remapping.
/// This is shared across all unroll iterations.
static Value computeColumnIndex(Value originalColIdx, Value laneInGroup,
                                int64_t intrinsicSize, IRRewriter &rewriter,
                                Location loc) {
  int64_t loadsPerColumn = kTransposeLoadLaneGroupSize / intrinsicSize;

  // Column index remapping:
  // 1. Compute baseColOffset = originalColIdx - laneInGroup
  Value baseColOffset =
      arith::SubIOp::create(rewriter, loc, originalColIdx, laneInGroup);

  // 2. Compute colGroupIdx = laneInGroup % loadsPerColumn
  Value cLoadsPerCol =
      arith::ConstantIndexOp::create(rewriter, loc, loadsPerColumn);
  Value colGroupIdx =
      arith::RemUIOp::create(rewriter, loc, laneInGroup, cLoadsPerCol);

  // 3. Compute offset within column group = colGroupIdx * intrinsicSize
  Value cIntrinsicSize =
      arith::ConstantIndexOp::create(rewriter, loc, intrinsicSize);
  Value colOffset =
      arith::MulIOp::create(rewriter, loc, colGroupIdx, cIntrinsicSize);

  // 4. newColIdx = baseColOffset + colOffset
  return arith::AddIOp::create(rewriter, loc, baseColOffset, colOffset);
}

/// Generates the transpose_load ops and combines results into a 1D vector,
/// then reshapes to the original N-D vector shape.
static Value generateTransposeLoads(vector::TransferReadOp transferOp,
                                    const TransposeLoadAnalysis &analysis,
                                    IRRewriter &rewriter) {
  Location loc = transferOp.getLoc();
  VectorType resultType = transferOp.getVectorType();
  Type elementType = resultType.getElementType();
  int64_t intrinsicSize = analysis.intrinsicVectorSize;
  int64_t unrollCount = analysis.unrollCount;
  int64_t totalRowSize = analysis.totalRowSize;
  Value source = transferOp.getBase();

  // The intrinsic produces a 1D vector
  VectorType intrinsicVecType = VectorType::get({intrinsicSize}, elementType);

  // Compute lane ID and derived values (shared across all unrolls)
  Value laneId = gpu::LaneIdOp::create(rewriter, loc, /*upper_bound=*/nullptr);
  Value cLaneGroupSize = arith::ConstantIndexOp::create(
      rewriter, loc, kTransposeLoadLaneGroupSize);
  Value laneInGroup =
      arith::RemUIOp::create(rewriter, loc, laneId, cLaneGroupSize);

  // Compute rowGroupIdx = laneInGroup / loadsPerColumn (shared across unrolls)
  int64_t loadsPerColumn = kTransposeLoadLaneGroupSize / intrinsicSize;
  Value cLoadsPerCol =
      arith::ConstantIndexOp::create(rewriter, loc, loadsPerColumn);
  Value rowGroupIdx =
      arith::DivUIOp::create(rewriter, loc, laneInGroup, cLoadsPerCol);

  // Compute column index (shared across all unrolls)
  Value originalColIdx = transferOp.getIndices()[analysis.columnMemrefDim];
  Value newColIdx = computeColumnIndex(originalColIdx, laneInGroup,
                                       intrinsicSize, rewriter, loc);

  // Generate transpose_load ops for each unroll iteration
  SmallVector<Value> results;
  for (int64_t i = 0; i < unrollCount; ++i) {
    // Compute all memref indices for this transpose_load
    SmallVector<Value> indices = computeTransposeLoadIndices(
        transferOp, analysis, i, rowGroupIdx, newColIdx, rewriter, loc);

    // Create transpose_load op with all indices
    auto transposeLoadOp = amdgpu::TransposeLoadOp::create(
        rewriter, loc, intrinsicVecType, source, indices);

    results.push_back(transposeLoadOp.getResult());
  }

  // Combine results into a 1D vector
  VectorType flat1DType = VectorType::get({totalRowSize}, elementType);

  Value combined;
  if (results.size() == 1) {
    // Single result - just use it directly (already 1D)
    combined = results[0];
  } else {
    // Multiple results - concatenate into 1D vector
    combined = arith::ConstantOp::create(rewriter, loc, flat1DType,
                                         rewriter.getZeroAttr(flat1DType));

    for (auto [idx, result] : llvm::enumerate(results)) {
      // Insert at offset [idx * intrinsicSize]
      SmallVector<int64_t> offsets = {static_cast<int64_t>(idx) *
                                      intrinsicSize};
      SmallVector<int64_t> strides = {1};
      combined = vector::InsertStridedSliceOp::create(
          rewriter, loc, result, combined, offsets, strides);
    }
  }

  // Reshape from 1D to original N-D shape
  return vector::ShapeCastOp::create(rewriter, loc, resultType, combined);
}

/// Attempts to lower a vector.transfer_read into amdgpu.transpose_load.
/// If the transformation is not possible or not profitable, does nothing.
static void lowerTransferReadToTransposeLoad(
    vector::TransferReadOp transferOp, IRRewriter &rewriter,
    std::optional<SmallVector<int64_t>> workgroupSize = std::nullopt) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing transfer_read for transpose_load: "
                          << transferOp << "\n");

  // Step 1: Check if column index comes from a hint
  std::optional<ColumnHintInfo> columnInfo = findColumnHintInfo(transferOp);
  if (!columnInfo) {
    LLVM_DEBUG(llvm::dbgs() << "  -> Column index not from hint\n");
    return;
  }

  // Step 2: Validate memory space
  auto memrefType = cast<MemRefType>(transferOp.getBase().getType());
  if (!isWorkgroupMemory(memrefType)) {
    LLVM_DEBUG(llvm::dbgs() << "  -> Source is not workgroup memory\n");
    return;
  }

  // Step 3: Analyze and validate access pattern
  std::optional<TransposeLoadAnalysis> analysisResult =
      analyzeTransferReadForTransposeLoad(transferOp, *columnInfo,
                                          workgroupSize);
  if (!analysisResult) {
    LLVM_DEBUG(llvm::dbgs() << "  -> Access pattern analysis failed\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "  -> Transforming to transpose_load (unroll="
                          << analysisResult->unrollCount << ")\n");

  // Step 4: Generate transpose_load ops
  Value result = generateTransposeLoads(transferOp, *analysisResult, rewriter);

  // Step 5: Replace the original transfer_read
  rewriter.replaceOp(transferOp, result);
}

//===----------------------------------------------------------------------===//
// Preprocessing: Inject hints on thread_id -> delinearize_index sequences
//===----------------------------------------------------------------------===//

/// Injects index_hint ops on the results of affine.delinearize_index operations
/// that are fed directly by gpu.thread_id operations.
///
/// For a sequence like:
///   %tid = gpu.thread_id x
///   %results:N = affine.delinearize_index %tid into (b0, b1, ..., bN-1)
///
/// This function adds hints:
/// - Outer results (0 to N-2) get lane_constant<product of inner bases>
/// - Final result (N-1) gets lane_increment<innermost basis>
///
/// For example, with `affine.delinearize_index %tid into (4, 16)`:
/// - Result #0 gets lane_constant<16>
/// - Result #1 gets lane_increment<16>
static void injectDelinearizeIndexHints(FunctionOpInterface funcOp,
                                        IRRewriter &rewriter) {
  SmallVector<affine::AffineDelinearizeIndexOp> delinearizeOps;
  funcOp.walk([&](affine::AffineDelinearizeIndexOp op) {
    delinearizeOps.push_back(op);
  });

  for (auto delinearizeOp : delinearizeOps) {
    // Check if the input is a gpu.thread_id operation
    Value input = delinearizeOp.getLinearIndex();
    auto threadIdOp = input.getDefiningOp<gpu::ThreadIdOp>();
    if (!threadIdOp) {
      continue;
    }

    // Get the static basis values
    ArrayRef<int64_t> staticBasis = delinearizeOp.getStaticBasis();
    if (staticBasis.empty()) {
      // No basis, skip
      continue;
    }

    // Check if any basis element is dynamic (kDynamic)
    bool hasDynamicBasis = llvm::any_of(
        staticBasis, [](int64_t v) { return ShapedType::isDynamic(v); });
    if (hasDynamicBasis) {
      // Non-constant basis, skip
      continue;
    }

    SmallVector<int64_t> basis(staticBasis);
    unsigned numResults = delinearizeOp.getNumResults();

    // Need at least 2 results to apply hints
    if (numResults < 2) {
      continue;
    }

    // Check if hints are already applied to results
    bool alreadyHinted = false;
    for (Value result : delinearizeOp.getResults()) {
      for (Operation *user : result.getUsers()) {
        if (auto hintOp = dyn_cast<IREE::Codegen::IndexHintOp>(user)) {
          if (isa<IREE::GPU::LaneConstantAttr, IREE::GPU::LaneIncrementAttr>(
                  hintOp.getHint())) {
            alreadyHinted = true;
            break;
          }
        }
      }
      if (alreadyHinted)
        break;
    }
    if (alreadyHinted) {
      continue;
    }

    // Calculate group sizes for each result
    // For basis (b0, b1, ..., bN-1):
    // - Result i gets lane_constant<product of bases from i+1 to N-1>
    // - Result N-1 gets lane_increment<bN-1>
    rewriter.setInsertionPointAfter(delinearizeOp);

    for (unsigned i = 0; i < numResults; ++i) {
      Value result = delinearizeOp.getResult(i);

      // Skip if no users
      if (result.use_empty()) {
        continue;
      }

      // Calculate the group size (product of inner bases)
      int64_t groupSize = 1;
      for (unsigned j = i + 1; j < basis.size(); ++j) {
        groupSize *= basis[j];
      }

      // For the last result, use lane_increment; for others, use lane_constant
      Attribute hintAttr;
      if (i == numResults - 1) {
        // Final result gets lane_increment
        hintAttr = IREE::GPU::LaneIncrementAttr::get(rewriter.getContext(),
                                                     basis.back());
      } else {
        // Outer results get lane_constant with group size
        hintAttr =
            IREE::GPU::LaneConstantAttr::get(rewriter.getContext(), groupSize);
      }

      // Create the hint op
      auto hintOp = IREE::Codegen::IndexHintOp::create(
          rewriter, delinearizeOp.getLoc(), result, hintAttr);

      // Replace all uses of the original result with the hint result,
      // except for the hint op itself
      result.replaceAllUsesExcept(hintOp.getResult(), hintOp);
    }
  }
}

/// Pass to lower vector.transfer_read operations to amdgpu.transpose_load
/// operations when profitable, based on iree_codegen.index_hint annotations
/// with lane_constant and lane_increment attributes.
struct ROCDLLoadToTransposeLoadPass final
    : impl::ROCDLLoadToTransposeLoadPassBase<ROCDLLoadToTransposeLoadPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    // Preprocessing: Inject hints on thread_id -> delinearize_index sequences
    injectDelinearizeIndexHints(funcOp, rewriter);

    // Step 1: Collect all vector.transfer_read operations
    SmallVector<vector::TransferReadOp> transferReads;
    funcOp.walk([&](vector::TransferReadOp transferOp) {
      transferReads.push_back(transferOp);
    });

    // Step 2: Attempt to transform each transfer_read
    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    for (auto transferOp : transferReads) {
      rewriter.setInsertionPoint(transferOp);
      lowerTransferReadToTransposeLoad(transferOp, rewriter, workgroupSize);
    }

    // Step 3: Remove all remaining hint operations (both new and legacy)
    // These hints are purely for optimization and safe to remove if not used

    // Remove index_hint operations with lane attributes
    SmallVector<IREE::Codegen::IndexHintOp> indexHintOps;
    funcOp.walk([&](IREE::Codegen::IndexHintOp hintOp) {
      // Only remove hints with lane_constant or lane_increment attributes
      Attribute hint = hintOp.getHint();
      if (isa<IREE::GPU::LaneConstantAttr, IREE::GPU::LaneIncrementAttr>(
              hint)) {
        indexHintOps.push_back(hintOp);
      }
    });

    for (auto hintOp : indexHintOps) {
      // Replace hint result with the original input (pass-through)
      hintOp.getResult().replaceAllUsesWith(hintOp.getInput());
      rewriter.eraseOp(hintOp);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
