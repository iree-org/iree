// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-rocdl-load-to-transpose-load"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLLOADTOTRANSPOSELOADPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

constexpr int64_t kTransposeLoadLaneGroupSize = 16;
constexpr amdgpu::Chipset kGfx950 = amdgpu::Chipset(9, 5, 0);
constexpr llvm::StringLiteral kPassLocalHintAttr = "__pass_local_hint";

//===----------------------------------------------------------------------===//
// Validation Helpers
//===----------------------------------------------------------------------===//

/// Returns true if the memref is either directly the result of a memref.alloc,
/// or if it is produced by a chain of ops that maintain the full view of the
/// memref. Currently, the only ops that are supported are memref.expand_shape
/// and memref.collapse_shape.
static bool isFullAllocationView(Value memref) {
  Operation *definingOp = memref.getDefiningOp();
  if (!definingOp) {
    return false;
  }

  // Use backwards slice to collect the full producer chain of the memref.
  SetVector<Operation *> slice;
  BackwardSliceOptions opts;
  opts.inclusive = true;
  opts.filter = [](Operation *op) {
    // We only care about memref values, since we want to find the chain of
    // views that produce the memref.
    return llvm::any_of(op->getResultTypes(),
                        [](Type t) { return isa<MemRefType>(t); });
  };
  LogicalResult result = getBackwardSlice(definingOp, &slice, opts);
  if (failed(result)) {
    return false;
  }

  // We must also find the source allocation, or else the memref may be coming
  // from a block argument.
  bool foundAlloc = false;
  for (Operation *op : slice) {
    if (isa<memref::AllocOp>(op)) {
      foundAlloc = true;
      continue;
    }
    if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(op)) {
      continue;
    }
    return false;
  }
  return foundAlloc;
}

/// Returns the required vector size for transpose_load given an element type.
/// Returns std::nullopt if the element type is not supported.
/// TODO(Max191): Add and test 4-bit and 6-bit element type support.
static std::optional<int64_t> getTransposeLoadVectorSize(Type elementType) {
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  switch (bitWidth) {
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
  assert(dim >= 0 && dim < memrefType.getRank() &&
         "dim must be a valid memref dimension");

  // getStridesAndOffset handles all standard layouts including identity.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memrefType.getStridesAndOffset(strides, offset))) {
    // For layouts where strides cannot be computed, be conservative.
    return false;
  }

  return strides[dim] == 1;
}

/// Checks if a row index value is uniform across lanes in a 16-lane group.
/// A row index is uniform if it comes only from:
/// - Constants
/// - index_hint ops with lane_constant attribute (uniform across lanes)
/// - Arithmetic/affine operations on these values
///
/// This function assumes that hint propagation has already been done, so all
/// relevant thread_id ops should already have index_hint users with appropriate
/// lane attributes.
static bool isRowIndexUniform(Value rowIndex) {
  LDBG() << "isRowIndexUniform: checking value: " << rowIndex << "\n";
  // Verify that the rowIndex is some function of constants and hint ops.
  // Use SetVector as worklist to avoid revisiting the same value.
  llvm::SetVector<Value> worklist;
  worklist.insert(rowIndex);
  while (!worklist.empty()) {
    Value currentVal = worklist.pop_back_val();
    LDBG() << "  checking worklist item: " << currentVal << "\n";
    auto opResult = dyn_cast<OpResult>(currentVal);
    if (!opResult) {
      LDBG() << "  FAIL: value is not an OpResult (likely a block argument)\n";
      return false;
    }
    Operation *definingOp = opResult.getOwner();
    LDBG() << "  defining op: " << *definingOp << "\n";
    if (matchPattern(definingOp, m_Constant())) {
      LDBG() << "  OK: is a constant\n";
      continue;
    }
    // Check for index_hint op with lane attributes.
    if (auto indexHintOp = dyn_cast<IREE::Codegen::IndexHintOp>(definingOp)) {
      Attribute hint = indexHintOp.getHint();
      if (auto laneConstant = dyn_cast<IREE::GPU::LaneConstantAttr>(hint)) {
        // lane_constant means uniform across lanes within group_size lanes.
        // For transpose_load, we need uniformity within 16-lane groups.
        // The group_size must be a multiple of 16 for proper alignment.
        if (laneConstant.getGroupSize() % kTransposeLoadLaneGroupSize == 0) {
          LDBG() << "  OK: index_hint with lane_constant, group_size="
                 << laneConstant.getGroupSize() << " is multiple of 16\n";
          continue;
        }
        LDBG() << "  FAIL: index_hint with lane_constant, group_size="
               << laneConstant.getGroupSize() << " is not a multiple of 16\n";
        return false;
      }
      if (isa<IREE::GPU::LaneIncrementAttr>(hint)) {
        // lane_increment means varying across lanes - NOT OK for row index.
        LDBG() << "  FAIL: index_hint with lane_increment attribute (column "
                  "index, not uniform)\n";
        return false;
      }
      // Unknown hint type - be conservative and fail.
      LDBG() << "  FAIL: index_hint with unknown hint attribute type\n";
      return false;
    }
    // Use a whitelist of arith/affine ops to be conservative for non-leaf ops.
    // Other ops may be valid, but they are not common, and allowing arbitrary
    // ops is unsafe when the semantics of the ops are unknown.
    if (!isa<arith::ArithDialect, affine::AffineDialect>(
            definingOp->getDialect())) {
      LDBG() << "  FAIL: op is not from arith or affine dialect (dialect: "
             << definingOp->getDialect()->getNamespace() << ")\n";
      return false;
    }
    LDBG() << "  OK: arith/affine op, adding " << definingOp->getNumOperands()
           << " operands to worklist\n";
    for (Value operand : definingOp->getOperands()) {
      worklist.insert(operand);
    }
  }
  LDBG() << "isRowIndexUniform: SUCCESS - all values uniform\n";
  return true;
}

/// Analysis result for a transfer_read that can be transformed.
struct TransposeLoadAnalysis {
  int64_t columnMemrefDim; // Which memref dimension is the column
  SmallVector<int64_t>
      rowMemrefDims;             // Memref dims for rows (ordered by vector dim)
  SmallVector<int64_t> rowSizes; // Vector sizes for row dims (same order)
  int64_t totalRowSize;          // Product of rowSizes
  int64_t intrinsicVectorSize;   // Required size per transpose_load
  int64_t unrollCount;           // Number of transpose_loads needed
};

/// Analyzes a transfer_read to determine if it can be lowered to
/// transpose_load. Returns analysis result if the transfer_read is suitable,
/// std::nullopt otherwise.
///
/// Requirements:
/// - Source must be workgroup (LDS) memory
/// - Innermost vector dimension must have size 1 (the column dimension)
/// - Column index must come from an index_hint with lane_increment attribute
/// - Column memref dimension must be contiguous (stride 1)
/// - All row indices must be uniform across lanes (derived from constants or
///   index_hint ops with lane_constant attribute)
static std::optional<TransposeLoadAnalysis>
analyzeTransferReadForTransposeLoad(vector::TransferReadOp transferOp) {
  VectorType vecType = transferOp.getVectorType();

  // Vector must have at least 1 dimension.
  if (vecType.getRank() < 1) {
    return std::nullopt;
  }

  // Innermost vector dimension must have size 1 (this is the column).
  if (vecType.getDimSize(vecType.getRank() - 1) != 1) {
    return std::nullopt;
  }

  AffineMap permMap = transferOp.getPermutationMap();

  // Permutation map must have same number of results as vector rank
  // and must be a projected permutation.
  if (permMap.getNumResults() != static_cast<unsigned>(vecType.getRank()) ||
      !permMap.isProjectedPermutation()) {
    return std::nullopt;
  }

  // Column dimension is the last result in the permutation map
  // (corresponds to innermost vector dimension).
  int64_t columnMemrefDim =
      getMemrefDimFromMapResult(permMap, vecType.getRank() - 1);
  Value columnIndex = transferOp.getIndices()[columnMemrefDim];

  // Check if the column index comes from an index_hint with lane_increment.
  auto opResult = dyn_cast<OpResult>(columnIndex);
  if (!opResult) {
    return std::nullopt;
  }
  auto indexHintOp = dyn_cast<IREE::Codegen::IndexHintOp>(opResult.getOwner());
  if (!indexHintOp) {
    return std::nullopt;
  }

  // Validate the lane_increment attribute:
  // - Must be lane_increment (not lane_constant)
  // - group_size must be >= 16 (transpose_load operates on 16-lane groups)
  // - step must be 1 (column indices must be consecutive)
  auto laneIncrement =
      dyn_cast<IREE::GPU::LaneIncrementAttr>(indexHintOp.getHint());
  if (!laneIncrement) {
    return std::nullopt;
  }
  // The group_size must be a multiple of 16 for proper alignment.
  if (laneIncrement.getGroupSize() % kTransposeLoadLaneGroupSize != 0) {
    LDBG() << "Column index lane_increment group_size "
           << laneIncrement.getGroupSize() << " is not a multiple of "
           << kTransposeLoadLaneGroupSize << "\n";
    return std::nullopt;
  }
  if (laneIncrement.getStep() != 1) {
    LDBG() << "Column index lane_increment step " << laneIncrement.getStep()
           << " != 1\n";
    return std::nullopt;
  }

  TransposeLoadAnalysis analysis;
  analysis.columnMemrefDim = columnMemrefDim;

  // 1. Validate element type and get intrinsic size.
  Type elementType = vecType.getElementType();
  std::optional<int64_t> intrinsicSize =
      getTransposeLoadVectorSize(elementType);
  if (!intrinsicSize) {
    LDBG() << "Unsupported element type\n";
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
    LDBG() << "Total row size " << analysis.totalRowSize
           << " is not a multiple of intrinsic size "
           << analysis.intrinsicVectorSize << "\n";
    return std::nullopt;
  }

  analysis.unrollCount = analysis.totalRowSize / analysis.intrinsicVectorSize;

  // 4. Validate column dimension is contiguous (stride 1).
  MemRefType memrefType = cast<MemRefType>(transferOp.getBase().getType());
  if (!isDimensionContiguous(memrefType, analysis.columnMemrefDim)) {
    LDBG() << "Column dimension is not contiguous\n";
    return std::nullopt;
  }

  // 5. Validate all row indices are uniform across lanes.
  for (int64_t memrefDim : analysis.rowMemrefDims) {
    Value rowIndex = transferOp.getIndices()[memrefDim];
    if (!isRowIndexUniform(rowIndex)) {
      LDBG() << "Row index for memref dim " << memrefDim
             << " is not uniform "
                "across lanes\n";
      return std::nullopt;
    }
  }

  return analysis;
}

//===----------------------------------------------------------------------===//
// Transformation Logic
//===----------------------------------------------------------------------===//

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
    RewriterBase &rewriter, Location loc) {
  OperandRange originalIndices = transferOp.getIndices();
  int64_t intrinsicSize = analysis.intrinsicVectorSize;

  // Compute linear element index for this unroll iteration
  // linearElemIdx = unrollIndex * intrinsicSize + rowGroupIdx
  Value cUnrollBase = arith::ConstantIndexOp::create(
      rewriter, loc, unrollIndex * intrinsicSize);
  Value linearElemIdx =
      arith::AddIOp::create(rewriter, loc, cUnrollBase, rowGroupIdx);

  // Delinearize to get indices for each row dimension
  SmallVector<Value> rowIndices;
  assert(!analysis.rowSizes.empty() && "expected at least one row dim");
  auto delinOp = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, linearElemIdx, analysis.rowSizes,
      /*hasOuterBound=*/true);
  rowIndices.assign(delinOp.getResults().begin(), delinOp.getResults().end());

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
                                int64_t intrinsicSize, RewriterBase &rewriter,
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
                                    RewriterBase &rewriter) {
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

//===----------------------------------------------------------------------===//
// Preprocessing: Seed and Propagate Lane Hints
//===----------------------------------------------------------------------===//

/// Injects lane hints on gpu.thread_id ops based on workgroup size.
///
/// The hints describe how thread IDs vary across consecutive lanes (threads
/// with consecutive linear IDs within a workgroup):
/// - thread_id x: lane_increment<wgSizeX, 1> - increments by 1, wraps at
/// wgSizeX
/// - thread_id y: lane_constant<wgSizeX> - constant within groups of wgSizeX
/// lanes
/// - thread_id z: lane_constant<wgSizeX*wgSizeY> - constant within groups of
///   wgSizeX*wgSizeY lanes
///
/// For transpose_load optimization, row indices must be uniform within 16-lane
/// groups. This is satisfied when the group_size >= 16 (i.e., wgSizeX >= 16
/// for thread_id y, or wgSizeX*wgSizeY >= 16 for thread_id z).
///
/// Note: getWorkgroupSize() typically returns a 3-element vector for GPU
/// workgroups. This function only uses elements 0 (x) and 1 (y).
static void seedThreadIdHints(FunctionOpInterface funcOp, IRRewriter &rewriter,
                              ArrayRef<int64_t> workgroupSize) {
  assert(workgroupSize.size() >= 2 &&
         "workgroupSize must have at least 2 elements (x, y)");
  int64_t wgSizeX = workgroupSize[0];
  int64_t wgSizeY = workgroupSize[1];

  // Collect ops first to avoid modifying IR during walk.
  SmallVector<gpu::ThreadIdOp> threadIdOps;
  funcOp.walk([&](gpu::ThreadIdOp threadIdOp) {
    // Skip if already has a hint user.
    for (Operation *user : threadIdOp.getResult().getUsers()) {
      if (isa<IREE::Codegen::IndexHintOp>(user)) {
        return;
      }
    }
    threadIdOps.push_back(threadIdOp);
  });

  for (gpu::ThreadIdOp threadIdOp : threadIdOps) {
    Value result = threadIdOp.getResult();
    rewriter.setInsertionPointAfter(threadIdOp);

    Attribute hintAttr;
    switch (threadIdOp.getDimension()) {
    case gpu::Dimension::x:
      // x varies across lanes. Thread IDs start at 0 which is always aligned.
      hintAttr = IREE::GPU::LaneIncrementAttr::get(
          rewriter.getContext(), wgSizeX, /*step=*/1, /*aligned=*/true);
      break;
    case gpu::Dimension::y:
      // y is constant within groups of wgSizeX consecutive lanes.
      hintAttr =
          IREE::GPU::LaneConstantAttr::get(rewriter.getContext(), wgSizeX);
      break;
    case gpu::Dimension::z:
      // z is constant within groups of wgSizeX*wgSizeY consecutive lanes.
      hintAttr = IREE::GPU::LaneConstantAttr::get(rewriter.getContext(),
                                                  wgSizeX * wgSizeY);
      break;
    }
    assert(hintAttr && "all gpu::Dimension cases should set hintAttr");

    auto hintOp = IREE::Codegen::IndexHintOp::create(
        rewriter, threadIdOp.getLoc(), result, hintAttr);
    hintOp->setAttr(kPassLocalHintAttr, rewriter.getUnitAttr());
    result.replaceAllUsesExcept(hintOp.getResult(), hintOp);
  }
}

//===----------------------------------------------------------------------===//
// Hint Propagation Patterns
//===----------------------------------------------------------------------===//

/// Propagates lane hints through affine.delinearize_index operations.
///
/// For an input with lane_increment<N>:
///   - Results 0 to K-2 get lane_constant<product of bases after that position>
///   - Result K-1 (last) gets lane_increment<innermost_basis>
///
/// For an input with lane_constant<N>:
///   - All results get lane_constant<N> (constant propagates through)
///
/// The pattern fails if any result already has an index_hint user, ensuring
/// hints are only propagated once.
struct PropagateHintThroughDelinearize final
    : OpRewritePattern<affine::AffineDelinearizeIndexOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(affine::AffineDelinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    auto hintOp =
        op.getLinearIndex().getDefiningOp<IREE::Codegen::IndexHintOp>();
    if (!hintOp) {
      return failure();
    }

    // Check for existing hints to provide a stop condition for the greedy
    // driver and avoid infinite loops from re-propagating hints.
    for (Value result : op.getResults()) {
      for (Operation *user : result.getUsers()) {
        if (isa<IREE::Codegen::IndexHintOp>(user)) {
          return failure();
        }
      }
    }

    ArrayRef<int64_t> basis = op.getStaticBasis();
    Attribute inputHint = hintOp.getHint();
    rewriter.setInsertionPointAfter(op);

    return llvm::TypeSwitch<Attribute, LogicalResult>(inputHint)
        // Case 1: lane_constant propagates to all results unchanged.
        // Dynamic basis is allowed since we don't need basis values.
        .Case([&](IREE::GPU::LaneConstantAttr laneConstant) {
          for (Value result : op.getResults()) {
            if (result.use_empty()) {
              continue;
            }
            auto newHint = IREE::Codegen::IndexHintOp::create(
                rewriter, op.getLoc(), result, laneConstant);
            newHint->setAttr(kPassLocalHintAttr, rewriter.getUnitAttr());
            result.replaceAllUsesExcept(newHint.getResult(), newHint);
          }
          return success();
        })
        // Case 2: lane_increment splits across results based on basis.
        // We process from innermost to outermost, computing group sizes from
        // static bases. Dynamic bases are pessimized to 1 (their minimum
        // value).
        .Case([&](IREE::GPU::LaneIncrementAttr laneIncrement) {
          if (basis.empty()) {
            return failure();
          }

          // Innermost basis must be static for lane_increment propagation.
          if (ShapedType::isDynamic(basis.back())) {
            return failure();
          }

          // If the input is not aligned, we cannot safely propagate the hint
          // through delinearize. The modulo operation in delinearize can cause
          // wrap-around within a lane group if the base is not aligned.
          if (!laneIncrement.getAligned()) {
            return failure();
          }

          int64_t originalGroupSize = laneIncrement.getGroupSize();

          // Compute group sizes from innermost to outermost.
          // groupSizes[i] = product of static bases from position i+1 to end.
          //
          // When originalGroupSize is not divisible by currentGroupSize, we
          // inherit the group size from the next inner result. This works
          // because for the outer dim to increment, the inner dim must have
          // incremented, which means it changed by a multiple of its group
          // size.
          //
          // When a dynamic dimension is hit, the currentGroupSize is
          // invalidated, so the remaining results inherit the last valid group
          // size.
          SmallVector<int64_t> groupSizes(basis.size());
          int64_t currentGroupSize = 1;
          int64_t lastValidGroupSize = 1;
          for (int64_t i = basis.size() - 1; i >= 0; --i) {
            if (originalGroupSize % currentGroupSize == 0) {
              groupSizes[i] = currentGroupSize;
              lastValidGroupSize = currentGroupSize;
            } else {
              // Not evenly divisible - inherit from inner result.
              groupSizes[i] = lastValidGroupSize;
            }
            // Only grow currentGroupSize for static bases.
            if (!ShapedType::isDynamic(basis[i])) {
              currentGroupSize *= basis[i];
            }
          }

          for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
            Value result = op.getResult(i);
            if (result.use_empty()) {
              continue;
            }

            Attribute hintAttr;
            if (i == op.getNumResults() - 1) {
              // Last result gets lane_increment with innermost basis.
              // Since the input was aligned and delinearize starts from 0,
              // the innermost result is also aligned.
              hintAttr = IREE::GPU::LaneIncrementAttr::get(
                  getContext(), basis.back(), /*step=*/1, /*aligned=*/true);
            } else {
              // Other results get lane_constant with computed group size.
              hintAttr =
                  IREE::GPU::LaneConstantAttr::get(getContext(), groupSizes[i]);
            }

            auto newHint = IREE::Codegen::IndexHintOp::create(
                rewriter, op.getLoc(), result, hintAttr);
            newHint->setAttr(kPassLocalHintAttr, rewriter.getUnitAttr());
            result.replaceAllUsesExcept(newHint.getResult(), newHint);
          }
          return success();
        })
        .Default([](auto) { return failure(); });
  }
};

//===----------------------------------------------------------------------===//
// Transfer Read to Transpose Load Pattern
//===----------------------------------------------------------------------===//

/// Converts vector.transfer_read operations to amdgpu.transpose_load when
/// profitable.
///
/// This pattern is designed to run in the same greedy driver as hint
/// propagation patterns, allowing hints to propagate incrementally and
/// transpose_load lowering to fire as soon as hints are available.
///
/// The pattern requires hints to already be present on the column index (via
/// seedThreadIdHints + PropagateHintThroughDelinearize). The greedy driver
/// iterates until fixpoint, so hints will propagate through all delinearize
/// ops before this pattern can match.
struct TransferReadToTransposeLoad final
    : OpRewritePattern<vector::TransferReadOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::TransferReadOp transferOp,
                                PatternRewriter &rewriter) const override {
    LDBG() << "TransferReadToTransposeLoad: checking " << transferOp << "\n";

    // Validate memory space.
    auto memrefType = cast<MemRefType>(transferOp.getBase().getType());
    if (!hasSharedMemoryAddressSpace(memrefType)) {
      return rewriter.notifyMatchFailure(transferOp,
                                         "source is not workgroup memory");
    }

    // Subviews of the shared memory allocation are not allowed, because they
    // may indirectly introduce indexing that will not be analyzed by this pass.
    // It is possible to extend the analysis and transformation to handle
    // subviews, but it adds additional complexity, and it is not necessary for
    // the cases we see in IREE.
    if (!isFullAllocationView(transferOp.getBase())) {
      return rewriter.notifyMatchFailure(
          transferOp,
          "transfer_read is not reading from a full view of the allocation");
    }

    // Analyze and validate access pattern.
    std::optional<TransposeLoadAnalysis> analysis =
        analyzeTransferReadForTransposeLoad(transferOp);
    if (!analysis) {
      return rewriter.notifyMatchFailure(transferOp,
                                         "access pattern analysis failed");
    }

    LDBG() << "  -> Transforming to transpose_load (unroll="
           << analysis->unrollCount << ")\n";

    // Generate transpose_load ops.
    rewriter.setInsertionPoint(transferOp);
    Value result = generateTransposeLoads(transferOp, *analysis, rewriter);

    // Replace the original transfer_read.
    rewriter.replaceOp(transferOp, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Pass to lower vector.transfer_read operations to amdgpu.transpose_load
/// operations when profitable, based on iree_codegen.index_hint annotations
/// with lane_constant and lane_increment attributes.
///
/// The pass operates in two phases:
/// 1. Seeding: Inject lane hints on gpu.thread_id ops based on workgroup size
/// 2. Pattern-based transformation: Run hint propagation and transpose_load
///    lowering patterns together in a single greedy driver
struct ROCDLLoadToTransposeLoadPass final
    : impl::ROCDLLoadToTransposeLoadPassBase<ROCDLLoadToTransposeLoadPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Check if target supports transpose_load (currently gfx950 only)
    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target) {
      return;
    }
    FailureOr<amdgpu::Chipset> chipset =
        amdgpu::Chipset::parse(target.getArch());
    if (failed(chipset) || *chipset != kGfx950) {
      return;
    }

    IRRewriter rewriter(funcOp.getContext());

    // Phase 1: Seed hints on gpu.thread_id ops
    std::optional<SmallVector<int64_t>> workgroupSize =
        getWorkgroupSize(funcOp);
    if (workgroupSize) {
      seedThreadIdHints(funcOp, rewriter, *workgroupSize);
    }

    // Phase 2: Propagate hints and lower transfer_reads via greedy patterns
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<PropagateHintThroughDelinearize, TransferReadToTransposeLoad>(
        funcOp.getContext());

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Phase 3: Remove pass-local index_hint ops for IR cleanliness
    funcOp.walk([&](IREE::Codegen::IndexHintOp hintOp) {
      if (hintOp->hasAttr(kPassLocalHintAttr)) {
        rewriter.replaceAllUsesWith(hintOp.getResult(), hintOp.getOperand());
        rewriter.eraseOp(hintOp);
      }
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler
