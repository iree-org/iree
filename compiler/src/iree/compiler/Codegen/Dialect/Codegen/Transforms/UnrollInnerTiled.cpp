// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Codegen {

// Tag constants for the HoistableConversionOp pair created by the unroll
// pattern. Inverse conversions sink/hoist out of the surrounding loop and
// the surviving round-trips fold to nothing.
static constexpr llvm::StringLiteral kUnrollAccDistribute =
    "unroll_acc_distribute";
static constexpr llvm::StringLiteral kUnrollAccReassemble =
    "unroll_acc_reassemble";

//===----------------------------------------------------------------------===//
// InnerTiledOp Unrolling
//===----------------------------------------------------------------------===//

static SmallVector<int64_t>
getUnrollOrder(unsigned numLoops, Operation *op,
               const vector::UnrollVectorOptions &options) {
  SmallVector<int64_t> loopOrder =
      llvm::to_vector(llvm::seq<int64_t>(0, static_cast<int64_t>(numLoops)));
  if (options.traversalOrderCallback != nullptr) {
    std::optional<SmallVector<int64_t>> order =
        options.traversalOrderCallback(op);
    if (order) {
      loopOrder = std::move(*order);
    }
  }
  return loopOrder;
}

namespace {

/// Helper structure to track partially accumulated values while unrolling.
struct OffsetMapInfo {
  static SmallVector<int64_t> getEmptyKey() { return {int64_t(-1)}; }

  static SmallVector<int64_t> getTombstoneKey() { return {int64_t(-2)}; }

  static unsigned getHashValue(const SmallVector<int64_t> &v) {
    return static_cast<unsigned>(llvm::hash_combine_range(v.begin(), v.end()));
  }

  static bool isEqual(const SmallVector<int64_t> &lhs,
                      const SmallVector<int64_t> &rhs) {
    return lhs == rhs;
  }
};

struct UnrollInnerTiledPattern final
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  UnrollInnerTiledPattern(MLIRContext *context,
                          const vector::UnrollVectorOptions &options,
                          PatternBenefit benefit = 1)
      : OpRewritePattern<IREE::Codegen::InnerTiledOp>(context, benefit),
        options(options) {}

  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          tiledOp, "don't know how to unroll multiple accumulators yet");
    }
    if (options.filterConstraint && failed(options.filterConstraint(tiledOp))) {
      return rewriter.notifyMatchFailure(tiledOp, "unrolling filter");
    }
    assert(options.nativeShape &&
           "vector unrolling expects the native shape or native shape call "
           "back function to be set");
    std::optional<SmallVector<int64_t, 4>> maybeUnrollShape =
        tiledOp.getShapeForUnroll();
    if (!maybeUnrollShape) {
      return rewriter.notifyMatchFailure(
          tiledOp, "unexpected failure to get unroll shape");
    }

    std::optional<SmallVector<int64_t>> targetShape =
        options.nativeShape(tiledOp);
    if (!targetShape) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "unspecified native unroll shape");
    }

    auto maybeShapeRatio = computeShapeRatio(*maybeUnrollShape, *targetShape);
    if (!maybeShapeRatio) {
      return rewriter.notifyMatchFailure(
          tiledOp, "operation unroll shape not divisible by target shape");
    }

    // Early exit if unrolling has no effect.
    if (llvm::all_of(*maybeShapeRatio, [](int64_t v) { return v == 1; })) {
      return rewriter.notifyMatchFailure(
          tiledOp, "operation already unrolled to native shape");
    }

    auto dstVecType = cast<VectorType>(tiledOp.getResultTypes().front());
    SmallVector<int64_t, 4> originalSize = *maybeUnrollShape;

    Location loc = tiledOp.getLoc();
    llvm::MapVector<
        SmallVector<int64_t>, Value,
        llvm::DenseMap<SmallVector<int64_t>, unsigned, OffsetMapInfo>>
        accCache;

    SmallVector<int64_t> loopOrder =
        getUnrollOrder(tiledOp.getIteratorTypes().size(), tiledOp, options);

    SmallVector<AffineMap> permutationMaps = tiledOp.getIndexingMapsArray();
    int64_t accIndex = permutationMaps.size() - 1;
    AffineMap accPermutationMap = permutationMaps.back();
    ArrayRef<int64_t> innerAccShape = tiledOp.getOperandInnerShape(accIndex);

    SmallVector<int64_t> accTileShape =
        applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(*targetShape));

    SmallVector<SmallVector<int64_t>> uniqueAccOffsets;
    for (SmallVector<int64_t> off :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      SmallVector<int64_t> accOff =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(off));
      if (!llvm::is_contained(uniqueAccOffsets, accOff)) {
        uniqueAccOffsets.push_back(accOff);
      }
    }

    Value accOperand = tiledOp.getOutputs().front();
    // Distribute the accumulator into per-intrinsic slices; the reassembly
    // conversion will be hoisted out of the reduction loop.
    auto distributeOp = IREE::Util::HoistableConversionOp::create(
        rewriter, loc, /*tag=*/kUnrollAccDistribute,
        /*inverseTag=*/kUnrollAccReassemble, accOperand,
        [&](OpBuilder &b, Location bLoc, ValueRange args) {
          SmallVector<Value> results;
          for (auto &accOff : uniqueAccOffsets) {
            SmallVector<int64_t> strides(accOff.size(), 1);
            results.push_back(vector::ExtractStridedSliceOp::create(
                b, bLoc, args[0], accOff, accTileShape, strides));
          }
          return results;
        });
    for (auto [idx, accOff] : llvm::enumerate(uniqueAccOffsets)) {
      accCache[accOff] = distributeOp.getResult(idx);
    }

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(originalSize, *targetShape, loopOrder)) {
      SmallVector<Value> slicesOperands(tiledOp.getNumOperands());

      auto extractOperand = [&](unsigned index, Value operand,
                                AffineMap permutationMap,
                                ArrayRef<int64_t> operandOffsets) {
        SmallVector<int64_t> operandShape = applyPermutationMap(
            permutationMap, ArrayRef<int64_t>(*targetShape));
        SmallVector<int64_t> operandStrides(operandOffsets.size(), 1);
        slicesOperands[index] = vector::ExtractStridedSliceOp::create(
            rewriter, loc, operand, operandOffsets, operandShape,
            operandStrides);
      };
      for (auto [inputIndex, input] : llvm::enumerate(tiledOp.getInputs())) {
        SmallVector<int64_t> inOffsets = applyPermutationMap(
            permutationMaps[inputIndex], ArrayRef<int64_t>(offsets));
        extractOperand(inputIndex, input, permutationMaps[inputIndex],
                       inOffsets);
      }

      SmallVector<int64_t> accOffsets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      slicesOperands[accIndex] = accCache[accOffsets];

      SmallVector<int64_t> dstShape = applyPermutationMap(
          accPermutationMap, ArrayRef<int64_t>(*targetShape));
      dstShape.append(innerAccShape.begin(), innerAccShape.end());
      auto targetType = VectorType::get(dstShape, dstVecType.getElementType());

      IREE::Codegen::InnerTiledOp newOp =
          mlir::clone(rewriter, tiledOp, targetType, slicesOperands);

      SmallVector<int64_t> dstOffsets =
          applyPermutationMap(accPermutationMap, ArrayRef<int64_t>(offsets));
      accCache[dstOffsets] = newOp.getResults().front();
    }

    SmallVector<Value> accResults;
    SmallVector<SmallVector<int64_t>> accResultOffsets;
    for (const auto &[offsets, partialResult] : accCache) {
      accResults.push_back(partialResult);
      accResultOffsets.push_back(SmallVector<int64_t>(offsets));
    }

    Value result =
        IREE::Util::HoistableConversionOp::create(
            rewriter, loc, /*tag=*/kUnrollAccReassemble,
            /*inverseTag=*/kUnrollAccDistribute, accResults,
            [&](OpBuilder &b, Location bLoc, ValueRange args) {
              Value res =
                  arith::ConstantOp::create(b, bLoc, b.getZeroAttr(dstVecType));
              for (auto [idx, offsets] : llvm::enumerate(accResultOffsets)) {
                SmallVector<int64_t> dstStrides(
                    offsets.size() + innerAccShape.size(), 1);
                SmallVector<int64_t> fullOffsets(offsets);
                fullOffsets.append(innerAccShape.size(), 0);
                res = vector::InsertStridedSliceOp::create(
                    b, bLoc, args[idx], res, fullOffsets, dstStrides);
              }
              return SmallVector<Value>{res};
            })
            .getResult(0);
    rewriter.replaceOp(tiledOp, result);
    return success();
  }

private:
  vector::UnrollVectorOptions options;
};
} // namespace

void populateUnrollInnerTiledPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options) {
  patterns.add<UnrollInnerTiledPattern>(patterns.getContext(), options);
}

static bool isReductionIterator(Attribute attr) {
  return cast<linalg::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::reduction;
}
static bool isParallelIterator(Attribute attr) {
  return cast<linalg::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::parallel;
}

/// Pick an unrolling order that reuses the LHS register, assuming that the LHS
/// register is the first argument.
static std::optional<SmallVector<int64_t>>
matmulLikeUnrollOrder(Operation *op) {
  IREE::Codegen::InnerTiledOp tiledOp =
      dyn_cast<IREE::Codegen::InnerTiledOp>(op);
  if (!tiledOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(tiledOp.getIteratorTypes())) {
    if (isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dimsInLhs;
  for (AffineExpr expr : tiledOp.getIndexingMapsArray()[0].getResults()) {
    dimsInLhs.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(tiledOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && dimsInLhs.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(tiledOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && !dimsInLhs.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

static std::optional<SmallVector<int64_t>>
getInnerTiledUnitShape(Operation *op) {
  auto tiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op);
  if (!tiledOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> targetOuterShape(tiledOp.getIteratorTypes().size(), 1);
  return targetOuterShape;
}

void populateUnrollInnerTiledPatterns(RewritePatternSet &patterns) {
  populateUnrollInnerTiledPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getInnerTiledUnitShape)
                    .setUnrollTraversalOrderFn(matmulLikeUnrollOrder));
}

} // namespace mlir::iree_compiler::IREE::Codegen
