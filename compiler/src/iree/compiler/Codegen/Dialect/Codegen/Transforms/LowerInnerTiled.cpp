// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Codegen {

// Tag constants for HoistableConversionOp pairs. Each pair of tags marks
// inverse conversions that can be cancelled or hoisted out of loops.
static constexpr llvm::StringLiteral kShapeCastToIntrinsic =
    "shape_cast_to_intrinsic";
static constexpr llvm::StringLiteral kShapeCastFromIntrinsic =
    "shape_cast_from_intrinsic";
static constexpr llvm::StringLiteral kDropUnitDims = "drop_unit_dims";
static constexpr llvm::StringLiteral kAddUnitDims = "add_unit_dims";

//===----------------------------------------------------------------------===//
// InnerTiledOp lowering to underlying operation
//===----------------------------------------------------------------------===//

namespace {
struct LowerInnerTiledPattern final
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          tiledOp, "lowering to concrete op requires vector semantics");
    }
    SmallVector<int64_t> bounds;
    tiledOp.getIterationBounds(bounds);
    if (!bounds.empty()) {
      return rewriter.notifyMatchFailure(
          tiledOp, "must be a single inner tiled operation");
    }

    SmallVector<Value> operands = tiledOp.getOperands();
    SmallVector<VectorType> regTypes;
    tiledOp.getKind().getDistributedTileTypes(regTypes);

    int64_t numInputs = tiledOp.getNumInputs();

    for (int64_t i = 0; i < numInputs; ++i) {
      if (operands[i].getType() != regTypes[i]) {
        operands[i] = vector::ShapeCastOp::create(rewriter, tiledOp.getLoc(),
                                                  regTypes[i], operands[i]);
      }
    }

    bool needsCast = !llvm::all_of_zip(
        ArrayRef(operands).drop_front(numInputs),
        ArrayRef(regTypes).drop_front(numInputs),
        [](Value v, VectorType t) { return v.getType() == t; });
    if (needsCast) {
      auto outputs = ArrayRef(operands).drop_front(numInputs);
      auto outputRegTypes = ArrayRef(regTypes).drop_front(numInputs);
      // Shape-cast accumulator to intrinsic register types; the inverse
      // conversion after the op will be hoisted out of the reduction loop.
      auto hoistOp = IREE::Util::HoistableConversionOp::create(
          rewriter, tiledOp.getLoc(), /*tag=*/kShapeCastToIntrinsic,
          /*inverseTag=*/kShapeCastFromIntrinsic, outputs,
          [&outputRegTypes](OpBuilder &b, Location loc, ValueRange args) {
            return llvm::map_to_vector(
                llvm::zip_equal(args, outputRegTypes), [&](auto pair) -> Value {
                  auto [arg, regType] = pair;
                  if (arg.getType() == regType) {
                    return arg;
                  }
                  return vector::ShapeCastOp::create(b, loc, regType, arg);
                });
          });

      llvm::copy(hoistOp.getResults(), operands.begin() + numInputs);
    }

    SmallVector<Value> results;
    LogicalResult couldLower = tiledOp.getKind().buildUnderlyingOperations(
        rewriter, tiledOp.getLoc(), ValueRange{operands}.take_front(numInputs),
        ValueRange{operands}.drop_front(numInputs), results);
    if (failed(couldLower)) {
      tiledOp.emitOpError(
          "failed to lower to concrete inner tiled operations.");
      return failure();
    }

    if (needsCast) {
      auto resultTypes = tiledOp.getResultTypes();
      auto hoistOp = IREE::Util::HoistableConversionOp::create(
          rewriter, tiledOp.getLoc(), /*tag=*/kShapeCastFromIntrinsic,
          /*inverseTag=*/kShapeCastToIntrinsic, results,
          [&resultTypes](OpBuilder &b, Location loc, ValueRange args) {
            return llvm::map_to_vector(
                llvm::zip_equal(args, resultTypes), [&](auto pair) -> Value {
                  auto [arg, extType] = pair;
                  if (arg.getType() == extType) {
                    return arg;
                  }
                  return vector::ShapeCastOp::create(b, loc, extType, arg);
                });
          });
      results = SmallVector<Value>(hoistOp.getResults());
    }
    rewriter.replaceOp(tiledOp, results);
    return success();
  }
};
} // namespace

void populateLowerInnerTiledPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerInnerTiledPattern>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// InnerTiledOp Unit Dim Folding
//===----------------------------------------------------------------------===//

namespace {
struct DropInnerTiledUnitDimsPattern final
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (tiledOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          tiledOp, "unimplemented: unit dim dropping for tensor mma ops");
    }
    SmallVector<int64_t> bounds;
    tiledOp.getIterationBounds(bounds);
    if (bounds.empty()) {
      return rewriter.notifyMatchFailure(tiledOp, "no dimensions to fold");
    }

    // TODO: Generalize to allow only some iteration bounds to be unit. This
    // pattern currently only supports the most common case of unrolling to the
    // intrinsic shape.
    if (!llvm::all_of(bounds, [](int64_t b) { return b == 1; })) {
      return rewriter.notifyMatchFailure(tiledOp,
                                         "not all iteration bounds are unit");
    }

    Location loc = tiledOp.getLoc();
    int64_t numInputs = tiledOp.getNumInputs();
    SmallVector<Value> newOperands;
    for (auto [opIndex, operand] : llvm::enumerate(tiledOp.getInputs())) {
      int64_t outerRank = tiledOp.getOperandOuterRank(opIndex);
      if (outerRank == 0) {
        newOperands.push_back(operand);
      } else {
        SmallVector<int64_t> zeros(outerRank, 0);
        newOperands.push_back(
            vector::ExtractOp::create(rewriter, loc, operand, zeros));
      }
    }

    auto outputs = tiledOp.getOutputs();
    // Drop outer unit dims from accumulator for the intrinsic; the inverse
    // broadcast will be hoisted out of the reduction loop.
    auto hoistExtractOp = IREE::Util::HoistableConversionOp::create(
        rewriter, loc, /*tag=*/kDropUnitDims, /*inverseTag=*/kAddUnitDims,
        outputs, [&](OpBuilder &b, Location bLoc, ValueRange args) {
          return llvm::map_to_vector(
              llvm::enumerate(args), [&](auto pair) -> Value {
                auto [i, arg] = pair;
                int64_t outerRank = tiledOp.getOperandOuterRank(numInputs + i);
                if (outerRank == 0) {
                  return arg;
                }
                SmallVector<int64_t> zeros(outerRank, 0);
                return vector::ExtractOp::create(b, bLoc, arg, zeros);
              });
        });
    llvm::append_range(newOperands, hoistExtractOp.getResults());

    SmallVector<AffineMap> emptyMaps(tiledOp.getNumOperands(),
                                     AffineMap::get(rewriter.getContext()));
    auto newTiledOp = IREE::Codegen::InnerTiledOp::create(
        rewriter, loc, ValueRange{newOperands}.take_front(numInputs),
        ValueRange{newOperands}.drop_front(numInputs),
        rewriter.getAffineMapArrayAttr(emptyMaps), rewriter.getArrayAttr({}),
        tiledOp.getKind(), tiledOp.getSemantics());

    // Pull extract / broadcast pairs out of the loop so the hardware
    // accumulator type becomes loop-carried.
    SmallVector<Value> newResults(newTiledOp.getResults());
    auto resultTypes = tiledOp.getResultTypes();
    auto hoistBroadcastOp = IREE::Util::HoistableConversionOp::create(
        rewriter, loc, /*tag=*/kAddUnitDims, /*inverseTag=*/kDropUnitDims,
        newResults,
        [&resultTypes](OpBuilder &b, Location bLoc, ValueRange args) {
          return llvm::map_to_vector(
              llvm::zip_equal(args, resultTypes), [&](auto pair) -> Value {
                auto [arg, ty] = pair;
                if (arg.getType() == ty) {
                  return arg;
                }
                return vector::BroadcastOp::create(b, bLoc, ty, arg);
              });
        });
    rewriter.replaceOp(tiledOp, hoistBroadcastOp.getResults());
    return success();
  }
};
} // namespace

void populateDropInnerTiledUnitDimsPatterns(RewritePatternSet &patterns) {
  patterns.add<DropInnerTiledUnitDimsPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::Codegen
