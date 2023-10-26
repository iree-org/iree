// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-fuse-dequantization-matmul"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

//----------------------------------------------------------------------------//
//                                Utility
//----------------------------------------------------------------------------//

// Creates a new flow.dipatch.region op and places the
// passed ops inside as long as the dequant op is a
// producer for the matmul op
static LogicalResult fuseDequantAndMatmul(RewriterBase &rewriter,
                                          Operation *dequant, Operation *matmul,
                                          std::optional<Operation *> fill) {

  auto regionOp = matmul->getParentOfType<IREE::Flow::DispatchRegionOp>();
  if (!regionOp) {
    FailureOr<IREE::Flow::DispatchRegionOp> maybeRegionOp =
        IREE::Flow::wrapOpInDispatchRegion(rewriter, matmul);
    if (failed(maybeRegionOp))
      return failure();
    regionOp = maybeRegionOp.value();
  }

  FailureOr<IREE::Flow::DispatchRegionOp> maybeFusedRegionOp =
      IREE::Flow::clonePrecedingOpIntoDispatchRegion(rewriter, dequant,
                                                     regionOp);
  if (failed(maybeFusedRegionOp))
    return failure();

  if (fill && *fill) {
    FailureOr<IREE::Flow::DispatchRegionOp> maybeFusedFillRegionOp =
        IREE::Flow::clonePrecedingOpIntoDispatchRegion(rewriter, fill.value(),
                                                       regionOp);
    if (failed(maybeFusedFillRegionOp))
      return failure();
  }

  return success();
}

static FailureOr<IREE::Flow::DispatchRegionOp>
wrapConsecutiveOpsInDispatchRegion(RewriterBase &rewriter,
                                   SmallVector<Operation *> ops) {
  FailureOr<IREE::Flow::DispatchRegionOp> maybeRegionOp =
      IREE::Flow::wrapOpInDispatchRegion(rewriter, ops.back());
  if (failed(maybeRegionOp)) {
    return failure();
  }
  IREE::Flow::DispatchRegionOp regionOp = maybeRegionOp.value();

  SmallVector<Operation *> precedingOps(ops.begin(), ops.end() - 1);
  FailureOr<IREE::Flow::DispatchRegionOp> maybeFusedRegionOp =
      IREE::Flow::movePrecedingOpsIntoDispatchRegion(rewriter, precedingOps,
                                                     regionOp);
  if (failed(maybeFusedRegionOp)) {
    return failure();
  }
  regionOp = maybeFusedRegionOp.value();

  return regionOp;
}

static SmallVector<utils::IteratorType>
getParallelAndReductionIterators(unsigned nLoops, unsigned nReduction) {
  SmallVector<utils::IteratorType> res(nLoops - nReduction,
                                       utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);
  return res;
}

// Takes as input the dequantization `linalg.generic` op and the matmul
// `linalg.generic` op, and returns the scales, zero points, quantized
// input matrix, unquantizaed input matrix, and dequantized result
// matrix.
static std::optional<SmallVector<OpOperand *>>
getDequantMatmulInputs_f32(linalg::GenericOp dequant,
                           linalg::GenericOp matmul) {
  OpOperand *scales, *zps, *quantMat, *unquantMat, *dequantMat;
  for (int operandIdx = 0; operandIdx < dequant.getNumDpsInputs();
       operandIdx++) {
    OpOperand *operand = dequant.getDpsInputOperand(operandIdx);
    Value input = operand->get();
    RankedTensorType inputType =
        llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      continue;
    }
    if (inputType.getElementTypeBitWidth() != 32) {
      quantMat = operand;
      continue;
    }
    for (Operation &bodyOp : dequant.getBlock()->getOperations()) {
      if (isa<arith::MulFOp>(bodyOp)) {
        if (bodyOp.getOperand(1) ==
            dequant.getBlock()->getArgument(operandIdx)) {
          scales = operand;
          break;
        }
      } else if (isa<arith::SubFOp>(bodyOp)) {
        if (bodyOp.getOperand(1) ==
            dequant.getBlock()->getArgument(operandIdx)) {
          zps = operand;
          break;
        }
      }
    }
  }
  Value dequantOut = dequant.getResult(0);
  if (matmul.getDpsInputOperand(0)->get() == dequantOut) {
    unquantMat = matmul.getDpsInputOperand(1);
    dequantMat = matmul.getDpsInputOperand(0);
  } else {
    unquantMat = matmul.getDpsInputOperand(0);
    dequantMat = matmul.getDpsInputOperand(1);
  }
  if (scales && zps && quantMat && unquantMat) {
    return SmallVector<OpOperand *>(
        {quantMat, unquantMat, scales, zps, dequantMat});
  }
  return std::nullopt;
}

// This function does the bulk of the rewrite for the dequantization + matmul.
//
// Starting with 2 `linalg.generic` ops (dequantization->matmul)
// %arg0 = quantized input
// %arg1 = scales
// %arg2 = zero points
// %arg3 = unquantized input
// ```mlir
// %0 = linalg.generic ins(%arg0, %arg1, %arg2 : tensor<8x4x2xi4>,
// tensor<8x4x1xf32>, tensor<8x4x1xf32>) outs(%1 : tensor<8x4x2xf32>) {
// ^bb0(%in: i4, %in_0: f32, %in_1: f32, %out: f32):
//   %9 = arith.extui %in : i4 to i32
//   %10 = arith.uitofp %9 : i32 to f32
//   %11 = arith.subf %10, %in_1 : f32
//   %12 = arith.mulf %11, %in_0 : f32
//   linalg.yield %12 : f32
// } -> tensor<8x4x2xf32>
// %2 = linalg.generic ins(%arg3, %0 : tensor<4x2xf32>, tensor<8x4x2xf32>)
// outs(%3 : tensor<8xf32>) { ^bb0(%in: f32, %in_0: f32, %out: f32):
//   %9 = arith.mulf %in, %in_0 : f32
//   %10 = arith.addf %9, %out : f32
//   linalg.yield %10 : f32
// } -> tensor<8xf32>
// ```
//
// This function rewrites the above ops as the following new sequence of 6 ops
// that does the following:
//
// a) Dynamically quantize the unquantized input:
//      1. Compute the absolute max of the unquantized input (%arg3) within each
//      group.
//      2. Compute scales for %arg3 by dividing the absolute max by (1 <<
//      newBitWidth) - 1),
//         where newBitWidth is the bitwidth of the new quantized type,
//         currently set to `i16`.
//      3. Compute the sum along groups of the unquantized input. This is not
//      necessary for
//         the quantization step, but it is needed to efficiently perform a
//         reassociated quantized matmul in steps 5-6.
//      4. Quantize the unquantized input (%arg3) by dividing elements in each
//      group by
//         the corresponding scale. This quantization is symmetric with no zero
//         point.
// b) Perform the reassociated quantized matmul, keeping the bulk of the
// computation in
//    integer arithmetic:
//      5. The first op performs a matmul-like operation that reduces the
//      innermost group
//         dimension. Note the indexing maps in the following example:
//         ```mlir
//         %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) ->
//         (d1, d2)>,
//                                                affine_map<(d0, d1, d2) ->
//                                                (d0, d1, d2)>, affine_map<(d0,
//                                                d1, d2) -> (d0, d1)>],
//                               iterator_types = ["parallel", "parallel",
//                               "reduction"]} ins(%17, %0 : tensor<4x2xi16>,
//                               tensor<8x4x2xi4>) outs(%19 : tensor<8x4xi32>) {
//             ^bb0(%in: i16, %in_4: i4, %out: i32):
//               %24 = arith.extsi %in : i16 to i32
//               %25 = arith.extui %in_4 : i4 to i32
//               %26 = arith.muli %24, %25 : i32
//               %27 = arith.addi %26, %out : i32
//               linalg.yield %27 : i32
//             } -> tensor<8x4xi32>
//         ```
//         This op also extends the inputs to the accumulation type, i32 in this
//         case, to target specific x86 instructions. We perform the matrix
//         multiplication before the dequantization arithmetic, which has been
//         reassociated into op 6.
//      6. The final op performs the remaining reduction across groups and does
//      the
//         dequantization arithmetic:
//         ```mlir
//         %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0,
//         d1)>,
//                                                affine_map<(d0, d1) -> (d1)>,
//                                                affine_map<(d0, d1) -> (d1)>,
//                                                affine_map<(d0, d1) -> (d0,
//                                                d1)>, affine_map<(d0, d1) ->
//                                                (d0, d1)>, affine_map<(d0, d1)
//                                                -> (d0)>],
//                               iterator_types = ["parallel", "reduction"]}
//                               ins(tensor<8x4xi32>, tensor<4xf32>,
//                               tensor<4xf32>, tensor<8x4xf32>,
//                               tensor<8x4xf32>) outs(tensor<8xf32>) {
//             ^bb0(%in: i32, %in_4: f32, %in_5: f32, %in_6: f32, %in_7: f32,
//             %out: f32):
//               %24 = arith.sitofp %in : i32 to f32
//               %25 = arith.mulf %24, %in_4 : f32
//               %26 = arith.mulf %25, %in_6 : f32
//               %27 = arith.mulf %in_7, %in_6 : f32
//               %28 = arith.mulf %27, %in_5 : f32
//               %29 = arith.subf %26, %28 : f32
//               %30 = arith.addf %29, %out : f32
//               linalg.yield %30 : f32
//             } -> tensor<8xf32>
//         ```
//
// The rewrite also forms a `flow.dispatch.region` op around ops 5 and 6, and
// sets a specific tiling on ops 5 and 6 to target a VectorContractCustomKernel.
//
// ** Note that this rewrite introduces precision loss in the matmul, and is a
//    tradeoff between precision and performance. This rewrite should most
//    likely be opt-in only. **
static LogicalResult reassociateAndFuseDequantMatmul(RewriterBase &rewriter,
                                                     linalg::GenericOp dequant,
                                                     linalg::GenericOp matmul) {
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "dequant:   " << dequant << "\n";
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "matmul:   " << matmul << "\n";
  std::optional<SmallVector<OpOperand *>> maybeInputs =
      getDequantMatmulInputs_f32(dequant, matmul);
  if (!maybeInputs) {
    return failure();
  }
  SmallVector<OpOperand *> ins = maybeInputs.value();
  OpOperand *quantInOperand = ins[0];
  OpOperand *unquantInOperand = ins[1];
  Value quantIn = quantInOperand->get();
  Value unquantIn = unquantInOperand->get();
  Value scales = ins[2]->get();
  Value zps = ins[3]->get();
  OpOperand *matmulDequantOperand = ins[4];
  RankedTensorType unquantInType =
      llvm::dyn_cast<RankedTensorType>(unquantIn.getType());
  if (!unquantInType) {
    return failure();
  }
  RankedTensorType quantInType =
      llvm::dyn_cast<RankedTensorType>(quantIn.getType());
  if (!quantInType) {
    return failure();
  }
  OpOperand *matmulOutputOperand = matmul.getDpsInitOperand(0);
  Value matmulOutput = matmulOutputOperand->get();
  RankedTensorType matmulOutputType =
      llvm::dyn_cast<RankedTensorType>(matmulOutput.getType());
  SmallVector<int64_t> matmulOutShape(matmulOutputType.getShape());
  SmallVector<int64_t> unquantInShape(unquantInType.getShape());
  SmallVector<int64_t> quantInShape(quantInType.getShape());
  SmallVector<AffineMap> dequantIndexingMaps = dequant.getIndexingMapsArray();
  SmallVector<AffineMap> matmulIndexingMaps = matmul.getIndexingMapsArray();
  SmallVector<utils::IteratorType> dequantIteratorTypes =
      dequant.getIteratorTypesArray();
  SmallVector<utils::IteratorType> matmulIteratorTypes =
      matmul.getIteratorTypesArray();
  FloatType f32Type = rewriter.getF32Type();
  IntegerType i32Type = rewriter.getI32Type();
  // Type for accumulation of integer matmul
  IntegerType accType = rewriter.getI32Type();
  // Type for dynamic quantization of unquantized input
  IntegerType quantType = rewriter.getI16Type();
  Type srcQuantType = quantInType.getElementType();
  // Type for multiplication in integer matmul (should probably be same as
  // `accType`)
  IntegerType mulType = rewriter.getI32Type();
  unsigned quantBitRange =
      std::min(quantType.getIntOrFloatBitWidth() - 1,
               mulType.getIntOrFloatBitWidth() -
                   srcQuantType.getIntOrFloatBitWidth() - 1);

  // ----- Quantize unquantized input ----- //
  Value cst = rewriter.create<arith::ConstantOp>(
      dequant.getLoc(), rewriter.getF32FloatAttr((1 << quantBitRange) - 1));
  Value zeroF32cst = rewriter.create<arith::ConstantOp>(
      dequant.getLoc(), rewriter.getF32FloatAttr(0));
  Value zeroI32cst = rewriter.create<arith::ConstantOp>(
      dequant.getLoc(), rewriter.getI32IntegerAttr(0));
  // Generic to find max along groups
  SmallVector<int64_t> groupMaxShape;
  SmallVector<utils::IteratorType> groupMaxIterators;
  int64_t numGroups = 0;

  SmallVector<AffineExpr> exprs;
  AffineMap indexingMap = matmul.getMatchingIndexingMap(unquantInOperand);
  for (const auto &expr : enumerate(indexingMap.getResults())) {
    if (auto dimExpr = expr.value().dyn_cast<AffineDimExpr>()) {
      if (matmulIteratorTypes[dimExpr.getPosition()] ==
              utils::IteratorType::parallel ||
          dimExpr.getPosition() != indexingMap.getNumDims() - 1) {
        groupMaxIterators.push_back(utils::IteratorType::parallel);
        groupMaxShape.push_back(unquantInShape[expr.index()]);
        exprs.push_back(rewriter.getAffineDimExpr(groupMaxShape.size() - 1));
        if (matmulIteratorTypes[dimExpr.getPosition()] ==
            utils::IteratorType::reduction) {
          numGroups = unquantInShape[expr.index()];
        }
      } else {
        groupMaxIterators.push_back(utils::IteratorType::reduction);
      }
    } else {
      return failure();
    }
  }
  if (!numGroups) {
    return failure();
  }
  RankedTensorType groupMaxType =
      RankedTensorType::get(groupMaxShape, unquantInType.getElementType());
  Value groupMaxEmpty = rewriter.create<tensor::EmptyOp>(
      dequant.getLoc(), groupMaxType.getShape(), groupMaxType.getElementType());
  Value groupMaxOut =
      rewriter
          .create<linalg::FillOp>(dequant.getLoc(), zeroF32cst, groupMaxEmpty)
          .result();
  SmallVector<AffineMap> groupMaxMaps;
  groupMaxMaps.push_back(
      rewriter.getMultiDimIdentityMap(unquantInShape.size()));
  groupMaxMaps.push_back(AffineMap::get(unquantInShape.size(), 0, exprs,
                                        exprs.front().getContext()));
  auto groupMaxOp = rewriter.create<linalg::GenericOp>(
      dequant.getLoc(), groupMaxOut.getType(), unquantIn, groupMaxOut,
      groupMaxMaps, groupMaxIterators,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value abs = b.create<math::AbsFOp>(loc, args[0]);
        Value max = b.create<arith::MaximumFOp>(loc, abs, args[1]);
        b.create<linalg::YieldOp>(loc, max);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "groupMaxOp:   " << groupMaxOp << "\n";
  Value groupMax = groupMaxOp.getResult(0);

  // Generic to find scales
  RankedTensorType unquantInScalesType = groupMaxType;
  Value unquantInScalesOut = rewriter.create<tensor::EmptyOp>(
      dequant.getLoc(), unquantInScalesType.getShape(),
      unquantInScalesType.getElementType());
  SmallVector<AffineMap> unquantInScalesMaps;
  unquantInScalesMaps.push_back(
      rewriter.getMultiDimIdentityMap(unquantInShape.size() - 1));
  unquantInScalesMaps.push_back(
      rewriter.getMultiDimIdentityMap(unquantInShape.size() - 1));

  auto unquantInScalesOp = rewriter.create<linalg::GenericOp>(
      dequant.getLoc(), unquantInScalesOut.getType(), groupMax,
      unquantInScalesOut, unquantInScalesMaps,
      getParallelAndReductionIterators(unquantInScalesType.getRank(), 0),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value scale = b.create<arith::DivFOp>(loc, args[0], cst);
        b.create<linalg::YieldOp>(loc, scale);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "unquantInScalesOp:   " << unquantInScalesOp << "\n";
  Value unquantInScales = unquantInScalesOp.getResult(0);

  // Generic to find scaled sums
  RankedTensorType scaledSumsType = groupMaxType;
  Value scaledSumsEmpty = rewriter.create<tensor::EmptyOp>(
      dequant.getLoc(), scaledSumsType.getShape(),
      scaledSumsType.getElementType());
  Value scaledSumsOut =
      rewriter
          .create<linalg::FillOp>(dequant.getLoc(), zeroF32cst, scaledSumsEmpty)
          .result();
  SmallVector<AffineMap> scaledSumsMaps;
  scaledSumsMaps.push_back(
      rewriter.getMultiDimIdentityMap(unquantInShape.size()));
  scaledSumsMaps.push_back(AffineMap::get(unquantInShape.size(), 0, exprs,
                                          exprs.front().getContext()));
  SmallVector<utils::IteratorType> scaledSumsIterators = groupMaxIterators;
  auto scaledSumsOp = rewriter.create<linalg::GenericOp>(
      dequant.getLoc(), scaledSumsOut.getType(), unquantIn, scaledSumsOut,
      scaledSumsMaps, scaledSumsIterators,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, sum);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "scaledSumsOp:   " << scaledSumsOp << "\n";
  Value scaledSums = scaledSumsOp.getResult(0);

  // Generic to quantized the unquantized input
  Value newQuantInOut = rewriter.create<tensor::EmptyOp>(
      dequant.getLoc(), unquantInShape, quantType);
  SmallVector<AffineMap> newQuantInMaps;
  newQuantInMaps.push_back(
      rewriter.getMultiDimIdentityMap(unquantInShape.size()));
  newQuantInMaps.push_back(AffineMap::get(unquantInShape.size(), 0, exprs,
                                          exprs.front().getContext()));
  newQuantInMaps.push_back(
      rewriter.getMultiDimIdentityMap(unquantInShape.size()));
  auto newQuantInOp = rewriter.create<linalg::GenericOp>(
      dequant.getLoc(), newQuantInOut.getType(),
      ValueRange{unquantIn, unquantInScales}, newQuantInOut, newQuantInMaps,
      getParallelAndReductionIterators(unquantInShape.size(), 0),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value scaled = b.create<arith::DivFOp>(loc, args[0], args[1]);
        Value quant = b.create<arith::FPToSIOp>(loc, quantType, scaled);
        b.create<linalg::YieldOp>(loc, quant);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "newQuantInOp:   " << newQuantInOp << "\n";
  Value newQuantIn = newQuantInOp.getResult(0);

  // ----- Reassociated dequantization matmul ----- //

  // Generic to perform integer matmul and reduce within groups
  SmallVector<int64_t> integerMatmulShape = matmulOutShape;
  integerMatmulShape.push_back(numGroups);
  Value integerMatmulEmpty = rewriter.create<tensor::EmptyOp>(
      dequant.getLoc(), integerMatmulShape, accType);
  Value integerMatmulOut =
      rewriter
          .create<linalg::FillOp>(dequant.getLoc(), zeroI32cst,
                                  integerMatmulEmpty)
          .result();
  SmallVector<utils::IteratorType> integerMatmulIterators =
      getParallelAndReductionIterators(matmul.getNumLoops(), 1);
  SmallVector<AffineMap> integerMatmulMaps;
  integerMatmulMaps.push_back(matmul.getMatchingIndexingMap(unquantInOperand));
  integerMatmulMaps.push_back(
      matmul.getMatchingIndexingMap(matmulDequantOperand));
  SmallVector<AffineExpr> outputExprs(
      matmul.getMatchingIndexingMap(matmulOutputOperand).getResults());
  outputExprs.push_back(rewriter.getAffineDimExpr(matmul.getNumLoops() - 2));
  integerMatmulMaps.push_back(AffineMap::get(integerMatmulIterators.size(), 0,
                                             outputExprs,
                                             outputExprs.front().getContext()));
  auto integerMatmulOp = rewriter.create<linalg::GenericOp>(
      dequant.getLoc(), integerMatmulOut.getType(),
      ValueRange{newQuantIn, quantIn}, integerMatmulOut, integerMatmulMaps,
      integerMatmulIterators, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value mul;
        if (quantType == mulType) {
          Value ext1 = b.create<arith::ExtUIOp>(loc, mulType, args[1]);
          mul = b.create<arith::MulIOp>(loc, args[0], ext1);
        } else {
          Value ext0 = b.create<arith::ExtSIOp>(loc, mulType, args[0]);
          Value ext1 = b.create<arith::ExtUIOp>(loc, mulType, args[1]);
          mul = b.create<arith::MulIOp>(loc, ext0, ext1);
        }
        Value sum;
        if (mulType == accType) {
          sum = b.create<arith::AddIOp>(loc, mul, args[2]);
        } else {
          Value extMul = b.create<arith::ExtSIOp>(loc, accType, mul);
          sum = b.create<arith::AddIOp>(loc, extMul, args[2]);
        }
        b.create<linalg::YieldOp>(loc, sum);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "integerMatmulOp:   " << integerMatmulOp << "\n";
  Value integerMatmul = integerMatmulOp.getResult(0);

  // Generic to perform dequantization and finish reduction
  SmallVector<utils::IteratorType> dequantizedMatmulIterators =
      getParallelAndReductionIterators(matmul.getNumLoops() - 1, 1);
  SmallVector<AffineMap> dequantizedMatmulMaps;
  dequantizedMatmulMaps.push_back(
      rewriter.getMultiDimIdentityMap(dequantizedMatmulIterators.size()));
  AffineMap intMatmulNewQuantMap =
      matmul.getMatchingIndexingMap(unquantInOperand);
  SmallVector<AffineExpr> newQuantScalesExprs;
  for (const auto &expr : enumerate(intMatmulNewQuantMap.getResults())) {
    if (auto dimExpr = expr.value().dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() != intMatmulNewQuantMap.getNumDims() - 1) {
        newQuantScalesExprs.push_back(
            rewriter.getAffineDimExpr(dimExpr.getPosition()));
      }
    } else {
      return failure();
    }
  }
  dequantizedMatmulMaps.push_back(
      AffineMap::get(dequantizedMatmulIterators.size(), 0, newQuantScalesExprs,
                     newQuantScalesExprs.front().getContext()));
  dequantizedMatmulMaps.push_back(
      AffineMap::get(dequantizedMatmulIterators.size(), 0, newQuantScalesExprs,
                     newQuantScalesExprs.front().getContext()));
  AffineMap matmulQuantInMap =
      matmul.getMatchingIndexingMap(matmulDequantOperand);
  SmallVector<AffineExpr> quantScalesExprs;
  for (const auto &expr : enumerate(matmulQuantInMap.getResults())) {
    if (auto dimExpr = expr.value().dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() != matmulQuantInMap.getNumDims() - 1) {
        quantScalesExprs.push_back(
            rewriter.getAffineDimExpr(dimExpr.getPosition()));
      }
    } else {
      return failure();
    }
  }
  RankedTensorType scalesType =
      llvm::dyn_cast<RankedTensorType>(scales.getType());
  if (!scalesType) {
    return failure();
  }
  if (quantScalesExprs.size() < scalesType.getShape().size()) {
    quantScalesExprs.push_back(rewriter.getAffineConstantExpr(0));
    if (quantScalesExprs.size() < scalesType.getShape().size()) {
      return failure();
    }
  }
  dequantizedMatmulMaps.push_back(
      AffineMap::get(dequantizedMatmulIterators.size(), 0, quantScalesExprs,
                     quantScalesExprs.front().getContext()));
  dequantizedMatmulMaps.push_back(
      AffineMap::get(dequantizedMatmulIterators.size(), 0, quantScalesExprs,
                     quantScalesExprs.front().getContext()));
  SmallVector<AffineExpr> finalOutputExprs(
      matmul.getMatchingIndexingMap(matmulOutputOperand).getResults());
  dequantizedMatmulMaps.push_back(
      AffineMap::get(dequantizedMatmulIterators.size(), 0, finalOutputExprs,
                     finalOutputExprs.front().getContext()));

  auto dequantizedMatmulOp = rewriter.create<linalg::GenericOp>(
      dequant.getLoc(), matmulOutput.getType(),
      ValueRange{integerMatmul, unquantInScales, scaledSums, scales, zps},
      matmulOutput, dequantizedMatmulMaps, dequantizedMatmulIterators,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value dq;
        if (accType == i32Type) {
          dq = b.create<arith::SIToFPOp>(loc, f32Type, args[0]);
        } else {
          Value ext = b.create<arith::ExtSIOp>(loc, i32Type, args[0]);
          dq = b.create<arith::SIToFPOp>(loc, f32Type, ext);
        }
        Value scaledRes0 = b.create<arith::MulFOp>(loc, dq, args[1]);
        Value scaledRes1 = b.create<arith::MulFOp>(loc, scaledRes0, args[3]);
        Value scaledZp0 = b.create<arith::MulFOp>(loc, args[4], args[3]);
        Value scaledZp1 = b.create<arith::MulFOp>(loc, scaledZp0, args[2]);
        Value groupRes = b.create<arith::SubFOp>(loc, scaledRes1, scaledZp1);
        Value sum = b.create<arith::AddFOp>(loc, groupRes, args[5]);
        b.create<linalg::YieldOp>(loc, sum);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "dequantizedMatmulOp:   " << dequantizedMatmulOp << "\n";
  Value dequantizedMatmul = dequantizedMatmulOp.getResult(0);

  rewriter.replaceOp(matmul, dequantizedMatmul);

  // Fuse dequantization + matmul ops into a single dispatch region
  SmallVector<Operation *> dequantMatmulOps(
      {integerMatmulOp, dequantizedMatmulOp});
  FailureOr<IREE::Flow::DispatchRegionOp> maybeDequantMatmulDispatch =
      wrapConsecutiveOpsInDispatchRegion(rewriter, dequantMatmulOps);
  if (failed(maybeDequantMatmulDispatch)) {
    return failure();
  }

  return success();
}

// Checks if the passed op is a contraction on grouped input
// This function checks that the genericOp:
// 1. isaContractionOpInterface
// 2. Has 2 reduction dimensions (for grouped input)
static LogicalResult isGroupedContractionOp(linalg::GenericOp genericOp) {
  unsigned numLoops = genericOp.getNumLoops();
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(genericOp.getOperation());
  if (numLoops == 0) {
    return failure();
  }
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return failure();
  }
  if (genericOp.getNumReductionLoops() != 2) {
    return failure();
  }
  return success();
}

// Checks if the passed op is a dequantization on grouped input
// This function checks that the genericOp:
// 1. Has a body like:
//      arith.extui
//      arith.uitofp
//      arith.subf
//      arith.mulf
// 2. Increases the bit width of the input
// 3. Has 3 parallel dims
// 4. Has 2 (weights, scales) or 3 (weights, scales, zero points)
//    inputs and 1 output
static LogicalResult isGroupedDequantizationOp(linalg::GenericOp genericOp) {
  // Check for 1 result, and 2 (input, scales) or 3 (input, scales, zero points)
  // inputs
  if (genericOp.getNumDpsInits() != 1) {
    return failure();
  }
  if (genericOp.getNumDpsInputs() != 2 && genericOp.getNumDpsInputs() != 3) {
    return failure();
  }
  // Check that the rank is at least 3 and all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops < 3) {
    return failure();
  }
  if (numLoops != numParallelLoops) {
    return failure();
  }

  // Work back from linalg.yield and check body of genericOp.
  // The genericOp should yield the result of an arith.mulf,
  // preceded by an arith.subf, arith.uitofp, and arith.extui
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;

  // Producer of linalg.yield op is arith.mulf
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
    if (!matchPattern(producer, m_Op<arith::MulFOp>())) {
      return failure();
    }
  }

  // Producer of arith.mulf op is arith.subf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
    if (!matchPattern(producer, m_Op<arith::SubFOp>())) {
      return failure();
    }
  }

  // Producer of arith.subf op is arith.uitofp
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
    if (!matchPattern(producer, m_Op<arith::UIToFPOp>())) {
      return failure();
    }
  }

  // Producer of arith.uitofp op is arith.extui
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer) {
      return failure();
    }
    if (!matchPattern(producer, m_Op<arith::ExtUIOp>())) {
      return failure();
    }
  }

  // Ensure that the dequantization increases the
  // bitwidth from the input to the output
  auto elementTypeOut =
      llvm::cast<ShapedType>(genericOp.getOutputs()[0].getType())
          .getElementType();
  if (!elementTypeOut.isIntOrFloat()) {
    return failure();
  }
  unsigned bitWidthOut = elementTypeOut.getIntOrFloatBitWidth();
  auto elementTypeIn =
      llvm::cast<ShapedType>(genericOp.getInputs()[0].getType())
          .getElementType();
  if (!elementTypeIn.isIntOrFloat()) {
    return failure();
  }
  unsigned bitWidthIn = elementTypeIn.getIntOrFloatBitWidth();
  if (bitWidthIn >= bitWidthOut) {
    return failure();
  }

  return success();
}

//----------------------------------------------------------------------------//
//                                Patterns
//----------------------------------------------------------------------------//

// This pattern does a basic fusion of dequantization + matmul `linalg.generic`
// ops, moving them into a single `flow.dispatch.region` op.
class FuseDequantizationMatmulPattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Fail if matmul is already in a dispatch
    if (!IREE::Flow::isNonNullAndOutsideDispatch(genericOp)) {
      return failure();
    }
    // Match first generic op as matmul
    if (failed(isGroupedContractionOp(genericOp))) {
      return failure();
    }

    Value genericOpResult = genericOp->getResult(0);
    Operation *matmulOp = genericOpResult.getDefiningOp();

    // Match operands to dequantizations and fuse if matched
    Value lhs = genericOp->getOperand(0);
    Value rhs = genericOp->getOperand(1);
    auto lhsOp = lhs.getDefiningOp<linalg::GenericOp>();
    auto rhsOp = rhs.getDefiningOp<linalg::GenericOp>();

    std::optional<Operation *> maybeFill = std::nullopt;
    if (auto fill = genericOp.getDpsInitOperand(0)
                        ->get()
                        .getDefiningOp<linalg::FillOp>()) {
      maybeFill = fill;
    }

    if (lhsOp)
      if (!failed(isGroupedDequantizationOp(
              llvm::dyn_cast<linalg::GenericOp>(*lhsOp)))) {
        return fuseDequantAndMatmul(rewriter, lhsOp, matmulOp, maybeFill);
      }
    if (rhsOp)
      if (!failed(isGroupedDequantizationOp(
              llvm::dyn_cast<linalg::GenericOp>(*rhsOp)))) {
        return fuseDequantAndMatmul(rewriter, rhsOp, matmulOp, maybeFill);
      }

    return failure();
  }
};

struct FuseDequantizationMatmulPass
    : public FuseDequantizationMatmulBase<FuseDequantizationMatmulPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    math::MathDialect>();
  }
  FuseDequantizationMatmulPass(bool enableQuantizedMatmulReassociation) {
    this->enableQuantizedMatmulReassociation =
        enableQuantizedMatmulReassociation;
  }
  FuseDequantizationMatmulPass(const FuseDequantizationMatmulPass &pass)
      : FuseDequantizationMatmulPass(pass.enableQuantizedMatmulReassociation) {}

  void runOnOperation() override;
};

} // namespace

void FuseDequantizationMatmulPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  // Perform reassociation if enabled
  if (this->enableQuantizedMatmulReassociation) {
    SmallVector<std::pair<linalg::GenericOp, linalg::GenericOp>> candidates;
    for (auto genericOp :
         funcOp.getFunctionBody().getOps<linalg::GenericOp>()) {
      if (failed(isGroupedContractionOp(genericOp))) {
        continue;
      }

      OpOperand *lhs = genericOp.getDpsInputOperand(0);
      OpOperand *rhs = genericOp.getDpsInputOperand(1);
      auto lhsOp = lhs->get().getDefiningOp<linalg::GenericOp>();
      auto rhsOp = rhs->get().getDefiningOp<linalg::GenericOp>();
      if (!llvm::cast<ShapedType>(genericOp.getInputs()[0].getType())
               .hasStaticShape() ||
          !llvm::cast<ShapedType>(genericOp.getInputs()[1].getType())
               .hasStaticShape() ||
          !llvm::cast<ShapedType>(genericOp.getResults()[0].getType())
               .hasStaticShape()) {
        // Codegen can't handle the dynamic case yet.
        continue;
      }
      if (lhsOp) {
        if (!failed(isGroupedDequantizationOp(lhsOp))) {
          candidates.push_back(std::make_pair(lhsOp, genericOp));
          continue;
        }
      }
      if (rhsOp) {
        if (!failed(isGroupedDequantizationOp(rhsOp))) {
          candidates.push_back(std::make_pair(rhsOp, genericOp));
        }
      }
    }
    IRRewriter rewriter(context);
    for (auto candidate : candidates) {
      rewriter.setInsertionPointAfter(candidate.second);
      if (failed(reassociateAndFuseDequantMatmul(rewriter, candidate.first,
                                                 candidate.second))) {
        return signalPassFailure();
      }
    }
  }

  // Normal fusion pattern.
  {
    RewritePatternSet patterns(context);
    patterns.insert<FuseDequantizationMatmulPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseDequantizationMatmulPass(bool enableQuantizedMatmulReassociation) {
  return std::make_unique<FuseDequantizationMatmulPass>(
      enableQuantizedMatmulReassociation);
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
