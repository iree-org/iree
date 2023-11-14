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

// Checks if the passed op is a contraction with two reduction dimensions
// This function checks that the genericOp:
// 1. isaContractionOpInterface
// 2. Has 2 reduction dimensions
static LogicalResult
isContractionWithTwoReductions(linalg::GenericOp genericOp) {
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

struct QuantizedMatmulRewriter {
  QuantizedMatmulRewriter(RewriterBase &rewriter, linalg::GenericOp dequant,
                          linalg::GenericOp matmul, int quantizedBitWidth);
  std::optional<SmallVector<OpOperand *>> getDequantMatmulInputs_f32();
  std::pair<SmallVector<AffineMap>, SmallVector<utils::IteratorType>>
  getGroupReductionMapsAndIterators(OpOperand *inputOperand);
  Value getGroupReductionInit(Value input);
  Value generateGroupMaxGeneric();
  Value generateScalesGeneric(Value groupMax);
  Value generateGroupSumsGeneric();
  SmallVector<Value> generateQuantizationGenerics();
  linalg::GenericOp
  generateQuantizedMatmulGeneric(SmallVector<Value> quantizeResults);
  linalg::GenericOp
  generateReassociatedDequantizationGeneric(SmallVector<Value> quantizeResults,
                                            Value quantizedIntegerMatmul);
  LogicalResult precondition();

private:
  // rewriter
  RewriterBase &rewriter;
  // linalg::GenericOp that performs the dequantization
  linalg::GenericOp dequant;
  // linalg::GenericOp that performs the matmul
  linalg::GenericOp matmul;
  // Destination bit width for dynamic quantization of the unquantized input
  // (`ins[1]`)
  int quantizedBitWidth;
  // Type for accumulation of integer matmul
  IntegerType accType;
  // Type for multiplication in integer matmul (should probably be same as
  // `accType`)
  IntegerType mulType;
  // Type for dynamic quantization of unquantized input
  IntegerType quantType;
  // inputs to the `dequant` and `matmul` ops
  // ins[0] = quantized matrix input to `dequant`
  // ins[1] = unquantized input to `matmul`
  // ins[2] = scales input to `dequant
  // ins[3] = zero points input to `dequant`
  // ins[4] = dequantized input to `matmul`
  SmallVector<OpOperand *> ins;
  // Location for rewrite
  Location loc;
};

// Takes as input the dequantization `linalg.generic` op and the matmul
// `linalg.generic` op, and returns the scales, zero points, quantized
// input matrix, unquantizaed input matrix, and dequantized result
// matrix.
// TODO(#) Have stricter matching on inputs. There may be cases where
// the current matching fails
std::optional<SmallVector<OpOperand *>>
QuantizedMatmulRewriter::getDequantMatmulInputs_f32() {
  assert(!failed(isContractionWithTwoReductions(matmul)) &&
         "expected `matmul` to be a contraction with two reduction dimensions");
  assert(!failed(isGroupedDequantizationOp(dequant)) &&
         "expected `dequant` to be a grouped dequantization");
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

QuantizedMatmulRewriter::QuantizedMatmulRewriter(RewriterBase &rewriter,
                                                 linalg::GenericOp dequant,
                                                 linalg::GenericOp matmul,
                                                 int quantizedBitWidth)
    : rewriter(rewriter), dequant(dequant), matmul(matmul),
      quantizedBitWidth(quantizedBitWidth), loc(dequant.getLoc()) {
  accType = rewriter.getI32Type();
  mulType = rewriter.getI32Type();
  quantType = rewriter.getIntegerType(quantizedBitWidth);
  std::optional<SmallVector<OpOperand *>> inputs = getDequantMatmulInputs_f32();
  if (inputs) {
    ins = *inputs;
  }
}

LogicalResult QuantizedMatmulRewriter::precondition() {
  if (ins.size() != 5) {
    llvm::dbgs() << "[" DEBUG_TYPE "]: "
                 << "expected `ins` to have 5 inputs\n";
    return failure();
  }
  OpOperand *unquantizedInputOperand = ins[1];
  Value unquantizedInput = ins[1]->get();
  RankedTensorType unquantizedInputType =
      llvm::cast<RankedTensorType>(unquantizedInput.getType());
  SmallVector<int64_t> unquantizedInputShape(unquantizedInputType.getShape());
  AffineMap indexingMap =
      matmul.getMatchingIndexingMap(unquantizedInputOperand);
  SmallVector<utils::IteratorType> matmulIteratorTypes =
      matmul.getIteratorTypesArray();
  if (unquantizedInputShape.size() < 2) {
    llvm::dbgs() << "[" DEBUG_TYPE "]: "
                 << "input expected to have a rank of at least 2\n";
    return failure();
  }
  if (matmulIteratorTypes[indexingMap.getNumDims() - 1] ==
          utils::IteratorType::parallel ||
      matmulIteratorTypes[indexingMap.getNumDims() - 2] ==
          utils::IteratorType::parallel) {
    llvm::dbgs() << "[" DEBUG_TYPE "]: "
                 << "inner 2 dimensions of matmul expected to be reduction\n";
    return failure();
  }
  auto affineExprs = indexingMap.getResults();
  auto innerDim0 = dyn_cast<AffineDimExpr>(affineExprs.back());
  auto innerDim1 = dyn_cast<AffineDimExpr>(affineExprs[affineExprs.size() - 2]);
  if (!innerDim0 || !innerDim1 ||
      innerDim0.getPosition() != indexingMap.getNumDims() - 1 ||
      innerDim1.getPosition() != indexingMap.getNumDims() - 2) {
    llvm::dbgs() << "[" DEBUG_TYPE "]: "
                 << "inner shape of input expected to be reduced in matmul\n";
    return failure();
  }
  if (!unquantizedInputType.getElementType().isa<FloatType>()) {
    llvm::dbgs() << "[" DEBUG_TYPE "]: "
                 << "expected float type\n";
    return failure();
  }
  Value scales = ins[2]->get();
  Value zps = ins[3]->get();
  OpOperand *matmulDequantizedOperand = ins[4];
  auto matmulDequantizedInputExprs =
      matmul.getMatchingIndexingMap(matmulDequantizedOperand).getResults();
  auto scalesType = llvm::dyn_cast<RankedTensorType>(scales.getType());
  auto zpsType = llvm::dyn_cast<RankedTensorType>(zps.getType());
  if (!scalesType || !zpsType) {
    llvm::dbgs()
        << "[" DEBUG_TYPE "]: "
        << "expected scales and zero points to have RankedTensorType\n";
    return failure();
  }
  if (scalesType.getShape().size() != matmulDequantizedInputExprs.size() - 1) {
    if (scalesType.getShape().size() != matmulDequantizedInputExprs.size() ||
        scalesType.getShape().back() != 1) {
      llvm::dbgs() << "[" DEBUG_TYPE "]: "
                   << "unexpected rank for scales\n";
      return failure();
    }
  }
  if (zpsType.getShape().size() != matmulDequantizedInputExprs.size() - 1) {
    if (zpsType.getShape().size() != matmulDequantizedInputExprs.size() ||
        zpsType.getShape().back() != 1) {
      llvm::dbgs() << "[" DEBUG_TYPE "]: "
                   << "unexpected rank for zero points\n";
      return failure();
    }
  }
  return success();
}

std::pair<SmallVector<AffineMap>, SmallVector<utils::IteratorType>>
QuantizedMatmulRewriter::getGroupReductionMapsAndIterators(
    OpOperand *inputOperand) {
  Value input = inputOperand->get();
  RankedTensorType inputType = llvm::cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> inputShape(inputType.getShape());
  AffineMap indexingMap = matmul.getMatchingIndexingMap(inputOperand);
  SmallVector<utils::IteratorType> matmulIteratorTypes =
      matmul.getIteratorTypesArray();
  assert(inputShape.size() >= 2 &&
         "input expected to have a rank of at least 2");
  assert((matmulIteratorTypes[indexingMap.getNumDims() - 1] ==
              utils::IteratorType::reduction &&
          matmulIteratorTypes[indexingMap.getNumDims() - 2] ==
              utils::IteratorType::reduction) &&
         "inner 2 dimensions of matmul expected to be reduction");
  auto affineExprs = indexingMap.getResults();
  auto innerDim0 = dyn_cast<AffineDimExpr>(affineExprs.back());
  auto innerDim1 = dyn_cast<AffineDimExpr>(affineExprs[affineExprs.size() - 2]);
  assert(innerDim0 && innerDim1 &&
         innerDim0.getPosition() == indexingMap.getNumDims() - 1 &&
         innerDim1.getPosition() == indexingMap.getNumDims() - 2 &&
         "inner shape of input expected to be reduced in matmul");
  (void)innerDim0;
  (void)innerDim1;

  SmallVector<utils::IteratorType> iterators(inputShape.size(),
                                             utils::IteratorType::parallel);
  iterators.back() = utils::IteratorType::reduction;
  AffineMap inputMap = rewriter.getMultiDimIdentityMap(inputShape.size());
  AffineMap outputMap = rewriter.getMultiDimIdentityMap(inputShape.size())
                            .getMajorSubMap(inputShape.size() - 1);
  SmallVector<AffineMap> maps{inputMap, outputMap};
  return std::make_pair(maps, iterators);
}

// Helper to create an init Value for reductions along the group dimension.
Value QuantizedMatmulRewriter::getGroupReductionInit(Value input) {
  RankedTensorType inputType = llvm::cast<RankedTensorType>(input.getType());
  assert(inputType.getElementType().isa<FloatType>() && "expected float type");
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(inputType.getElementType(), 0.0));
  SmallVector<int64_t> inputShape(inputType.getShape());
  SmallVector<int64_t> outputShape(inputShape.begin(), inputShape.end() - 1);
  RankedTensorType outputType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  Value emptyOut = rewriter.create<tensor::EmptyOp>(
      loc, outputType.getShape(), outputType.getElementType());
  return rewriter.create<linalg::FillOp>(loc, zero, emptyOut).result();
}

// Creates a generic that computes the absolute max along the group
// dimension, and returns the result.
Value QuantizedMatmulRewriter::generateGroupMaxGeneric() {
  OpOperand *inputOperand = ins[1];
  std::pair<SmallVector<AffineMap>, SmallVector<utils::IteratorType>>
      mapsAndIterators = getGroupReductionMapsAndIterators(inputOperand);
  auto maps = mapsAndIterators.first;
  auto iterators = mapsAndIterators.second;
  Value input = inputOperand->get();
  Value output = getGroupReductionInit(input);
  auto groupMaxOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), input, output, maps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value abs = b.create<math::AbsFOp>(nestedLoc, args[0]);
        Value max = b.create<arith::MaximumFOp>(nestedLoc, abs, args[1]);
        b.create<linalg::YieldOp>(nestedLoc, max);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "groupMaxOp:   " << groupMaxOp << "\n";
  return groupMaxOp.getResult(0);
}

// Creates a generic that computes the scales for each group, and
// returns the result.
Value QuantizedMatmulRewriter::generateScalesGeneric(Value groupMax) {
  auto groupMaxType = llvm::cast<RankedTensorType>(groupMax.getType());
  assert(groupMaxType.getElementType().isa<FloatType>() &&
         "expected float type");
  Value cst = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(groupMaxType.getElementType(),
                                 (1 << (quantizedBitWidth - 1)) - 1));
  Value output = rewriter.create<tensor::EmptyOp>(
      loc, groupMaxType.getShape(), groupMaxType.getElementType());
  SmallVector<AffineMap> maps(
      2, rewriter.getMultiDimIdentityMap(groupMaxType.getShape().size()));
  auto scalesOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), groupMax, output, maps,
      getParallelAndReductionIterators(groupMaxType.getRank(), 0),
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value scale = b.create<arith::DivFOp>(nestedLoc, args[0], cst);
        b.create<linalg::YieldOp>(nestedLoc, scale);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "scalesOp:   " << scalesOp << "\n";
  return scalesOp.getResult(0);
}

// Creates a generic that computes the sums for each group, and
// returns the result.
Value QuantizedMatmulRewriter::generateGroupSumsGeneric() {
  OpOperand *inputOperand = ins[1];
  std::pair<SmallVector<AffineMap>, SmallVector<utils::IteratorType>>
      mapsAndIterators = getGroupReductionMapsAndIterators(inputOperand);
  auto maps = mapsAndIterators.first;
  auto iterators = mapsAndIterators.second;
  Value input = inputOperand->get();
  Value output = getGroupReductionInit(input);
  auto groupSumsOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), input, output, maps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value sum = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
        b.create<linalg::YieldOp>(nestedLoc, sum);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "groupSumsOp:   " << groupSumsOp << "\n";
  return groupSumsOp.getResult(0);
}

// Creates 4 linalg::GenericOps that collectively perform a symmetric
// quantization of the unquantized input, and returns the results.
SmallVector<Value> QuantizedMatmulRewriter::generateQuantizationGenerics() {
  assert(ins.size() == 5 && "expected `ins` to have 5 inputs");
  OpOperand *unquantizedInputOperand = ins[1];
  Value unquantizedInput = unquantizedInputOperand->get();

  auto unquantizedType =
      llvm::cast<RankedTensorType>(unquantizedInput.getType());
  SmallVector<int64_t> unquantizedShape(unquantizedType.getShape());

  IntegerType quantizedElementType = rewriter.getIntegerType(quantizedBitWidth);
  Value groupMax = generateGroupMaxGeneric();
  Value scales = generateScalesGeneric(groupMax);
  Value groupSums = generateGroupSumsGeneric();

  Value output = rewriter.create<tensor::EmptyOp>(loc, unquantizedShape,
                                                  quantizedElementType);
  AffineMap inputMap = rewriter.getMultiDimIdentityMap(unquantizedShape.size());
  AffineMap scalesMap = rewriter.getMultiDimIdentityMap(unquantizedShape.size())
                            .getMajorSubMap(unquantizedShape.size() - 1);
  AffineMap outputMap =
      rewriter.getMultiDimIdentityMap(unquantizedShape.size());
  SmallVector<AffineMap> maps{inputMap, scalesMap, outputMap};
  auto quantizeOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{unquantizedInput, scales}, output, maps,
      getParallelAndReductionIterators(unquantizedShape.size(), 0),
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value scaled = b.create<arith::DivFOp>(nestedLoc, args[0], args[1]);
        Value quant =
            b.create<arith::FPToSIOp>(nestedLoc, quantizedElementType, scaled);
        b.create<linalg::YieldOp>(nestedLoc, quant);
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "quantizeOp:   " << quantizeOp << "\n";
  Value newQuantizedInput = quantizeOp.getResult(0);
  SmallVector<Value> results{groupMax, scales, groupSums, newQuantizedInput};
  return results;
}

// Creates a generic that computes the main matmul computation in integer
// arithmetic, while values are still quantized, and returns the result.
linalg::GenericOp QuantizedMatmulRewriter::generateQuantizedMatmulGeneric(
    SmallVector<Value> quantizeResults) {
  Value newQuantizedInput = quantizeResults.back();
  assert(ins.size() == 5 && "expected `ins` to have 5 inputs");
  Value quantizedInput = ins[0]->get();
  OpOperand *matmulUnquantizedOperand = ins[1];
  OpOperand *matmulDequantizedOperand = ins[4];
  OpOperand *matmulOutputOperand = matmul.getDpsInitOperand(0);

  SmallVector<utils::IteratorType> iterators =
      getParallelAndReductionIterators(matmul.getNumLoops(), 1);
  SmallVector<AffineMap> maps;
  AffineMap matmulUnquantizedMap =
      matmul.getMatchingIndexingMap(matmulUnquantizedOperand);
  AffineMap matmulDequantizedMap =
      matmul.getMatchingIndexingMap(matmulDequantizedOperand);
  AffineMap matmulOutputMap =
      matmul.getMatchingIndexingMap(matmulOutputOperand);
  maps.push_back(matmulUnquantizedMap);
  maps.push_back(matmulDequantizedMap);
  SmallVector<AffineExpr> outputExprs(matmulOutputMap.getResults());
  outputExprs.push_back(rewriter.getAffineDimExpr(matmul.getNumLoops() - 2));
  maps.push_back(AffineMap::get(iterators.size(), 0, outputExprs,
                                outputExprs.front().getContext()));

  SmallVector<int64_t> newQuantizedInputShape(
      llvm::cast<RankedTensorType>(newQuantizedInput.getType()).getShape());
  assert(newQuantizedInputShape.size() >= 2 &&
         "expected new quantized input to have a rank of at least 2");
  SmallVector<int64_t> outputShape(
      llvm::cast<RankedTensorType>(matmulOutputOperand->get().getType())
          .getShape());
  outputShape.push_back(
      newQuantizedInputShape[newQuantizedInputShape.size() - 2]);
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(accType, 0.0));
  Value emptyOut = rewriter.create<tensor::EmptyOp>(loc, outputShape, accType);
  Value output = rewriter.create<linalg::FillOp>(loc, zero, emptyOut).result();
  auto integerMatmulOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), ValueRange{newQuantizedInput, quantizedInput},
      output, maps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value mul;
        if (quantType == mulType) {
          Value ext1 = b.create<arith::ExtUIOp>(nestedLoc, mulType, args[1]);
          mul = b.create<arith::MulIOp>(nestedLoc, args[0], ext1);
        } else {
          Value ext0 = b.create<arith::ExtSIOp>(nestedLoc, mulType, args[0]);
          Value ext1 = b.create<arith::ExtUIOp>(nestedLoc, mulType, args[1]);
          mul = b.create<arith::MulIOp>(nestedLoc, ext0, ext1);
        }
        Value sum;
        if (mulType == accType) {
          sum = b.create<arith::AddIOp>(nestedLoc, mul, args[2]);
        } else {
          Value extMul = b.create<arith::ExtSIOp>(nestedLoc, accType, mul);
          sum = b.create<arith::AddIOp>(nestedLoc, extMul, args[2]);
        }
        b.create<linalg::YieldOp>(nestedLoc, sum);
        llvm::dbgs() << "[" DEBUG_TYPE "]: "
                     << "15\n";
      });
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "16\n";
  llvm::dbgs() << "[" DEBUG_TYPE "]: "
               << "integerMatmulOp:   " << integerMatmulOp << "\n";
  return integerMatmulOp;
}

// Creates a generic that does the reassociated dequantization math, and does
// the final reduction along the number of groups dimension.
linalg::GenericOp
QuantizedMatmulRewriter::generateReassociatedDequantizationGeneric(
    SmallVector<Value> quantizeResults, Value quantizedIntegerMatmul) {
  assert(quantizeResults.size() == 4 &&
         "expected 4 ops from quantization step");
  Value scales = ins[2]->get();
  Value zps = ins[3]->get();
  Value newScales = quantizeResults[1];
  Value groupSums = quantizeResults[2];

  SmallVector<utils::IteratorType> iterators =
      getParallelAndReductionIterators(matmul.getNumLoops() - 1, 1);
  SmallVector<AffineMap> maps;
  AffineMap quantizedIntegerMatmulMap =
      rewriter.getMultiDimIdentityMap(iterators.size());
  maps.push_back(quantizedIntegerMatmulMap);

  {
    OpOperand *matmulUnquantizedOperand = ins[1];
    auto exprs =
        matmul.getMatchingIndexingMap(matmulUnquantizedOperand).getResults();
    exprs = exprs.slice(0, exprs.size() - 1);
    auto newScalesAndSumsMap =
        AffineMap::get(iterators.size(), 0, exprs, exprs.front().getContext());
    maps.push_back(newScalesAndSumsMap);
    maps.push_back(newScalesAndSumsMap);
  }

  {
    OpOperand *matmulDequantizedOperand = ins[4];
    auto exprsRef =
        matmul.getMatchingIndexingMap(matmulDequantizedOperand).getResults();
    SmallVector<AffineExpr> exprs(exprsRef.slice(0, exprsRef.size() - 1));
    RankedTensorType scalesType =
        llvm::cast<RankedTensorType>(scales.getType());
    RankedTensorType zpsType = llvm::cast<RankedTensorType>(zps.getType());
    if (exprs.size() < scalesType.getShape().size() &&
        scalesType.getShape().back() == 1 && zpsType.getShape().back() == 1) {
      exprs.push_back(rewriter.getAffineConstantExpr(0));
    }
    assert(exprs.size() == scalesType.getShape().size() &&
           "unexpected rank for scales");
    assert(exprs.size() == zpsType.getShape().size() &&
           "unexpected rank for zero points");
    auto scalesAndZpsMap =
        AffineMap::get(iterators.size(), 0, exprs, exprs.front().getContext());
    maps.push_back(scalesAndZpsMap);
    maps.push_back(scalesAndZpsMap);
  }

  OpOperand *matmulOutputOperand = matmul.getDpsInitOperand(0);
  SmallVector<AffineExpr> outputExprs(
      matmul.getMatchingIndexingMap(matmulOutputOperand).getResults());
  auto outputMap = AffineMap::get(iterators.size(), 0, outputExprs,
                                  outputExprs.front().getContext());
  maps.push_back(outputMap);

  Type i32Type = rewriter.getI32Type();
  Type f32Type = rewriter.getF32Type();
  Value output = matmulOutputOperand->get();
  auto reassociatedDequantizationOp = rewriter.create<linalg::GenericOp>(
      loc, output.getType(),
      ValueRange{quantizedIntegerMatmul, newScales, groupSums, scales, zps},
      output, maps, iterators,
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
               << "reassociatedDequantizationOp:   "
               << reassociatedDequantizationOp << "\n";
  return reassociatedDequantizationOp;
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
                                                     linalg::GenericOp matmul,
                                                     int quantizeBitWidth) {
  QuantizedMatmulRewriter qmr(rewriter, dequant, matmul, quantizeBitWidth);
  if (failed(qmr.precondition())) {
    return success();
  }
  SmallVector<Value> quantizeResults = qmr.generateQuantizationGenerics();
  linalg::GenericOp quantizedIntegerMatmul =
      qmr.generateQuantizedMatmulGeneric(quantizeResults);
  linalg::GenericOp reassociatedDequantization =
      qmr.generateReassociatedDequantizationGeneric(
          quantizeResults, quantizedIntegerMatmul.getResult(0));

  rewriter.replaceOp(matmul, reassociatedDequantization.getResult(0));

  // Fuse dequantization + matmul ops into a single dispatch region
  SmallVector<Operation *> dequantMatmulOps{quantizedIntegerMatmul,
                                            reassociatedDequantization};
  FailureOr<IREE::Flow::DispatchRegionOp> maybeDequantMatmulDispatch =
      wrapConsecutiveOpsInDispatchRegion(rewriter, dequantMatmulOps);
  if (failed(maybeDequantMatmulDispatch)) {
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
    if (failed(isContractionWithTwoReductions(genericOp))) {
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
    int quantizeBitWidth = 16;
    SmallVector<std::pair<linalg::GenericOp, linalg::GenericOp>> candidates;
    for (auto genericOp :
         funcOp.getFunctionBody().getOps<linalg::GenericOp>()) {
      if (failed(isContractionWithTwoReductions(genericOp))) {
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
      if (failed(reassociateAndFuseDequantMatmul(
              rewriter, candidate.first, candidate.second, quantizeBitWidth))) {
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
