// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rewrites contractions over affine-dequantized operands using
//
//   sum_k (Aq - zA)*(Bq - zB) = D - zB*RA - zA*RB + N*zA*zB,
//
// where D is the integer contraction and RA/RB sum the corresponding operand.
// Quantization parameters must be invariant over the integer reduction dims.
// Other reduction dims become parallel in D and are reduced after scaling in
// the epilogue. The integer contraction preserves the original contraction's
// iteration dimension numbering; operand sums and the epilogue use their own.
// Integer corrections precede conversion to float to retain cancellation when
// D exceeds the floating-point mantissa.

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTQDQTOINTEGERMATHPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;

namespace {

static constexpr unsigned kAccumulatorWidth = 32;

/// Type of every integer intermediate: the contraction, the operand sums, and
/// the zero-point corrections. The legality check bounds all of them against
/// this width.
static IntegerType getAccumulatorType(MLIRContext *context) {
  return IntegerType::get(context, kAccumulatorWidth);
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// One dequantized operand. All map domains are the original contraction's
/// iteration space.
struct QuantizedOperand {
  DequantizeAffineOp dequantize;
  /// Range: quantized input data space.
  AffineMap inputMap;
  /// Range: scale data space.
  AffineMap scaleMap;
  /// Range: zero-point data space. Null if absent.
  AffineMap zeroPointMap;
};

/// A matched contraction with a nonempty, statically bounded integer reduction.
/// Construction succeeds only for a plain multiply-add body and a zero init.
/// Dimension indices refer to the original contraction's iteration space.
struct QuantizedContraction {
  QuantizedOperand lhs;
  QuantizedOperand rhs;
  /// Original contraction's indexing map for the output.
  AffineMap outputMap;
  /// Reduction dims over which all quantization parameters are invariant.
  SmallVector<unsigned> integerReductionDims;
  /// Reduction dims retained as parallel dims until the scaling epilogue.
  SmallVector<unsigned> floatingReductionDims;
  /// Product of integer reduction extents, bounded for the i32 accumulator.
  int64_t reductionExtent = 1;

  // Each operand sum is multiplied by the opposite operand's zero point.
  bool needsLhsSum() { return !rhs.dequantize.isSymmetric(); }
  bool needsRhsSum() { return !lhs.dequantize.isSymmetric(); }
};

/// Maximum magnitude representable by the quantized storage grid. Returns zero
/// when the grid cannot be accumulated in i32.
static int64_t getStorageMagnitude(DequantizeAffineOp dequantize) {
  unsigned width =
      cast<IntegerType>(dequantize.getInputType().getElementType()).getWidth();
  if (width >= kAccumulatorWidth) {
    return 0;
  }
  return dequantize.getInputUnsigned() ? llvm::maxUIntN(width)
                                       : int64_t{1} << (width - 1);
}

/// Conservative maximum magnitude of a dequantized operand's integer
/// difference, `input - zero_point`.
static int64_t getDifferenceMagnitude(DequantizeAffineOp dequantize) {
  int64_t storageMagnitude = getStorageMagnitude(dequantize);
  if (!storageMagnitude) {
    return 0;
  }

  int64_t inputMagnitude = storageMagnitude;
  if (dequantize.getQuantMin()) {
    inputMagnitude =
        std::max(-*dequantize.getQuantMin(), *dequantize.getQuantMax());
  }
  // A zero point is a value on the input's quantized grid. Its SSA carrier
  // type does not enlarge that grid; PT2E commonly uses i64 for i8 values.
  int64_t zeroPointMagnitude = dequantize.isSymmetric() ? 0 : storageMagnitude;
  return inputMagnitude + zeroPointMagnitude;
}

/// Maximum reduction extent for which every i32 partial sum and correction is
/// bounded by N * |Aq - zA|max * |Bq - zB|max <= INT32_MAX. Division avoids
/// overflow in the bound and preserves the positive endpoint of signed i32.
/// Returns zero when no positive reduction extent can be proven safe.
static int64_t getMaxReductionExtent(QuantizedContraction &contraction) {
  int64_t lhsMagnitude = getDifferenceMagnitude(contraction.lhs.dequantize);
  int64_t rhsMagnitude = getDifferenceMagnitude(contraction.rhs.dequantize);
  if (!lhsMagnitude || !rhsMagnitude) {
    return 0;
  }
  return llvm::maxIntN(kAccumulatorWidth) / lhsMagnitude / rhsMagnitude;
}

/// Matches exactly two scalar operations: input multiplication and accumulator
/// addition, in either operand order. Casts and other intervening operations
/// prevent matching.
static bool hasPlainFloatMulAddBody(linalg::LinalgOp op) {
  Block *body = op.getBlock();
  if (body->getNumArguments() != 3 || body->getOperations().size() != 3) {
    return false;
  }
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1) {
    return false;
  }
  auto addition = yield.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!addition) {
    return false;
  }
  Value accumulator = body->getArgument(2);
  Value product;
  if (addition.getLhs() == accumulator) {
    product = addition.getRhs();
  } else if (addition.getRhs() == accumulator) {
    product = addition.getLhs();
  } else {
    return false;
  }
  auto multiplication = product.getDefiningOp<arith::MulFOp>();
  if (!multiplication) {
    return false;
  }
  Value lhs = body->getArgument(0);
  Value rhs = body->getArgument(1);
  return (multiplication.getLhs() == lhs && multiplication.getRhs() == rhs) ||
         (multiplication.getLhs() == rhs && multiplication.getRhs() == lhs);
}

/// Rebase dequantization indexing maps onto the original contraction's
/// iteration space. The inverse output map maps dequantized data space back to
/// the dequantization iteration space, accounting for folded permutations.
static QuantizedOperand getQuantizedOperand(DequantizeAffineOp dequantize,
                                            AffineMap operandMap) {
  AffineMap toDequantizeIteration =
      inversePermutation(dequantize.getOutputMap()).compose(operandMap);
  return {dequantize, dequantize.getInputMap().compose(toDequantizeIteration),
          dequantize.getScaleMap().compose(toDequantizeIteration),
          dequantize.isSymmetric()
              ? AffineMap()
              : dequantize.getZeroPointMap().compose(toDequantizeIteration)};
}

static FailureOr<QuantizedContraction>
getQuantizedContraction(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      !op.hasPureTensorSemantics() || !hasPlainFloatMulAddBody(op)) {
    return failure();
  }
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat())) {
    return failure();
  }

  auto lhs = op.getDpsInputs()[0].getDefiningOp<DequantizeAffineOp>();
  auto rhs = op.getDpsInputs()[1].getDefiningOp<DequantizeAffineOp>();
  if (!lhs || !rhs) {
    return failure();
  }
  QuantizedContraction detail;
  detail.lhs = getQuantizedOperand(
      lhs, op.getMatchingIndexingMap(op.getDpsInputOperand(0)));
  detail.rhs = getQuantizedOperand(
      rhs, op.getMatchingIndexingMap(op.getDpsInputOperand(1)));

  detail.outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  // The output map has to be a projected permutation for the epilogue to be
  // expressible over the output data space and remaining reduction dimensions.
  if (!detail.outputMap.isProjectedPermutation()) {
    return failure();
  }

  int64_t maxReductionExtent = getMaxReductionExtent(detail);
  if (maxReductionExtent == 0) {
    return failure();
  }

  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  for (auto [dim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != utils::IteratorType::reduction) {
      continue;
    }
    // Contraction reductions are shared by both inputs and absent from output.
    if (detail.outputMap.isFunctionOfDim(dim) ||
        !detail.lhs.inputMap.isFunctionOfDim(dim) ||
        !detail.rhs.inputMap.isFunctionOfDim(dim)) {
      return failure();
    }
    if (detail.lhs.scaleMap.isFunctionOfDim(dim) ||
        detail.rhs.scaleMap.isFunctionOfDim(dim) ||
        (detail.lhs.zeroPointMap &&
         detail.lhs.zeroPointMap.isFunctionOfDim(dim)) ||
        (detail.rhs.zeroPointMap &&
         detail.rhs.zeroPointMap.isFunctionOfDim(dim))) {
      detail.floatingReductionDims.push_back(dim);
      continue;
    }
    detail.integerReductionDims.push_back(dim);
    int64_t extent = loopRanges[dim];
    // Unknown and empty reductions retain the original computation. Checking
    // before multiplying also prevents overflow for very large static shapes.
    if (ShapedType::isDynamic(extent) || extent == 0 ||
        detail.reductionExtent > maxReductionExtent / extent) {
      return failure();
    }
    detail.reductionExtent *= extent;
  }
  if (detail.integerReductionDims.empty()) {
    return failure();
  }
  return detail;
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static Value buildZeroFilledTensor(OpBuilder &b, Location loc,
                                   ArrayRef<OpFoldResult> sizes,
                                   Type elementType) {
  Value empty = tensor::EmptyOp::create(b, loc, sizes, elementType);
  Value zero = arith::ConstantOp::create(b, loc, b.getZeroAttr(elementType));
  return linalg::FillOp::create(b, loc, zero, empty).getResult(0);
}

static SmallVector<OpFoldResult>
buildIterationSizes(OpBuilder &b, Location loc, linalg::LinalgOp op,
                    QuantizedContraction &detail) {
  // Query storage shapes so tensor.dim uses do not keep dequantize ops alive
  // after the rewrite. Account for permutations folded into dequantization.
  SmallVector<OpFoldResult> flatShapes;
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    if (auto dequantize = value.getDefiningOp<DequantizeAffineOp>()) {
      SmallVector<OpFoldResult> inputSizes =
          tensor::getMixedSizes(b, loc, dequantize.getInput());
      AffineMap outputShapeMap = dequantize.getOutputMap().compose(
          inversePermutation(dequantize.getInputMap()));
      llvm::append_range(flatShapes, applyPermutationMap<OpFoldResult>(
                                         outputShapeMap, inputSizes));
    } else {
      llvm::append_range(flatShapes, tensor::getMixedSizes(b, loc, value));
    }
  }
  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      op.getShapesToLoopsMap().getResults(), [&](AffineExpr expr) {
        return affine::makeComposedFoldedAffineApply(b, loc, expr, flatShapes);
      });

  // Use init extents for result dimensions to preserve the original result
  // type, even when an input provides a static extent for a dynamic result dim.
  SmallVector<OpFoldResult> initSizes =
      tensor::getMixedSizes(b, loc, op.getDpsInits()[0]);
  for (auto [position, expr] : llvm::enumerate(detail.outputMap.getResults())) {
    sizes[cast<AffineDimExpr>(expr).getPosition()] = initSizes[position];
  }
  return sizes;
}

//===----------------------------------------------------------------------===//
// Integer contraction
//===----------------------------------------------------------------------===//

static linalg::GenericOp
buildIntegerContraction(OpBuilder &b, Location loc, linalg::LinalgOp op,
                        QuantizedContraction &detail,
                        ArrayRef<OpFoldResult> iterationSizes) {
  Value lhs = detail.lhs.dequantize.getInput();
  Value rhs = detail.rhs.dequantize.getInput();
  bool lhsUnsigned = detail.lhs.dequantize.getInputUnsigned();
  bool rhsUnsigned = detail.rhs.dequantize.getInputUnsigned();

  MLIRContext *context = b.getContext();
  IntegerType accumulatorType = getAccumulatorType(context);

  // Retain a partial result for each block; the epilogue scales and reduces it.
  SmallVector<AffineExpr> resultExprs(detail.outputMap.getResults());
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  for (unsigned dim : detail.floatingReductionDims) {
    resultExprs.push_back(getAffineDimExpr(dim, context));
    iteratorTypes[dim] = utils::IteratorType::parallel;
  }
  AffineMap resultMap =
      AffineMap::get(detail.outputMap.getNumDims(), 0, resultExprs, context);
  SmallVector<OpFoldResult> resultSizes =
      applyPermutationMap<OpFoldResult>(resultMap, iterationSizes);

  Value init = buildZeroFilledTensor(b, loc, resultSizes, accumulatorType);

  SmallVector<AffineMap> maps{detail.lhs.inputMap, detail.rhs.inputMap,
                              resultMap};
  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), ValueRange{lhs, rhs}, ValueRange{init}, maps,
      iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value lhsValue =
            convertScalarToDtype(nested, nestedLoc, args[0], accumulatorType,
                                 /*isUnsignedCast=*/lhsUnsigned);
        Value rhsValue =
            convertScalarToDtype(nested, nestedLoc, args[1], accumulatorType,
                                 /*isUnsignedCast=*/rhsUnsigned);
        Value product =
            arith::MulIOp::create(nested, nestedLoc, lhsValue, rhsValue);
        Value sum = arith::AddIOp::create(nested, nestedLoc, args[2], product);
        linalg::YieldOp::create(nested, nestedLoc, sum);
      });
  return genericOp;
}

//===----------------------------------------------------------------------===//
// Operand sums
//===----------------------------------------------------------------------===//

/// Returns a map selecting domain dimensions that do not appear as bare results
/// in any of `maps`, a nonempty list with a shared domain. These dimensions
/// need explicit extents because Linalg cannot infer them from operand shapes.
static AffineMap getShapeMap(ArrayRef<AffineMap> maps) {
  unsigned rank = maps.front().getNumDims();
  llvm::SmallBitVector namedDims(rank);
  for (AffineMap map : maps) {
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        namedDims.set(dimExpr.getPosition());
      }
    }
  }

  return AffineMap::getMultiDimIdentityMap(rank, maps.front().getContext())
      .dropResults(namedDims);
}

/// Builds an integer sum with the given input and output indexing maps.
/// `outputMap` is a projected permutation. Both maps use the domain described
/// by `iterationSizes`. Dimensions absent from both maps are dropped; those
/// used only by `inputMap` are reduced. Input values are extended to
/// `accumulatorType` according to `isUnsigned`.
static Value buildIntegerSum(OpBuilder &b, Location loc, Value input,
                             AffineMap inputMap, AffineMap outputMap,
                             ArrayRef<OpFoldResult> iterationSizes,
                             IntegerType accumulatorType, bool isUnsigned) {
  SmallVector<AffineMap> maps{inputMap, outputMap};
  llvm::SmallBitVector unusedDims = getUnusedDimsBitVector(maps);
  maps = compressUnusedDims(maps);

  SmallVector<OpFoldResult> sizes;
  SmallVector<utils::IteratorType> iteratorTypes;
  for (auto [dim, size] : llvm::enumerate(iterationSizes)) {
    if (unusedDims.test(dim)) {
      continue;
    }
    sizes.push_back(size);
    iteratorTypes.push_back(outputMap.isFunctionOfDim(dim)
                                ? utils::IteratorType::parallel
                                : utils::IteratorType::reduction);
  }

  // Window expressions such as oh * stride + kh * dilation do not expose kh's
  // extent. A shape-only input supplies missing extents without loading data.
  SmallVector<Value> inputs{input};
  AffineMap shapeMap = getShapeMap(maps);
  if (shapeMap.getNumResults() != 0) {
    SmallVector<OpFoldResult> shapeSizes =
        applyPermutationMap<OpFoldResult>(shapeMap, sizes);
    inputs.push_back(tensor::EmptyOp::create(
        b, loc, shapeSizes, getElementTypeOrSelf(input.getType())));
    maps.insert(maps.begin() + 1, shapeMap);
  }

  SmallVector<OpFoldResult> resultSizes =
      applyPermutationMap<OpFoldResult>(outputMap, iterationSizes);
  Value init = buildZeroFilledTensor(b, loc, resultSizes, accumulatorType);

  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), inputs, ValueRange{init}, maps, iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value value =
            convertScalarToDtype(nested, nestedLoc, args[0], accumulatorType,
                                 /*isUnsignedCast=*/isUnsigned);
        Value sum =
            arith::AddIOp::create(nested, nestedLoc, args.back(), value);
        linalg::YieldOp::create(nested, nestedLoc, sum);
      });
  return genericOp.getResult(0);
}

//===----------------------------------------------------------------------===//
// Zero-point correction
//===----------------------------------------------------------------------===//

struct IndexedValue {
  Value value;
  /// Indexing map from the caller's iteration space to this value's data space.
  AffineMap map;
};

/// Applies `buildValue` elementwise over the dimensions referenced by `inputs`.
/// Input maps share the domain described by `iterationSizes`. The result has
/// one axis per referenced dimension, in domain order; its map retains the
/// caller's domain so it can be used alongside the input maps. With no
/// referenced dimensions, evaluates the body once and returns a scalar.
static IndexedValue buildElementwise(
    OpBuilder &b, Location loc, ArrayRef<IndexedValue> inputs,
    ArrayRef<OpFoldResult> iterationSizes, Type elementType,
    llvm::function_ref<Value(OpBuilder &, Location, ValueRange)> buildValue) {
  SmallVector<Value> values;
  SmallVector<AffineMap> maps;
  for (IndexedValue input : inputs) {
    values.push_back(input.value);
    maps.push_back(input.map);
  }
  AffineMap resultMap =
      AffineMap::getMultiDimIdentityMap(iterationSizes.size(), b.getContext())
          .dropResults(getUnusedDimsBitVector(maps));
  unsigned rank = resultMap.getNumResults();
  if (rank == 0) {
    for (auto [value, map] : llvm::zip_equal(values, maps)) {
      if (isa<RankedTensorType>(value.getType())) {
        SmallVector<Value> indices;
        for (AffineExpr expr : map.getResults()) {
          indices.push_back(arith::ConstantIndexOp::create(
              b, loc, cast<AffineConstantExpr>(expr).getValue()));
        }
        value = tensor::ExtractOp::create(b, loc, value, indices);
      }
    }
    return {buildValue(b, loc, values), resultMap};
  }
  SmallVector<OpFoldResult> sizes =
      applyPermutationMap<OpFoldResult>(resultMap, iterationSizes);
  maps = compressUnusedDims(maps);
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, b.getContext()));
  Value init = tensor::EmptyOp::create(b, loc, sizes, elementType);
  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), values, ValueRange{init}, maps,
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value result = buildValue(nested, nestedLoc, args.drop_back());
        linalg::YieldOp::create(nested, nestedLoc, result);
      });
  return {genericOp.getResult(0), resultMap};
}

/// Computes factor * lhs * rhs, extending each input to `resultType` with its
/// own signedness before multiplication. Broadcasting follows the input maps.
static IndexedValue buildIntegerProduct(OpBuilder &b, Location loc,
                                        IndexedValue lhs, bool lhsUnsigned,
                                        IndexedValue rhs, bool rhsUnsigned,
                                        IntegerType resultType,
                                        ArrayRef<OpFoldResult> iterationSizes,
                                        int64_t factor = 1) {
  return buildElementwise(
      b, loc, {lhs, rhs}, iterationSizes, resultType,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) -> Value {
        Value lhsValue = convertScalarToDtype(nested, nestedLoc, args[0],
                                              resultType, lhsUnsigned);
        Value rhsValue = convertScalarToDtype(nested, nestedLoc, args[1],
                                              resultType, rhsUnsigned);
        Value product =
            arith::MulIOp::create(nested, nestedLoc, lhsValue, rhsValue);
        if (factor == 1) {
          return product;
        }
        Value multiplier = arith::ConstantOp::create(
            nested, nestedLoc, nested.getIntegerAttr(resultType, factor));
        return arith::MulIOp::create(nested, nestedLoc, product, multiplier);
      });
}

/// Builds zB*sum(Aq) + zA*sum(Bq) - N*zA*zB. Each product is a separate
/// computation so constant terms can be evaluated independently of runtime
/// terms and the integer contraction. Absent zero points contribute zero.
static IndexedValue
buildZeroPointCorrection(OpBuilder &b, Location loc,
                         QuantizedContraction &detail,
                         ArrayRef<OpFoldResult> iterationSizes) {
  IntegerType accumulatorType = getAccumulatorType(b.getContext());
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(iterationSizes.size(), b.getContext());
  llvm::SmallBitVector reductionDims(identity.getNumDims());
  for (unsigned dim : detail.integerReductionDims) {
    reductionDims.set(dim);
  }
  QuantizedOperand &lhs = detail.lhs;
  QuantizedOperand &rhs = detail.rhs;
  SmallVector<IndexedValue, 2> terms;
  if (detail.needsLhsSum()) {
    AffineMap sumMap = identity.dropResults(
        getUnusedDimsBitVector({lhs.inputMap}) | reductionDims);
    Value sum = buildIntegerSum(b, loc, lhs.dequantize.getInput(), lhs.inputMap,
                                sumMap, iterationSizes, accumulatorType,
                                lhs.dequantize.getInputUnsigned());
    terms.push_back(buildIntegerProduct(
        b, loc, {rhs.dequantize.getZeroPoint(), rhs.zeroPointMap},
        rhs.dequantize.getZpUnsigned(), {sum, sumMap}, /*rhsUnsigned=*/false,
        accumulatorType, iterationSizes));
  }
  if (detail.needsRhsSum()) {
    AffineMap sumMap = identity.dropResults(
        getUnusedDimsBitVector({rhs.inputMap}) | reductionDims);
    Value sum = buildIntegerSum(b, loc, rhs.dequantize.getInput(), rhs.inputMap,
                                sumMap, iterationSizes, accumulatorType,
                                rhs.dequantize.getInputUnsigned());
    terms.push_back(buildIntegerProduct(
        b, loc, {lhs.dequantize.getZeroPoint(), lhs.zeroPointMap},
        lhs.dequantize.getZpUnsigned(), {sum, sumMap}, /*rhsUnsigned=*/false,
        accumulatorType, iterationSizes));
  }
  if (terms.empty()) {
    Value zero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(accumulatorType));
    return {zero, AffineMap::get(identity.getNumDims(), 0, {}, b.getContext())};
  }
  if (terms.size() == 1) {
    return terms.front();
  }

  IndexedValue crossTerm = buildIntegerProduct(
      b, loc, {lhs.dequantize.getZeroPoint(), lhs.zeroPointMap},
      lhs.dequantize.getZpUnsigned(),
      {rhs.dequantize.getZeroPoint(), rhs.zeroPointMap},
      rhs.dequantize.getZpUnsigned(), accumulatorType, iterationSizes,
      detail.reductionExtent);
  return buildElementwise(
      b, loc, {terms[0], terms[1], crossTerm}, iterationSizes, accumulatorType,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) -> Value {
        Value sum = arith::AddIOp::create(nested, nestedLoc, args[0], args[1]);
        return arith::SubIOp::create(nested, nestedLoc, sum, args[2]);
      });
}

//===----------------------------------------------------------------------===//
// Scaling epilogue
//===----------------------------------------------------------------------===//

/// A finite f16 result need not have an f16-representable integer accumulator
/// or scale product. Scaling and block accumulation use at least f32,
/// preserving wider scale and output types as well.
static FloatType getEpilogueComputeType(Type resultType, Type lhsScaleType,
                                        Type rhsScaleType) {
  FloatType computeType = Float32Type::get(resultType.getContext());
  for (Type type : {resultType, lhsScaleType, rhsScaleType}) {
    auto floatType = cast<FloatType>(type);
    if (floatType.getWidth() > computeType.getWidth()) {
      computeType = floatType;
    }
  }
  return computeType;
}

static Value buildOutputConversion(OpBuilder &b, Location loc, Value input,
                                   RankedTensorType outputType,
                                   ArrayRef<OpFoldResult> sizes) {
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(outputType.getRank(), b.getContext());
  Type elementType = outputType.getElementType();
  Value output = tensor::EmptyOp::create(b, loc, sizes, elementType);
  return linalg::GenericOp::create(
             b, loc, outputType, ValueRange{input}, ValueRange{output},
             SmallVector<AffineMap>{identity, identity},
             SmallVector<utils::IteratorType>(outputType.getRank(),
                                              utils::IteratorType::parallel),
             [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
               Value result =
                   convertScalarToDtype(nested, nestedLoc, args[0], elementType,
                                        /*isUnsignedCast=*/false);
               linalg::YieldOp::create(nested, nestedLoc, result);
             })
      .getResult(0);
}

/// Subtracts the integer correction before converting and scaling each partial.
/// Dimensions omitted by outputMap are reduced after scaling. All input maps
/// use the domain described by iterationSizes; partials.map is a projected
/// permutation covering the output and remaining reduction dimensions.
static Value buildEpilogue(OpBuilder &b, Location loc, IndexedValue partials,
                           IndexedValue correction, IndexedValue lhsScale,
                           IndexedValue rhsScale, AffineMap outputMap,
                           RankedTensorType outputType,
                           ArrayRef<OpFoldResult> iterationSizes) {
  // Each integer partial becomes one epilogue iteration. The inverse maps that
  // iteration back to the caller's domain, filling integer reductions with
  // zero.
  AffineMap epilogueMap = inverseAndBroadcastProjectedPermutation(partials.map);
  unsigned rank = epilogueMap.getNumDims();
  SmallVector<Value> inputs;
  SmallVector<AffineMap> maps;
  for (IndexedValue input : {partials, correction, lhsScale, rhsScale}) {
    inputs.push_back(input.value);
    maps.push_back(input.map.compose(epilogueMap));
  }
  AffineMap resultMap = outputMap.compose(epilogueMap);
  maps.push_back(resultMap);
  SmallVector<utils::IteratorType> iteratorTypes;
  for (unsigned dim : llvm::seq<unsigned>(rank)) {
    iteratorTypes.push_back(resultMap.isFunctionOfDim(dim)
                                ? utils::IteratorType::parallel
                                : utils::IteratorType::reduction);
  }

  Type outputElementType = outputType.getElementType();
  FloatType computeType = getEpilogueComputeType(
      outputElementType, getElementTypeOrSelf(lhsScale.value.getType()),
      getElementTypeOrSelf(rhsScale.value.getType()));
  bool isReduction = rank > outputType.getRank();
  Type epilogueElementType = isReduction ? computeType : outputElementType;

  // A reduction accumulates into its init; an elementwise epilogue overwrites
  // it.
  SmallVector<OpFoldResult> resultSizes =
      applyPermutationMap<OpFoldResult>(outputMap, iterationSizes);
  Value init =
      isReduction
          ? buildZeroFilledTensor(b, loc, resultSizes, epilogueElementType)
          : tensor::EmptyOp::create(b, loc, resultSizes, epilogueElementType)
                .getResult();

  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), inputs, ValueRange{init}, maps, iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value corrected =
            arith::SubIOp::create(nested, nestedLoc, args[0], args[1]);
        Value real =
            arith::SIToFPOp::create(nested, nestedLoc, computeType, corrected);
        Value lhsScaleValue = convertScalarToDtype(
            nested, nestedLoc, args[2], computeType, /*isUnsignedCast=*/false);
        Value rhsScaleValue = convertScalarToDtype(
            nested, nestedLoc, args[3], computeType, /*isUnsignedCast=*/false);
        Value scale = arith::MulFOp::create(nested, nestedLoc, lhsScaleValue,
                                            rhsScaleValue);
        Value result = arith::MulFOp::create(nested, nestedLoc, real, scale);
        if (isReduction) {
          result =
              arith::AddFOp::create(nested, nestedLoc, args.back(), result);
        }
        result =
            convertScalarToDtype(nested, nestedLoc, result, epilogueElementType,
                                 /*isUnsignedCast=*/false);
        linalg::YieldOp::create(nested, nestedLoc, result);
      });
  if (epilogueElementType == outputElementType) {
    return genericOp.getResult(0);
  }

  // Blockwise accumulation stays wide; only its final result is narrowed.
  return buildOutputConversion(b, loc, genericOp.getResult(0), outputType,
                               resultSizes);
}

struct ConvertQDQToIntegerMath : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<QuantizedContraction> maybeDetail = getQuantizedContraction(op);
    if (failed(maybeDetail)) {
      return failure();
    }
    QuantizedContraction &detail = *maybeDetail;

    Location loc = op.getLoc();
    SmallVector<OpFoldResult> iterationSizes =
        buildIterationSizes(rewriter, loc, op, detail);

    linalg::GenericOp integerContraction =
        buildIntegerContraction(rewriter, loc, op, detail, iterationSizes);

    IndexedValue partials{integerContraction.getResult(0),
                          integerContraction.getIndexingMapsArray().back()};
    IndexedValue correction =
        buildZeroPointCorrection(rewriter, loc, detail, iterationSizes);
    Value result = buildEpilogue(
        rewriter, loc, partials, correction,
        {detail.lhs.dequantize.getScale(), detail.lhs.scaleMap},
        {detail.rhs.dequantize.getScale(), detail.rhs.scaleMap},
        detail.outputMap, cast<RankedTensorType>(op.getDpsInits()[0].getType()),
        iterationSizes);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertQDQToIntegerMathPass final
    : impl::ConvertQDQToIntegerMathPassBase<ConvertQDQToIntegerMathPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect,
                tensor::TensorDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<ConvertQDQToIntegerMath>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
