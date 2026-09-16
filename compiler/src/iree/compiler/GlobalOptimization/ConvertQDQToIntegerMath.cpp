// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rewrites contractions over affine-dequantized operands using
//
//   sum_k (Aq - zA)*(Bq - zB) = D - zB*RA - zA*RB + N*zA*zB,
//
// where D is the integer contraction, RA/RB sum the corresponding quantized
// operand, and N is the product of the integer reduction extents.
// Quantization parameters must be invariant over the integer reduction dims.
// Other reduction dims become parallel in D and are reduced after scaling in
// the epilogue. The integer contraction preserves the original contraction's
// iteration dimension numbering; operand sums and the epilogue use their own.
// Integer corrections precede conversion to float to retain cancellation when
// D is too large to be represented exactly in the floating-point type.

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"
#include "iree/compiler/GlobalOptimization/QuantizedContraction.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTQDQTOINTEGERMATHPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;

namespace {

using detail::getExplicitExtentDimsMap;
using detail::getQuantizedContraction;
using detail::QuantizedContraction;
using detail::QuantizedOperand;

/// Type of every integer intermediate: the contraction, the operand sums, and
/// the zero-point corrections. The legality check bounds all of them against
/// this width.
static IntegerType getAccumulatorType(MLIRContext *context) {
  return IntegerType::get(context, kAccumulatorWidth);
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

/// Computes static or dynamic extents in the original contraction's loop order
/// (e.g. [M, N, K] for matmul), including output and reduction dimensions.
/// Uses quantized storage shapes and preserves the output init's extents.
static SmallVector<OpFoldResult>
buildIterationSizes(OpBuilder &b, Location loc, linalg::LinalgOp op,
                    const QuantizedContraction &detail) {
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
                        const QuantizedContraction &detail,
                        ArrayRef<OpFoldResult> iterationSizes) {
  DequantizeAffineOp lhsDequantize = detail.lhs.dequantize;
  DequantizeAffineOp rhsDequantize = detail.rhs.dequantize;
  Value lhs = lhsDequantize.getInput();
  Value rhs = rhsDequantize.getInput();
  bool lhsUnsigned = lhsDequantize.getInputUnsigned();
  bool rhsUnsigned = rhsDequantize.getInputUnsigned();

  MLIRContext *context = b.getContext();
  const IntegerType accumulatorType = getAccumulatorType(context);

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

  // Scalar body: accumulate the product of quantized inputs into an i32 sum.
  //   a = lhsUnsigned ? extui(lhsElement) : extsi(lhsElement)
  //   b = rhsUnsigned ? extui(rhsElement) : extsi(rhsElement)
  //   yield accumulator + a * b
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
  AffineMap shapeMap = getExplicitExtentDimsMap(maps);
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

/// Computes factor * lhs * rhs after converting each input to `resultType`.
/// Widening uses each input's signedness; wider zero-point carriers are
/// narrowed under the requirement that their values fit the storage grid.
/// Broadcasting follows the input maps.
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

/// Builds the correction subtracted from D: zB*sum(Aq) + zA*sum(Bq) - N*zA*zB.
/// Sums span the integer reduction dims, whose extents multiply to N. Each
/// product is a separate computation so constant terms can be evaluated
/// independently of runtime terms and the integer contraction. Absent zero
/// points contribute zero.
static IndexedValue
buildZeroPointCorrection(OpBuilder &b, Location loc,
                         const QuantizedContraction &detail,
                         ArrayRef<OpFoldResult> iterationSizes) {
  IntegerType accumulatorType = getAccumulatorType(b.getContext());
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(iterationSizes.size(), b.getContext());
  llvm::SmallBitVector reductionDims(identity.getNumDims());
  for (unsigned dim : detail.integerReductionDims) {
    reductionDims.set(dim);
  }
  const QuantizedOperand &lhs = detail.lhs;
  DequantizeAffineOp lhsDequantize = lhs.dequantize;
  const QuantizedOperand &rhs = detail.rhs;
  DequantizeAffineOp rhsDequantize = rhs.dequantize;
  SmallVector<IndexedValue, 2> terms;
  if (detail.needsLhsSum()) {
    AffineMap sumMap = identity.dropResults(
        getUnusedDimsBitVector({lhs.inputMap}) | reductionDims);
    Value sum = buildIntegerSum(b, loc, lhsDequantize.getInput(), lhs.inputMap,
                                sumMap, iterationSizes, accumulatorType,
                                lhsDequantize.getInputUnsigned());
    terms.push_back(buildIntegerProduct(
        b, loc, {rhsDequantize.getZeroPoint(), rhs.zeroPointMap},
        rhsDequantize.getZpUnsigned(), {sum, sumMap}, /*rhsUnsigned=*/false,
        accumulatorType, iterationSizes));
  }
  if (detail.needsRhsSum()) {
    AffineMap sumMap = identity.dropResults(
        getUnusedDimsBitVector({rhs.inputMap}) | reductionDims);
    Value sum = buildIntegerSum(b, loc, rhsDequantize.getInput(), rhs.inputMap,
                                sumMap, iterationSizes, accumulatorType,
                                rhsDequantize.getInputUnsigned());
    terms.push_back(buildIntegerProduct(
        b, loc, {lhsDequantize.getZeroPoint(), lhs.zeroPointMap},
        lhsDequantize.getZpUnsigned(), {sum, sumMap}, /*rhsUnsigned=*/false,
        accumulatorType, iterationSizes));
  }
  if (terms.empty()) {
    // Keep a uniform epilogue input list. Subsequent cleanup passes are
    // expected to fold away this zero correction and the subtraction from D.
    Value zero =
        arith::ConstantOp::create(b, loc, b.getZeroAttr(accumulatorType));
    return {zero, AffineMap::get(identity.getNumDims(), 0, {}, b.getContext())};
  }
  if (terms.size() == 1) {
    return terms.front();
  }

  IndexedValue crossTerm = buildIntegerProduct(
      b, loc, {lhsDequantize.getZeroPoint(), lhs.zeroPointMap},
      lhsDequantize.getZpUnsigned(),
      {rhsDequantize.getZeroPoint(), rhs.zeroPointMap},
      rhsDequantize.getZpUnsigned(), accumulatorType, iterationSizes,
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

/// Rewrite contractions over affine-dequantized operands as integer
/// contractions with zero-point corrections and a scaling epilogue:
///
/// For fixed output coordinates and fixed remaining reduction coordinates,
/// k spans the integer reduction dims and N is the product of their extents.
/// Scales sA/sB and zero points zA/zB are invariant over k. Algebraically:
///
/// ```text
/// A[k] = sA * (Aq[k] - zA)
/// B[k] = sB * (Bq[k] - zB)
/// D    = sum_k Aq[k] * Bq[k]
/// RA   = sum_k Aq[k]
/// RB   = sum_k Bq[k]
///
/// C = sum_k A[k] * B[k]
///   = sA * sB * sum_k (Aq[k] - zA) * (Bq[k] - zB)
///   = sA * sB * (D - zB * RA - zA * RB + N * zA * zB)
/// ```
///
/// Here C is one scaled partial; the epilogue sums these partials over any
/// remaining reduction dims. With no remaining reductions, C is the output.
///
/// Compose storage and parameter maps into contraction coordinates. Reduce
/// in integer arithmetic where parameters are invariant; retain the other
/// reduction dimensions as partial results for the floating-point epilogue.
/// Windowed operand sums use the same access expressions as the
/// contraction.
struct ConvertQDQToIntegerMath : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // Match a zero-initialized multiply-add contraction of two dequantized
    // inputs. Separate reductions with invariant quantization parameters from
    // those requiring floating-point accumulation, and prove that the integer
    // intermediates fit in i32 before constructing the replacement.
    FailureOr<QuantizedContraction> maybeDetail = getQuantizedContraction(op);
    if (failed(maybeDetail)) {
      return failure();
    }
    QuantizedContraction &detail = *maybeDetail;

    // Recover the original loop extents to size the intermediate tensors.
    // Query quantized storage shapes so shape computations do not keep the
    // dequantization ops alive, while preserving the output init's extents.
    Location loc = op.getLoc();
    SmallVector<OpFoldResult> iterationSizes =
        buildIterationSizes(rewriter, loc, op, detail);

    // Compute D = sum_k Aq * Bq using i32 arithmetic, where k spans the integer
    // reduction dims. Retain other reduction dims as parallel dims so each
    // partial result can later receive its own scales and zero-point
    // correction.
    linalg::GenericOp integerContraction =
        buildIntegerContraction(rewriter, loc, op, detail, iterationSizes);

    // Keep the partials' indexing map alongside their values so the epilogue
    // can align them with the correction and scales in the original loop space.
    IndexedValue partials{integerContraction.getResult(0),
                          integerContraction.getIndexingMapsArray().back()};

    // Compute correction = zB * sum_k Aq + zA * sum_k Bq - N * zA * zB,
    // where N is the product of the integer reduction extents. Subtracting it
    // from D gives sum_k (Aq - zA) * (Bq - zB); absent zero points contribute
    // zero.
    IndexedValue correction =
        buildZeroPointCorrection(rewriter, loc, detail, iterationSizes);

    // Subtract the correction in integer arithmetic before converting to float
    // to preserve cancellation in large partials. Apply both scales, sum any
    // remaining reduction dims in floating point, and convert to the original
    // output element type after accumulation.
    Value result = buildEpilogue(
        rewriter, loc, partials, correction,
        {detail.lhs.dequantize.getScale(), detail.lhs.scaleMap},
        {detail.rhs.dequantize.getScale(), detail.rhs.scaleMap},
        detail.outputMap, cast<RankedTensorType>(op.getDpsInits()[0].getType()),
        iterationSizes);

    // Redirect uses to the completed result and erase the original contraction.
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
