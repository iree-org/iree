// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-llvmgpu-legalize-nd-vectors"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPULEGALIZENDVECTORSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

/// Type converter that splits n-D vectors into multiple 1-D vectors, keeping
/// the innermost dimension intact.
struct NDVectorTypeConverter final : public TypeConverter {
  NDVectorTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> { return type; });

    addConversion([](VectorType type, SmallVectorImpl<Type> &types)
                      -> std::optional<LogicalResult> {
      if (type.getRank() <= 1) {
        types.push_back(type);
        return success();
      }
      int64_t innerDim = type.getShape().back();
      int64_t num1DVectors = type.getNumElements() / innerDim;
      Type innerDimVector = VectorType::get({innerDim}, type.getElementType());
      for (int64_t i = 0; i < num1DVectors; i++) {
        types.push_back(innerDimVector);
      }
      return success();
    });

    // Some ops (e.g. nvgpu.ldmatrix, nvgpu.mma.sync) abuse n-D vector types
    // to represent a "struct of vectors". When these ops interact with the
    // converted 1-D values, we need materializations to bridge the gap.
    // This is unfortunate — ops that want to represent structs should use
    // struct types, not n-D vectors.
    addSourceMaterialization([](OpBuilder &builder, VectorType targetType,
                                ValueRange inputs, Location loc) -> Value {
      Value result = ub::PoisonOp::create(builder, loc, targetType);
      SmallVector<int64_t> iteratorSpace(targetType.getShape().drop_back());
      for (auto [input, idx] : llvm::zip_equal(
               inputs, StaticTileOffsetRange(
                           iteratorSpace,
                           SmallVector<int64_t>(iteratorSpace.size(), 1)))) {
        result = vector::InsertOp::create(builder, loc, input, result, idx);
      }
      return result;
    });

    addTargetMaterialization([](OpBuilder &builder, TypeRange targetTypes,
                                ValueRange sources, Location loc,
                                Type originalType) -> SmallVector<Value> {
      Value source = sources[0];
      SmallVector<Value> results;
      VectorType originalVecType = cast<VectorType>(originalType);
      SmallVector<int64_t> iteratorSpace(
          originalVecType.getShape().drop_back());
      for (SmallVector<int64_t> idx : StaticTileOffsetRange(
               iteratorSpace, SmallVector<int64_t>(iteratorSpace.size(), 1))) {
        results.push_back(vector::ExtractOp::create(builder, loc, source, idx));
      }
      return results;
    });
  }
};

/// Unroll elementwise ops on n-D vectors into ops on 1-D vectors using the
/// type converter's 1:N mapping.
struct UnrollElementwiseOps final
    : public OpTraitConversionPattern<OpTrait::Elementwise> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasTrait<OpTrait::Elementwise>()) {
      return failure();
    }
    VectorType dstVecTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!dstVecTy || dstVecTy.getRank() <= 1) {
      return failure();
    }

    SmallVector<Type> convertedTypes;
    if (failed(getTypeConverter()->convertType(op->getResult(0).getType(),
                                               convertedTypes))) {
      return failure();
    }

    // Invert operands: from list-of-ValueRanges to N lists of operands.
    SmallVector<SmallVector<Value>> opOperands;
    for (auto i : llvm::seq<int64_t>(convertedTypes.size())) {
      SmallVector<Value> newOperands;
      for (ValueRange operandList : operands) {
        newOperands.push_back(operandList[i]);
      }
      opOperands.push_back(newOperands);
    }

    int64_t numResults = op->getNumResults();
    SmallVector<Type> clonedResultTypes(numResults);
    SmallVector<SmallVector<Value>> replacements(numResults);
    for (auto [inputs, convertedTy] :
         llvm::zip_equal(opOperands, convertedTypes)) {
      clonedResultTypes.assign(numResults, convertedTy);
      Operation *clonedOp = clone(rewriter, op, clonedResultTypes, inputs);
      for (auto j : llvm::seq<int64_t>(numResults)) {
        replacements[j].push_back(clonedOp->getResult(j));
      }
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

/// Convert vector.extract on n-D vectors. The source is already split into
/// flat 1-D vectors by the type converter; this pattern selects the right
/// slice (or scalar-extracts from a single 1-D vector).
struct ConvertVectorExtract final
    : public OpConversionPattern<vector::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = op.getSourceVectorType();
    if (srcType.getRank() <= 1) {
      return failure();
    }
    if (op.hasDynamicPosition()) {
      op.emitRemark()
          << "dynamic positions on extract are intentionally disabled";
      return failure();
    }

    ArrayRef<int64_t> staticPos = op.getStaticPosition();
    SmallVector<Value> srcValues(adaptor.getSource());
    ArrayRef<int64_t> outerShape = srcType.getShape().drop_back();
    SmallVector<int64_t> outerStrides = computeStrides(outerShape);

    int64_t numOuter = outerShape.size();
    int64_t numOuterIndices =
        std::min(static_cast<int64_t>(staticPos.size()), numOuter);
    bool hasInnerIndex = static_cast<int64_t>(staticPos.size()) > numOuter;

    int64_t offset = 0;
    for (int64_t i = 0; i < numOuterIndices; i++) {
      offset += staticPos[i] * outerStrides[i];
    }
    int64_t count = outerStrides[numOuterIndices - 1];

    if (hasInnerIndex) {
      // Scalar extraction: pick the 1-D vector, then extract the element.
      rewriter.replaceOp(op, vector::ExtractOp::create(rewriter, op.getLoc(),
                                                       srcValues[offset],
                                                       staticPos.back()));
    } else {
      // Sub-vector extraction: select a contiguous slice of converted values.
      SmallVector<Value> results(srcValues.begin() + offset,
                                 srcValues.begin() + offset + count);
      rewriter.replaceOpWithMultiple(op, {results});
    }
    return success();
  }
};

/// Convert vector.insert on n-D vectors. The dest is already split into flat
/// 1-D vectors by the type converter; this pattern replaces the right slice
/// (or scalar-inserts into a single 1-D vector).
struct ConvertVectorInsert final
    : public OpConversionPattern<vector::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType destType = op.getDestVectorType();
    if (destType.getRank() <= 1) {
      return failure();
    }
    if (op.hasDynamicPosition()) {
      return failure();
    }

    ArrayRef<int64_t> staticPos = op.getStaticPosition();
    SmallVector<Value> srcValues(adaptor.getValueToStore());
    SmallVector<Value> destValues(adaptor.getDest());
    ArrayRef<int64_t> outerShape = destType.getShape().drop_back();
    SmallVector<int64_t> outerStrides = computeStrides(outerShape);

    int64_t numOuter = outerShape.size();
    int64_t numOuterIndices =
        std::min(static_cast<int64_t>(staticPos.size()), numOuter);
    bool hasInnerIndex = static_cast<int64_t>(staticPos.size()) > numOuter;

    int64_t offset = 0;
    for (int64_t i = 0; i < numOuterIndices; i++) {
      offset += staticPos[i] * outerStrides[i];
    }
    int64_t count = outerStrides[numOuterIndices - 1];

    SmallVector<Value> results(destValues.begin(), destValues.end());
    if (hasInnerIndex) {
      // Scalar insertion into a specific 1-D vector.
      results[offset] =
          vector::InsertOp::create(rewriter, op.getLoc(), srcValues[0],
                                   destValues[offset], staticPos.back());
    } else {
      // Replace a contiguous slice with the converted source values.
      for (int64_t i = 0; i < count; i++) {
        results[offset + i] = srcValues[i];
      }
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert vector.transpose on n-D vectors. If the permutation does not
/// affect the inner dimension, this is just a reordering of the flat 1-D
/// vectors. Otherwise, decompose to scalars and reassemble via
/// vector.to_elements / vector.from_elements.
struct ConvertVectorTranspose final
    : public OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransposeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = op.getSourceVectorType();
    if (srcType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    SmallVector<Value> srcValues(adaptor.getVector());
    ArrayRef<int64_t> perm = op.getPermutation();
    ArrayRef<int64_t> srcShape = srcType.getShape();
    VectorType resultType = op.getResultVectorType();

    ArrayRef<int64_t> srcOuterShape = srcShape.drop_back();
    SmallVector<int64_t> srcOuterStrides = computeStrides(srcOuterShape);

    if (perm.back() == srcType.getRank() - 1) {
      // Inner dimension is not permuted — just reorder the 1-D vectors.
      SmallVector<int64_t> outerPerm(perm.drop_back());
      SmallVector<int64_t> invOuterPerm = invertPermutationVector(outerPerm);

      ArrayRef<int64_t> resultOuterShape = resultType.getShape().drop_back();
      SmallVector<Value> results;
      SmallVector<int64_t> tileShape(resultOuterShape.size(), 1);
      for (SmallVector<int64_t> resultIdx :
           StaticTileOffsetRange(resultOuterShape, tileShape)) {
        SmallVector<int64_t> srcIdx(srcOuterShape.size());
        for (size_t k = 0; k < srcIdx.size(); k++) {
          srcIdx[k] = resultIdx[invOuterPerm[k]];
        }

        int64_t srcLinear = 0;
        for (size_t k = 0; k < srcIdx.size(); k++) {
          srcLinear += srcIdx[k] * srcOuterStrides[k];
        }
        results.push_back(srcValues[srcLinear]);
      }
      rewriter.replaceOpWithMultiple(op, {results});
      return success();
    }

    // Inner dimension is permuted — decompose to scalars and reassemble.
    SmallVector<int64_t> invPerm = invertPermutationVector(perm);
    // Decompose each source 1-D vector into scalars.
    SmallVector<SmallVector<Value>> allScalars;
    for (Value srcVec : srcValues) {
      auto toElem = vector::ToElementsOp::create(rewriter, loc, srcVec);
      allScalars.push_back(SmallVector<Value>(toElem.getResults()));
    }

    ArrayRef<int64_t> resultShape = resultType.getShape();
    int64_t resultInnerDim = resultShape.back();
    ArrayRef<int64_t> resultOuterShape = resultShape.drop_back();
    auto resultElemVecType =
        VectorType::get({resultInnerDim}, srcType.getElementType());

    SmallVector<Value> results;
    SmallVector<int64_t> tileShape(resultOuterShape.size(), 1);
    for (SmallVector<int64_t> resultOuterIdx :
         StaticTileOffsetRange(resultOuterShape, tileShape)) {
      SmallVector<Value> elements;
      for (int64_t j = 0; j < resultInnerDim; j++) {
        // Full result index = (resultOuterIdx..., j).
        SmallVector<int64_t> resultFullIdx(resultOuterIdx);
        resultFullIdx.push_back(j);

        // Apply inverse permutation: src[k] = result[invPerm[k]].
        SmallVector<int64_t> srcFullIdx(srcShape.size());
        for (size_t k = 0; k < srcFullIdx.size(); k++) {
          srcFullIdx[k] = resultFullIdx[invPerm[k]];
        }

        // Split into source outer index and inner index.
        int64_t srcInner = srcFullIdx.back();
        int64_t srcOuterLinear = 0;
        for (size_t k = 0; k < srcOuterShape.size(); k++) {
          srcOuterLinear += srcFullIdx[k] * srcOuterStrides[k];
        }

        elements.push_back(allScalars[srcOuterLinear][srcInner]);
      }
      results.push_back(vector::FromElementsOp::create(
          rewriter, loc, resultElemVecType, elements));
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert vector.shape_cast on n-D vectors. If the inner dimension is the
/// same, just pass through the 1-D vectors. Otherwise, flatten all scalars
/// via vector.to_elements and regroup into new 1-D vectors via
/// vector.from_elements.
struct ConvertVectorShapeCast final
    : public OpConversionPattern<vector::ShapeCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ShapeCastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = op.getSourceVectorType();
    VectorType resultType = op.getResultVectorType();
    if (srcType.getRank() <= 1 && resultType.getRank() <= 1) {
      return failure();
    }

    SmallVector<Value> srcValues(adaptor.getSource());

    if (srcType.getShape().back() == resultType.getShape().back()) {
      // Inner dimension unchanged — the flat 1-D vectors are already correct,
      // just reinterpret the outer shape.
      rewriter.replaceOpWithMultiple(op, {srcValues});
      return success();
    }

    // Inner dimension changed — flatten to scalars and regroup.
    Location loc = op.getLoc();
    SmallVector<Value> allScalars;
    for (Value srcVec : srcValues) {
      auto toElem = vector::ToElementsOp::create(rewriter, loc, srcVec);
      llvm::append_range(allScalars, toElem.getResults());
    }

    int64_t resultInnerDim = resultType.getShape().back();
    auto resultElemVecType =
        VectorType::get({resultInnerDim}, resultType.getElementType());

    SmallVector<Value> results;
    for (int64_t i = 0, e = allScalars.size(); i < e; i += resultInnerDim) {
      ArrayRef<Value> chunk(allScalars.data() + i, resultInnerDim);
      results.push_back(vector::FromElementsOp::create(
          rewriter, loc, resultElemVecType, chunk));
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Helper to extract int64_t values from an I64ArrayAttr.
static SmallVector<int64_t> getI64ArrayAttrValues(ArrayAttr attr) {
  return llvm::map_to_vector(
      attr, [](Attribute a) { return cast<IntegerAttr>(a).getInt(); });
}

/// Convert vector.extract_strided_slice on n-D vectors. Selects the
/// appropriate sub-block of 1-D vectors, and if the inner dimension is also
/// sliced, creates 1-D extract_strided_slice ops.
struct ConvertVectorExtractStridedSlice final
    : public OpConversionPattern<vector::ExtractStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = op.getSourceVectorType();
    if (srcType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    SmallVector<Value> srcValues(adaptor.getSource());
    SmallVector<int64_t> offsets = getI64ArrayAttrValues(op.getOffsets());
    SmallVector<int64_t> sizes = getI64ArrayAttrValues(op.getSizes());
    ArrayRef<int64_t> srcShape = srcType.getShape();
    int64_t rank = srcType.getRank();
    int64_t k = offsets.size();

    // Extend offsets/sizes to cover all n dimensions.
    SmallVector<int64_t> fullOffsets(offsets);
    SmallVector<int64_t> fullSizes(sizes);
    for (int64_t i = k; i < rank; i++) {
      fullOffsets.push_back(0);
      fullSizes.push_back(srcShape[i]);
    }

    ArrayRef<int64_t> outerOffsets(fullOffsets.data(), rank - 1);
    ArrayRef<int64_t> outerSizes(fullSizes.data(), rank - 1);
    int64_t innerOffset = fullOffsets.back();
    int64_t innerSize = fullSizes.back();
    bool innerSliced = innerOffset != 0 || innerSize != srcShape.back();

    ArrayRef<int64_t> srcOuterShape = srcShape.drop_back();
    SmallVector<int64_t> srcOuterStrides = computeStrides(srcOuterShape);

    SmallVector<Value> results;
    SmallVector<int64_t> tileShape(outerSizes.size(), 1);
    for (SmallVector<int64_t> resultOuterIdx :
         StaticTileOffsetRange(outerSizes, tileShape)) {
      // Compute source outer index = result index + offset.
      int64_t srcLinear = 0;
      for (size_t i = 0; i < resultOuterIdx.size(); i++) {
        srcLinear += (resultOuterIdx[i] + outerOffsets[i]) * srcOuterStrides[i];
      }

      Value val = srcValues[srcLinear];
      if (innerSliced) {
        val = vector::ExtractStridedSliceOp::create(
            rewriter, loc, val,
            /*offsets=*/ArrayRef<int64_t>{innerOffset},
            /*sizes=*/ArrayRef<int64_t>{innerSize},
            /*strides=*/ArrayRef<int64_t>{1});
      }
      results.push_back(val);
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert vector.insert_strided_slice on n-D vectors. For each source 1-D
/// vector, computes the corresponding dest 1-D vector position and either
/// replaces it directly or creates a 1-D insert_strided_slice.
struct ConvertVectorInsertStridedSlice final
    : public OpConversionPattern<vector::InsertStridedSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::InsertStridedSliceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType destType = op.getDestVectorType();
    if (destType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    SmallVector<Value> srcValues(adaptor.getValueToStore());
    SmallVector<Value> destValues(adaptor.getDest());
    SmallVector<int64_t> offsets = getI64ArrayAttrValues(op.getOffsets());
    VectorType srcType = op.getSourceVectorType();

    int64_t n = destType.getRank();
    int64_t k = srcType.getRank();
    int64_t nk = n - k; // number of leading dest dims not covered by source.

    ArrayRef<int64_t> destOuterShape = destType.getShape().drop_back();
    SmallVector<int64_t> destOuterStrides = computeStrides(destOuterShape);
    ArrayRef<int64_t> srcOuterShape = srcType.getShape().drop_back();

    int64_t innerOffset = offsets.back();
    bool innerSliced =
        srcType.getShape().back() != destType.getShape().back() ||
        innerOffset != 0;

    SmallVector<Value> results(destValues.begin(), destValues.end());
    int64_t srcFlatIdx = 0;
    SmallVector<int64_t> tileShape(srcOuterShape.size(), 1);
    for (SmallVector<int64_t> srcOuterIdx :
         StaticTileOffsetRange(srcOuterShape, tileShape)) {
      // Build dest outer index: leading dims from offsets, trailing from
      // source index + offset.
      SmallVector<int64_t> destOuterIdx;
      for (int64_t i = 0; i < nk; i++) {
        destOuterIdx.push_back(offsets[i]);
      }
      for (size_t i = 0; i < srcOuterIdx.size(); i++) {
        destOuterIdx.push_back(offsets[nk + i] + srcOuterIdx[i]);
      }

      int64_t destLinear = 0;
      for (size_t i = 0; i < destOuterIdx.size(); i++) {
        destLinear += destOuterIdx[i] * destOuterStrides[i];
      }

      if (innerSliced) {
        results[destLinear] = vector::InsertStridedSliceOp::create(
            rewriter, loc, srcValues[srcFlatIdx], destValues[destLinear],
            /*offsets=*/ArrayRef<int64_t>{innerOffset},
            /*strides=*/ArrayRef<int64_t>{1});
      } else {
        results[destLinear] = srcValues[srcFlatIdx];
      }
      srcFlatIdx++;
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Split an arith.constant producing an n-D vector into multiple 1-D
/// vector constants.
struct ConvertArithConstant final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<VectorType>(op.getType());
    if (!vecType || vecType.getRank() <= 1) {
      return failure();
    }

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr) {
      return failure();
    }

    Location loc = op.getLoc();
    int64_t innerDim = vecType.getShape().back();
    int64_t numVectors = vecType.getNumElements() / innerDim;
    auto vec1DType = VectorType::get({innerDim}, vecType.getElementType());

    SmallVector<Value> results;
    if (denseAttr.isSplat()) {
      auto splatAttr = DenseElementsAttr::get(
          vec1DType, denseAttr.getSplatValue<Attribute>());
      for (int64_t i = 0; i < numVectors; i++) {
        results.push_back(arith::ConstantOp::create(rewriter, loc, splatAttr));
      }
    } else {
      auto values = llvm::to_vector(denseAttr.getValues<Attribute>());
      for (int64_t i = 0; i < numVectors; i++) {
        ArrayRef<Attribute> chunk(values.data() + i * innerDim, innerDim);
        auto chunkAttr = DenseElementsAttr::get(vec1DType, chunk);
        results.push_back(arith::ConstantOp::create(rewriter, loc, chunkAttr));
      }
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Split a ub.poison producing an n-D vector into multiple 1-D poison values.
struct ConvertUBPoison final : public OpConversionPattern<ub::PoisonOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ub::PoisonOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<VectorType>(op.getType());
    if (!vecType || vecType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    int64_t innerDim = vecType.getShape().back();
    int64_t numVectors = vecType.getNumElements() / innerDim;
    auto vec1DType = VectorType::get({innerDim}, vecType.getElementType());

    SmallVector<Value> results;
    for (int64_t i = 0; i < numVectors; i++) {
      results.push_back(ub::PoisonOp::create(rewriter, loc, vec1DType));
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert vector.to_elements on an n-D vector. The source is already split
/// into 1-D vectors; decompose each with a 1-D to_elements.
struct ConvertVectorToElements final
    : public OpConversionPattern<vector::ToElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ToElementsOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getSource().getType();
    if (srcType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    SmallVector<Value> srcValues(adaptor.getSource());

    SmallVector<Value> allScalars;
    for (Value src1D : srcValues) {
      auto toElem = vector::ToElementsOp::create(rewriter, loc, src1D);
      llvm::append_range(allScalars, toElem.getResults());
    }
    rewriter.replaceOp(op, allScalars);
    return success();
  }
};

/// Convert vector.from_elements producing an n-D vector. Chunk the scalar
/// inputs into groups of innerDim and create 1-D from_elements for each.
struct ConvertVectorFromElements final
    : public OpConversionPattern<vector::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FromElementsOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = op.getDest().getType();
    if (resultType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    // Flatten all adapted scalar operands.
    SmallVector<Value> elements;
    for (ValueRange vr : adaptor.getElements()) {
      llvm::append_range(elements, vr);
    }

    int64_t innerDim = resultType.getShape().back();
    auto vec1DType = VectorType::get({innerDim}, resultType.getElementType());

    SmallVector<Value> results;
    for (int64_t i = 0, e = elements.size(); i < e; i += innerDim) {
      ArrayRef<Value> chunk(elements.data() + i, innerDim);
      results.push_back(
          vector::FromElementsOp::create(rewriter, loc, vec1DType, chunk));
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert vector.broadcast from a scalar/0-D source to an n-D result.
/// Broadcasts the source to each result 1-D vector.
struct ConvertVectorBroadcast final
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = op.getResultVectorType();
    if (resultType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    Value src = adaptor.getSource()[0];
    int64_t innerDim = resultType.getShape().back();
    int64_t numVectors = resultType.getNumElements() / innerDim;
    auto vec1DType = VectorType::get({innerDim}, resultType.getElementType());

    Value broadcasted =
        vector::BroadcastOp::create(rewriter, loc, vec1DType, src);
    SmallVector<Value> results(numVectors, broadcasted);
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert vector.bitcast on n-D vectors. Since bitcast only affects the
/// innermost dimension, just bitcast each 1-D vector individually.
struct ConvertVectorBitcast final
    : public OpConversionPattern<vector::BitCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BitCastOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = op.getSourceVectorType();
    if (srcType.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    SmallVector<Value> srcValues(adaptor.getSource());
    VectorType resultType = op.getResultVectorType();
    auto result1DType = VectorType::get({resultType.getShape().back()},
                                        resultType.getElementType());

    SmallVector<Value> results;
    for (Value src : srcValues) {
      results.push_back(
          vector::BitCastOp::create(rewriter, loc, result1DType, src));
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Convert any ReturnLike op with 1:N type-converted operands.
struct ConvertReturnLike final
    : public OpTraitConversionPattern<OpTrait::ReturnLike> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasTrait<OpTrait::ReturnLike>()) {
      return failure();
    }
    SmallVector<Value> flatOperands;
    for (ValueRange vals : operands) {
      llvm::append_range(flatOperands, vals);
    }
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(flatOperands); });
    return success();
  }
};

struct ConvertTransferRead final
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultTy = op.getVectorType();
    if (resultTy.getRank() <= 1) {
      return failure();
    }

    Location loc = op.getLoc();
    ArrayRef<int64_t> shape = resultTy.getShape();
    ArrayRef<int64_t> outerShape = shape.drop_back();
    int64_t numOuterDims = outerShape.size();
    auto vec1DType = VectorType::get({shape.back()}, resultTy.getElementType());

    // New permutation map: keep only the last result (innermost vector dim).
    AffineMap permMap = op.getPermutationMap();
    AffineMap newPermMap =
        AffineMap::get(permMap.getNumDims(), permMap.getNumSymbols(),
                       {permMap.getResults().back()}, rewriter.getContext());

    // New in_bounds: keep only the innermost entry.
    ArrayAttr newInBoundsAttr;
    if (ArrayAttr inBounds = op.getInBoundsAttr()) {
      newInBoundsAttr = rewriter.getArrayAttr({inBounds.getValue().back()});
    }

    ValueRange convertedMask = adaptor.getMask();

    int32_t idx = 0;
    SmallVector<Value> results;
    SmallVector<int64_t> tileShape(numOuterDims, 1);
    for (SmallVector<int64_t> outerIdx :
         StaticTileOffsetRange(outerShape, tileShape)) {
      // Copy the original indices so we can adjust per-iteration.
      SmallVector<Value> newIndices(op.getIndices());
      for (int64_t d = 0; d < numOuterDims; ++d) {
        auto dimExpr = dyn_cast<AffineDimExpr>(permMap.getResult(d));
        if (!dimExpr || outerIdx[d] == 0) {
          continue;
        }
        int64_t memrefDim = dimExpr.getPosition();
        Value offset =
            arith::ConstantIndexOp::create(rewriter, loc, outerIdx[d]);
        newIndices[memrefDim] =
            arith::AddIOp::create(rewriter, loc, newIndices[memrefDim], offset);
      }

      Value newMask = convertedMask.empty() ? Value{} : convertedMask[idx++];

      auto readOp = vector::TransferReadOp::create(
          rewriter, loc, vec1DType, op.getBase(), newIndices,
          AffineMapAttr::get(newPermMap), op.getPadding(), newMask,
          newInBoundsAttr);

      results.push_back(readOp);
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

struct LLVMGPULegalizeNDVectorsPass final
    : impl::LLVMGPULegalizeNDVectorsPassBase<LLVMGPULegalizeNDVectorsPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    NDVectorTypeConverter typeConverter;
    ConversionTarget target(*ctx);

    scf::populateSCFStructuralTypeConversionTarget(typeConverter, target);
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                        typeConverter);
    patterns.add<UnrollElementwiseOps, ConvertReturnLike>(typeConverter, ctx);
    patterns.add<
        ConvertVectorExtract, ConvertVectorInsert, ConvertVectorTranspose,
        ConvertVectorShapeCast, ConvertVectorExtractStridedSlice,
        ConvertVectorInsertStridedSlice, ConvertArithConstant, ConvertUBPoison,
        ConvertVectorToElements, ConvertVectorFromElements,
        ConvertVectorBroadcast, ConvertVectorBitcast,
	ConvertTransferRead>(typeConverter, ctx);

    // Some nvgpu ops abuse n-D vector types to represent a "struct of
    // vectors". These ops are legal despite having n-D vectors — the
    // materializations above bridge the gap. This is unfortunate; ops that
    // want to represent structs should use struct types, not n-D vectors.
    target.addLegalOp<nvgpu::LdMatrixOp, nvgpu::MmaSyncOp>();

    auto hasNDVector = [](TypeRange types) {
      return llvm::any_of(types, [](Type t) {
        auto vecType = dyn_cast<VectorType>(t);
        return vecType && vecType.getRank() > 1;
      });
    };
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        return typeConverter.isSignatureLegal(
            cast<FunctionType>(funcOp.getFunctionType()));
      }
      return !hasNDVector(op->getOperandTypes()) &&
             !hasNDVector(op->getResultTypes());
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
