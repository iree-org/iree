// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
// TODO(benvanik): have a stream/upstream equivalent of the flow.dispatch.* ops.
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/ElementPackingUtils.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ENCODEHOSTTENSORSPASS
#define GEN_PASS_DEF_ENCODEDEVICETENSORSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Encoding utilities
//===----------------------------------------------------------------------===//

// Asserts that the given encoding is supported by this code right now.
// Non-trivial dense tensor encodings need special handling.
static LogicalResult checkEncoding(Operation *op, RankedTensorType encodingType,
                                   ValueRange encodingDims,
                                   PatternRewriter &rewriter) {
  auto encoding = encodingType.getEncoding();
  if (encoding && !llvm::isa<IREE::Encoding::EncodingAttr,
                             IREE::Encoding::PackedStorageAttr>(encoding)) {
    return rewriter.notifyMatchFailure(op, [=](Diagnostic &d) {
      d << "unsupported tensor encoding: " << encodingType;
    });
  }
  return success();
}

// Aligns the element type of a tensor<> to a byte-aligned power of 2 bit width.
static RankedTensorType alignTensorType(RankedTensorType originalType) {
  Type elementType = originalType.getElementType();
  Type alignedType = legalizeStorageElementType(elementType);
  if (alignedType == elementType)
    return originalType;
  return RankedTensorType::get(originalType.getShape(), alignedType,
                               originalType.getEncoding());
}

// Returns a ConstantIndexOp with the value of the given dimension.
static Value makeTensorDim(Location loc, RankedTensorType tensorType,
                           ValueRange dynamicDims, unsigned i,
                           PatternRewriter &rewriter) {
  // Static dimension early-out:
  if (!tensorType.isDynamicDim(i)) {
    return rewriter.create<arith::ConstantIndexOp>(loc,
                                                   tensorType.getDimSize(i));
  }

  // Map from absolute dimension index to the compact dynamic index.
  unsigned di = 0;
  for (unsigned j = 0; j < i; ++j) {
    if (tensorType.isDynamicDim(j))
      ++di;
  }
  return dynamicDims[di];
}

// Returns an element offset within a dense tensor based on indices.
// TODO(benvanik): when partially static try to avoid emitting so much IR.
static Value calculateElementOffset(Location loc, RankedTensorType tensorType,
                                    ValueRange dynamicDims, ValueRange indices,
                                    PatternRewriter &rewriter) {
  assert(indices.size() == tensorType.getRank());
  auto offset = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  for (size_t i = 0; i < indices.size(); ++i) {
    auto axisOffset = indices[i];
    for (size_t j = i + 1; j < tensorType.getRank(); ++j) {
      auto axisDim = makeTensorDim(loc, tensorType, dynamicDims, j, rewriter);
      axisOffset =
          rewriter.createOrFold<arith::MulIOp>(loc, axisOffset, axisDim);
    }
    offset = rewriter.createOrFold<arith::AddIOp>(loc, offset, axisOffset);
  }
  return offset;
}

// Returns an element offset within a dense tensor based on indices, in bytes.
static Value calculateElementByteOffset(Location loc,
                                        RankedTensorType tensorType,
                                        ValueRange dynamicDims,
                                        ValueRange indices,
                                        PatternRewriter &rewriter) {
  Value linearizedIndex =
      calculateElementOffset(loc, tensorType, dynamicDims, indices, rewriter);
  return calculateStorageElementOffsetInBytes(loc, tensorType, linearizedIndex,
                                              rewriter);
}

// Canonicalizes a fill pattern into a power of 2 byte-aligned integer type.
// The stream dialect splat/fill ops require one of I8, I16, or I32 - any other
// type must be converted to one of those here. This prevents encoding policy
// such as what to do with i1 or float types from leaking into lower levels of
// the stack: fill ops are just setting bytes.
//
// The other reason to handle things here is that the fill pattern must be
// <= 32-bits - if it's over that we need to insert a dispatch to perform the
// fill and the only time we can do that in the pipeline is here.
// This is a somewhat annoying abstraction leak from the HAL which also has a
// 32-bit fill limit, but that is an abstraction leak from the underlying APIs
// and hardware (Metal/Vulkan/CUDA/etc) that also don't have 64-bit fills.
// Instead of forcing all runtime implementations to include emulation for
// 64-bit fills we take care of that here on an as-needed basis.
//
// Returns the pattern converted to one of [i8, i16, i32, i64] (with i64 needing
// to be handled via emulation) or nullptr if the type is unsupported.
static Value canonicalizeFillPattern(Value pattern, OpBuilder &builder) {
  auto loc = pattern.getLoc();

  // Decompose complex numbers into the real/imag components and pack into an
  // int. Note that this only works for 32-bit complex types today.
  if (auto complexType = dyn_cast<ComplexType>(pattern.getType())) {
    unsigned elementBitWidth =
        complexType.getElementType().getIntOrFloatBitWidth();
    assert(elementBitWidth <= 32 && "unsupported complex<f64>");
    Type bwType = builder.getIntegerType(elementBitWidth * 2);
    return builder.create<complex::BitcastOp>(loc, bwType, pattern);
  }

  // Get floats into integer form first; may need additional processing below.
  if (auto floatType = dyn_cast<FloatType>(pattern.getType())) {
    pattern = builder.createOrFold<arith::BitcastOp>(
        loc, builder.getIntegerType(floatType.getIntOrFloatBitWidth()),
        pattern);
  }

  // HACK: extend i1 to i8. This is really not something we should be doing here
  // in optimized programs as this is a super shady operation.
  unsigned elementBitWidth = IREE::Util::getTypeBitWidth(pattern.getType());
  if (elementBitWidth == 1) {
    return builder.createOrFold<arith::ExtUIOp>(loc, builder.getI8Type(),
                                                pattern);
  }

  // For packed sub-byte patterns, duplicate the sub-byte parts into a full
  // byte. We first extend the sub-byte parts into full bytes, and then keep
  // shifting left and bitwise or the sub-byte parts. For example, to create an
  // i8 pattern from i2 parts, we generate the following sequence:
  //   %i8_val = arith.extui %i2_val
  //   %i8_val = (%i8_val << 2) | %i2_val
  //   %i8_val = (%i8_val << 2) | %i2_val
  //   %i8_val = (%i8_val << 2) | %i2_val
  if (needToPackSubByteElementBitWidth(elementBitWidth)) {
    Type i8Type = builder.getI8Type();
    Value bitwidth = builder.createOrFold<arith::ConstantOp>(
        loc, i8Type, builder.getIntegerAttr(i8Type, elementBitWidth));
    Value subByteVal =
        builder.createOrFold<arith::ExtUIOp>(loc, i8Type, pattern);
    Value i8Val = subByteVal;
    for (unsigned i = 1, e = 8 / elementBitWidth; i < e; ++i) {
      Value shifted = builder.createOrFold<arith::ShLIOp>(loc, i8Val, bitwidth);
      i8Val = builder.createOrFold<arith::OrIOp>(loc, shifted, subByteVal);
    }
    return i8Val;
  }
  if ((elementBitWidth % 8) != 0) {
    // We'd need some policy to determine how to handle non-byte-aligned widths.
    return {};
  }

  // 8/16/32/64-bit value pass through (possibly after a bitcast).
  return pattern;
}

//===----------------------------------------------------------------------===//
// stream.tensor.import
//===----------------------------------------------------------------------===//

struct EncodeTensorImportOp
    : public OpRewritePattern<IREE::Stream::TensorImportOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorImportOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResultEncoding());
    auto resultDims = op.getResultEncodingDims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // TODO(benvanik): decompose this into a conditional or call to a transfer
    // utility function. Want to compare the source type (somehow) and then
    // clone or directly use the input somehow. For now we punt to HAL.

    return rewriter.notifyMatchFailure(op, "tensor import not handled");
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.export
//===----------------------------------------------------------------------===//

struct EncodeTensorExportOp
    : public OpRewritePattern<IREE::Stream::TensorExportOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorExportOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = llvm::cast<RankedTensorType>(op.getSourceEncoding());
    auto sourceDims = op.getSourceEncodingDims();
    if (failed(checkEncoding(op, sourceType, sourceDims, rewriter))) {
      return failure();
    }

    // TODO(benvanik): decompose this into a conditional or call to a transfer
    // utility function. Want to compare the source type (somehow) and then
    // clone or directly use the input somehow. For now we punt to HAL.

    return rewriter.notifyMatchFailure(op, "tensor export not handled");
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.sizeof
//===----------------------------------------------------------------------===//

struct EncodeTensorSizeOfOp
    : public OpRewritePattern<IREE::Stream::TensorSizeOfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorSizeOfOp op,
                                PatternRewriter &rewriter) const override {
    auto encodingType = llvm::cast<RankedTensorType>(op.getEncoding());
    auto encodingDims = op.getEncodingDims();
    if (failed(checkEncoding(op, encodingType, encodingDims, rewriter))) {
      return failure();
    }

    // Dense: element count * element size.
    Value totalSize = calculateStorageElementCountInBytes(
        op.getLoc(), encodingType, encodingDims, rewriter);
    if (!totalSize) {
      return op.emitOpError("failed to calculate total byte count: ")
             << encodingType << " does not have integral number of total bytes";
    }
    rewriter.replaceOp(op, totalSize);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.empty
//===----------------------------------------------------------------------===//

struct EncodeTensorEmptyOp
    : public OpRewritePattern<IREE::Stream::TensorEmptyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorEmptyOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResultEncoding());
    auto resultDims = op.getResultEncodingDims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // Dense:
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncAllocaOp>(
        op, op.getResult().getType(), op.getResultSize(), op.getAffinityAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.constant
//===----------------------------------------------------------------------===//

struct EncodeTensorConstantOp
    : public OpRewritePattern<IREE::Stream::TensorConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResultEncoding());
    auto resultDims = op.getResultEncodingDims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // TODO(benvanik): compute the size based on the contents of the elements
    // and perform arbitrary unpacking logic here, such as doing partial splats/
    // scatters/etc ala run-length-encoding. Lots of models have constants that
    // are very low entropy and instead of a compression algorithm a simple RLE
    // may be enough - even if just for the suffix.

    // TODO(benvanik): bit pack and emit a __builtin_zext_i1_i8 builtin.
    // Really we should be doing bitpacking at the flow/linalg level - doing it
    // here only saves us file size as we'd have to allocate the extended memory
    // and keep it around. If we see models with large unaligned constants we
    // can make the tradeoff for minimizing file size vs minimizing startup
    // cost.

    // Sub-byte aligned constants, if not explicitly allowed, need to be
    // expanded to a power of 2 byte-aligned width. This is unfortunate: it's
    // wasted bits in the final binary that we could otherwise use productively.
    RankedTensorType alignedType = alignTensorType(resultType);
    Attribute encodedAttr = op.getValue();
    if (alignedType != resultType) {
      if (auto sourceAttr = llvm::dyn_cast<DenseIntElementsAttr>(encodedAttr)) {
        auto alignedBitWidth = alignedType.getElementTypeBitWidth();
        encodedAttr = sourceAttr.mapValues(
            alignedType.getElementType(), [=](APInt sourceValue) {
              // NOTE: this is super slow! We should be doing a conversion in
              // a loop ourselves - don't want to be mapping for millions of
              // elements.
              return sourceValue.zext(alignedBitWidth);
            });
      }
    }

    // Dense:
    Value resultSize = calculateStorageElementCountInBytes(
        op.getLoc(), alignedType, resultDims, rewriter);
    if (!resultSize) {
      return op.emitOpError("failed to calculate total byte count: ")
             << alignedType << " does not have integral number of total bytes";
    }
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncConstantOp>(
        op, op.getResult().getType(), encodedAttr, resultSize,
        op.getAffinityAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.splat
//===----------------------------------------------------------------------===//

struct EncodeTensorSplatOp
    : public OpRewritePattern<IREE::Stream::TensorSplatOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorSplatOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResultEncoding());
    auto resultDims = op.getResultEncodingDims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // Canonicalize the fill pattern into an integer type [i8, i16, i32, i64].
    auto pattern = canonicalizeFillPattern(op.getValue(), rewriter);
    if (!pattern) {
      return op.emitOpError()
             << "has unsupported pattern type " << op.getValue().getType()
             << " with too large/non-byte-aligned bit width; a new builtin is "
                "required";
    }

    // Dense:
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncSplatOp>(
        op, op.getResult().getType(), pattern, op.getResultSize(),
        op.getAffinityAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.clone
//===----------------------------------------------------------------------===//

struct EncodeTensorCloneOp
    : public OpRewritePattern<IREE::Stream::TensorCloneOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorCloneOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = llvm::cast<RankedTensorType>(op.getSourceEncoding());
    auto sourceDims = op.getSourceEncodingDims();
    if (failed(checkEncoding(op, sourceType, sourceDims, rewriter))) {
      return failure();
    }
    auto resultType = llvm::cast<RankedTensorType>(op.getResultEncoding());
    auto resultDims = op.getResultEncodingDims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // Dense:
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCloneOp>(
        op, op.getResult().getType(), op.getSource(), op.getSourceSize(),
        op.getResultSize(), op.getAffinityAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.slice
//===----------------------------------------------------------------------===//

struct EncodeTensorSliceOp
    : public OpRewritePattern<IREE::Stream::TensorSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = llvm::cast<RankedTensorType>(op.getSourceEncoding());
    auto sourceDims = op.getSourceEncodingDims();
    if (failed(checkEncoding(op, sourceType, sourceDims, rewriter))) {
      return failure();
    }
    auto resultType = llvm::cast<RankedTensorType>(op.getResultEncoding());
    auto resultDims = op.getResultEncodingDims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // Dense:
    auto sourceOffset = calculateElementByteOffset(
        op.getLoc(), sourceType, sourceDims, op.getStartIndices(), rewriter);
    auto sourceEnd = rewriter.createOrFold<arith::AddIOp>(
        op.getLoc(), sourceOffset, op.getResultSize());
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncSliceOp>(
        op, op.getResult().getType(), op.getSource(), op.getSourceSize(),
        sourceOffset, sourceEnd, op.getResultSize(), op.getAffinityAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.fill
//===----------------------------------------------------------------------===//

struct EncodeTensorFillOp
    : public OpRewritePattern<IREE::Stream::TensorFillOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorFillOp op,
                                PatternRewriter &rewriter) const override {
    auto targetType = llvm::cast<RankedTensorType>(op.getTargetEncoding());
    auto targetDims = op.getTargetEncodingDims();
    if (failed(checkEncoding(op, targetType, targetDims, rewriter))) {
      return failure();
    }

    // Canonicalize the fill pattern into an integer type [i8, i16, i32, i64].
    auto pattern = canonicalizeFillPattern(op.getValue(), rewriter);
    if (!pattern) {
      return op.emitOpError()
             << "has unsupported pattern type " << op.getValue().getType()
             << " with too large/non-byte-aligned bit width; a new builtin is "
                "required";
    }

    // Dense:
    auto targetOffset = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.getStartIndices(), rewriter);
    auto targetLength = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.getLengths(), rewriter);
    auto targetEnd = rewriter.createOrFold<arith::AddIOp>(
        op.getLoc(), targetOffset, targetLength);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncFillOp>(
        op, op.getResult().getType(), op.getTarget(), op.getTargetSize(),
        targetOffset, targetEnd, targetLength, pattern, op.getAffinityAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.update
//===----------------------------------------------------------------------===//

struct EncodeTensorUpdateOp
    : public OpRewritePattern<IREE::Stream::TensorUpdateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorUpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto updateType = llvm::cast<RankedTensorType>(op.getUpdateEncoding());
    auto updateDims = op.getUpdateEncodingDims();
    if (failed(checkEncoding(op, updateType, updateDims, rewriter))) {
      return failure();
    }
    auto targetType = llvm::cast<RankedTensorType>(op.getTargetEncoding());
    auto targetDims = op.getTargetEncodingDims();
    if (failed(checkEncoding(op, targetType, targetDims, rewriter))) {
      return failure();
    }

    // Dense:
    auto targetOffset = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.getStartIndices(), rewriter);
    auto targetEnd = rewriter.createOrFold<arith::AddIOp>(
        op.getLoc(), targetOffset, op.getUpdateSize());
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncUpdateOp>(
        op, op.getResult().getType(), op.getTarget(), op.getTargetSize(),
        targetOffset, targetEnd, op.getUpdate(), op.getUpdateSize(),
        op.getAffinityAttr());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.load
//===----------------------------------------------------------------------===//

struct EncodeTensorLoadOp
    : public OpRewritePattern<IREE::Stream::TensorLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = llvm::cast<RankedTensorType>(op.getSourceEncoding());
    auto sourceDims = op.getSourceEncodingDims();
    auto loadType = op.getResult().getType();
    if (auto complexTy = dyn_cast<ComplexType>(loadType)) {
      loadType = IntegerType::get(
          loadType.getContext(),
          complexTy.getElementType().getIntOrFloatBitWidth() * 2);
    }
    if (failed(checkEncoding(op, sourceType, sourceDims, rewriter))) {
      return failure();
    }

    if (needToPackSubByteElements(sourceType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported load with sub-byte elements");
    }

    // Dense:
    auto sourceOffset = calculateElementByteOffset(
        op.getLoc(), sourceType, sourceDims, op.getIndices(), rewriter);
    Value load = rewriter.create<IREE::Stream::AsyncLoadOp>(
        op.getLoc(), loadType, op.getSource(), op.getSourceSize(),
        sourceOffset);

    if (loadType != op.getType()) {
      load =
          rewriter.create<complex::BitcastOp>(op.getLoc(), op.getType(), load);
    }

    rewriter.replaceOp(op, load);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.store
//===----------------------------------------------------------------------===//

struct EncodeTensorStoreOp
    : public OpRewritePattern<IREE::Stream::TensorStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto targetType = llvm::cast<RankedTensorType>(op.getTargetEncoding());
    auto targetDims = op.getTargetEncodingDims();
    if (failed(checkEncoding(op, targetType, targetDims, rewriter))) {
      return failure();
    }

    if (needToPackSubByteElements(targetType)) {
      return rewriter.notifyMatchFailure(
          op, "unsupported store with sub-byte elements");
    }

    // Dense:
    auto targetOffset = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncStoreOp>(
        op, op.getTarget(), op.getTargetSize(), targetOffset, op.getValue());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// stream.tensor.dispatch
//===----------------------------------------------------------------------===//

struct EncodeTensorDispatchOp
    : public OpRewritePattern<IREE::Stream::TensorDispatchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorDispatchOp op,
                                PatternRewriter &rewriter) const override {
    // Strip off the tensor encoding information - it's not used at all here. If
    // we changed the tensor dispatch op to accept indices and lengths for
    // offsetting we would need to account for that here but today we require
    // that to happen on slices/updates instead.
    Value zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    SmallVector<Value> operandOffsets;
    SmallVector<Value> operandEnds;
    SmallVector<Value> operandLengths;
    auto operandSizes = op.getOperandSizes();
    for (auto operand : op.getMixedOperands()) {
      if (isa<IREE::Stream::ResourceType>(operand.getType())) {
        operandOffsets.push_back(zeroOffset);
        operandEnds.push_back(operandSizes.front());
        operandLengths.push_back(operandSizes.front());
        operandSizes = operandSizes.drop_front(1);
      }
    }
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncDispatchOp>(
        op, op.getResultTypes(), op.getWorkload(), op.getEntryPointsAttr(),
        op.getMixedOperands(), op.getOperandSizes(), operandOffsets,
        operandEnds, operandLengths, op.getResultSizes(),
        op.getTiedOperandsAttr(), op.getAffinityAttr());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-encode-host-tensors
//===----------------------------------------------------------------------===//

struct EncodeHostTensorsPass
    : public IREE::Stream::impl::EncodeHostTensorsPassBase<
          EncodeHostTensorsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<
        EncodeTensorImportOp, EncodeTensorExportOp, EncodeTensorSizeOfOp,
        EncodeTensorEmptyOp, EncodeTensorConstantOp, EncodeTensorSplatOp,
        EncodeTensorCloneOp, EncodeTensorSliceOp, EncodeTensorFillOp,
        EncodeTensorUpdateOp, EncodeTensorLoadOp, EncodeTensorStoreOp,
        EncodeTensorDispatchOp>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// stream.binding.subspan
//===----------------------------------------------------------------------===//

// Aligns the element type of a !flow.dispatch.tensor<> to a byte-aligned power
// of 2 bit width.
static IREE::Flow::DispatchTensorType
alignDispatchTensorType(IREE::Flow::DispatchTensorType originalType) {
  Type elementType = originalType.getBoundElementType();
  Type alignedType = legalizeStorageElementType(elementType);
  if (alignedType == elementType)
    return originalType;
  return IREE::Flow::DispatchTensorType::get(
      originalType.getAccess(), originalType.getShape(), alignedType);
}

// Aligns binding element types to power-of-two byte boundaries.
// The loads and stores to the binding will need to be updated to perform the
// truncation and extension as required.
//
// We could do more handling here; today we are just doing sub-byte alignment
// conversion to ensure both host and device agree upon the number of bytes in
// a resource.
struct EncodeBindingSubspanOp
    : public OpRewritePattern<IREE::Stream::BindingSubspanOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::BindingSubspanOp op,
                                PatternRewriter &rewriter) const override {
    auto originalType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
        op.getResult().getType());
    if (!originalType) {
      return rewriter.notifyMatchFailure(op, "binding type not supported");
    }

    // Align the element type, if needed.
    IREE::Flow::DispatchTensorType alignedType =
        alignDispatchTensorType(originalType);
    if (originalType == alignedType)
      return failure(); // already aligned.

    // Directly swap the type with the one, changing all uses in the IR.
    // This works because
    rewriter.modifyOpInPlace(op,
                             [&]() { op.getResult().setType(alignedType); });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.load
//===----------------------------------------------------------------------===//

struct EncodeDispatchTensorLoadOp
    : public OpRewritePattern<IREE::Flow::DispatchTensorLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto targetType = llvm::cast<RankedTensorType>(op.getResult().getType());

    // Align the element type, if needed.
    RankedTensorType alignedType = alignTensorType(targetType);
    if (targetType == alignedType)
      return failure(); // already aligned.

    // Loads always truncate from an byte aligned type to a sub-byte one.
    assert(targetType.getElementTypeBitWidth() <
               alignedType.getElementTypeBitWidth() &&
           "loads must truncate");

    // Truncate the byte -> sub-byte type; e.g. i8 -> i1.
    auto loadedValue = op.getResult();
    rewriter.setInsertionPointAfterValue(loadedValue);
    auto truncOp =
        rewriter.create<arith::TruncIOp>(op.getLoc(), targetType, loadedValue);
    rewriter.modifyOpInPlace(op, [&]() {
      loadedValue.replaceAllUsesExcept(truncOp, truncOp);
      loadedValue.setType(alignedType);
    });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// flow.dispatch.tensor.store
//===----------------------------------------------------------------------===//

struct EncodeDispatchTensorStoreOp
    : public OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceType = llvm::cast<RankedTensorType>(op.getValue().getType());

    // Align the element type, if needed.
    RankedTensorType alignedType = alignTensorType(sourceType);
    if (sourceType == alignedType)
      return failure(); // already aligned.

    // Stores always extend from a sub-byte aligned type to a byte aligned one.
    assert(sourceType.getElementTypeBitWidth() <
               alignedType.getElementTypeBitWidth() &&
           "stores must extend");

    // Extend the sub-byte -> byte type; e.g. i1 -> i8.
    auto extOp = rewriter.create<arith::ExtUIOp>(op.getLoc(), alignedType,
                                                 op.getValue());
    rewriter.modifyOpInPlace(
        op, [&]() { op.getValueMutable().assign(extOp.getResult()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-encode-device-tensors
//===----------------------------------------------------------------------===//

struct EncodeDeviceTensorsPass
    : public IREE::Stream::impl::EncodeDeviceTensorsPassBase<
          EncodeDeviceTensorsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<EncodeBindingSubspanOp, EncodeDispatchTensorLoadOp,
                    EncodeDispatchTensorStoreOp>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
