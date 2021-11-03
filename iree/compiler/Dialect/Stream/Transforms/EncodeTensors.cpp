// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

//===----------------------------------------------------------------------===//
// Encoding utilities
//===----------------------------------------------------------------------===//

// Asserts that the given encoding is supported by this code right now.
// Non-trivial dense tensor encodings need special handling.
static LogicalResult checkEncoding(Operation *op, RankedTensorType encodingType,
                                   ValueRange encodingDims,
                                   PatternRewriter &rewriter) {
  if (encodingType.getEncoding()) {
    return rewriter.notifyMatchFailure(op, [=](Diagnostic &d) {
      d << "unsupported tensor encoding: " << encodingType;
    });
  }
  return success();
}

// Returns an 8-bit aligned element byte count.
static int64_t getElementByteSize(Type elementType) {
  int64_t bitCount = elementType.getIntOrFloatBitWidth();
  int64_t byteCount = (bitCount + 8 - 1) / 8;
  return byteCount;
}

// Returns the element count of a tensor with optional dynamic dimensions.
// Many of these will be static and since this is used _a lot_ we do a bit of
// work to try to avoid a bunch of trivially foldable ops.
static Value calculateElementCount(Location loc, RankedTensorType tensorType,
                                   ValueRange dynamicDims, int64_t multiplier,
                                   PatternRewriter &rewriter) {
  // Calculate all static dims first, if any.
  int64_t staticCount = multiplier;
  for (unsigned i = 0; i < tensorType.getRank(); ++i) {
    if (!tensorType.isDynamicDim(i)) staticCount *= tensorType.getDimSize(i);
  }

  // Scale by dynamic dims, if present.
  auto value =
      rewriter.create<arith::ConstantIndexOp>(loc, staticCount).getResult();
  for (auto dim : dynamicDims) {
    value = rewriter.createOrFold<arith::MulIOp>(loc, value, dim);
  }
  return value;
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
    if (tensorType.isDynamicDim(j)) ++di;
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
  return rewriter.createOrFold<arith::MulIOp>(
      loc,
      calculateElementOffset(loc, tensorType, dynamicDims, indices, rewriter),
      rewriter.create<arith::ConstantIndexOp>(
          loc, getElementByteSize(tensorType.getElementType())));
}

//===----------------------------------------------------------------------===//
// stream.tensor.import
//===----------------------------------------------------------------------===//

struct EncodeTensorImportOp
    : public OpRewritePattern<IREE::Stream::TensorImportOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorImportOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.result_encoding().cast<RankedTensorType>();
    auto resultDims = op.result_encoding_dims();
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
    auto sourceType = op.source_encoding().cast<RankedTensorType>();
    auto sourceDims = op.source_encoding_dims();
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
    auto encodingType = op.encoding().cast<RankedTensorType>();
    auto encodingDims = op.encoding_dims();
    if (failed(checkEncoding(op, encodingType, encodingDims, rewriter))) {
      return failure();
    }

    // Dense: element count * element size.
    auto elementByteSize = getElementByteSize(encodingType.getElementType());
    auto totalSize = calculateElementCount(
        op.getLoc(), encodingType, encodingDims, elementByteSize, rewriter);
    rewriter.replaceOp(op, totalSize);

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
    auto resultType = op.result_encoding().cast<RankedTensorType>();
    auto resultDims = op.result_encoding_dims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // NOTE: we could compute size based on the contents of the elements and
    // perform arbitrary unpacking logic here.

    // Dense:
    auto resultSize = calculateElementCount(
        op.getLoc(), resultType, resultDims,
        getElementByteSize(resultType.getElementType()), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncConstantOp>(
        op, op.result().getType(), op.value(), resultSize, op.affinityAttr());

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
    auto resultType = op.result_encoding().cast<RankedTensorType>();
    auto resultDims = op.result_encoding_dims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // Dense:
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncSplatOp>(
        op, op.result().getType(), op.value(), op.result_size(),
        op.affinityAttr());

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
    auto sourceType = op.source_encoding().cast<RankedTensorType>();
    auto sourceDims = op.source_encoding_dims();
    if (failed(checkEncoding(op, sourceType, sourceDims, rewriter))) {
      return failure();
    }
    auto resultType = op.result_encoding().cast<RankedTensorType>();
    auto resultDims = op.result_encoding_dims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // Dense:
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCloneOp>(
        op, op.result().getType(), op.source(), op.source_size(),
        op.result_size(), op.affinityAttr());

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
    auto sourceType = op.source_encoding().cast<RankedTensorType>();
    auto sourceDims = op.source_encoding_dims();
    if (failed(checkEncoding(op, sourceType, sourceDims, rewriter))) {
      return failure();
    }
    auto resultType = op.result_encoding().cast<RankedTensorType>();
    auto resultDims = op.result_encoding_dims();
    if (failed(checkEncoding(op, resultType, resultDims, rewriter))) {
      return failure();
    }

    // Dense:
    auto sourceOffset = calculateElementByteOffset(
        op.getLoc(), sourceType, sourceDims, op.start_indices(), rewriter);
    auto sourceEnd = rewriter.createOrFold<arith::AddIOp>(
        op.getLoc(), sourceOffset, op.result_size());
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncSliceOp>(
        op, op.result().getType(), op.source(), op.source_size(), sourceOffset,
        sourceEnd, op.result_size(), op.affinityAttr());

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
    auto targetType = op.target_encoding().cast<RankedTensorType>();
    auto targetDims = op.target_encoding_dims();
    if (failed(checkEncoding(op, targetType, targetDims, rewriter))) {
      return failure();
    }

    // Dense:
    auto targetOffset = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.start_indices(), rewriter);
    auto targetLength = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.lengths(), rewriter);
    auto targetEnd = rewriter.createOrFold<arith::AddIOp>(
        op.getLoc(), targetOffset, targetLength);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncFillOp>(
        op, op.result().getType(), op.target(), op.target_size(), targetOffset,
        targetEnd, targetLength, op.value(), op.affinityAttr());

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
    auto updateType = op.update_encoding().cast<RankedTensorType>();
    auto updateDims = op.update_encoding_dims();
    if (failed(checkEncoding(op, updateType, updateDims, rewriter))) {
      return failure();
    }
    auto targetType = op.target_encoding().cast<RankedTensorType>();
    auto targetDims = op.target_encoding_dims();
    if (failed(checkEncoding(op, targetType, targetDims, rewriter))) {
      return failure();
    }

    // Dense:
    auto targetOffset = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.start_indices(), rewriter);
    auto targetEnd = rewriter.createOrFold<arith::AddIOp>(
        op.getLoc(), targetOffset, op.update_size());
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncUpdateOp>(
        op, op.result().getType(), op.target(), op.target_size(), targetOffset,
        targetEnd, op.update(), op.update_size(), op.affinityAttr());

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
    auto sourceType = op.source_encoding().cast<RankedTensorType>();
    auto sourceDims = op.source_encoding_dims();
    if (failed(checkEncoding(op, sourceType, sourceDims, rewriter))) {
      return failure();
    }

    // Dense:
    auto sourceOffset = calculateElementByteOffset(
        op.getLoc(), sourceType, sourceDims, op.indices(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncLoadOp>(
        op, op.result().getType(), op.source(), op.source_size(), sourceOffset);

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
    auto targetType = op.target_encoding().cast<RankedTensorType>();
    auto targetDims = op.target_encoding_dims();
    if (failed(checkEncoding(op, targetType, targetDims, rewriter))) {
      return failure();
    }

    // Dense:
    auto targetOffset = calculateElementByteOffset(
        op.getLoc(), targetType, targetDims, op.indices(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncStoreOp>(
        op, op.target(), op.target_size(), targetOffset, op.value());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// -iree-stream-encode-tensors
//===----------------------------------------------------------------------===//

class EncodeTensorsPass : public EncodeTensorsBase<EncodeTensorsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<
        EncodeTensorImportOp, EncodeTensorExportOp, EncodeTensorSizeOfOp,
        EncodeTensorConstantOp, EncodeTensorSplatOp, EncodeTensorCloneOp,
        EncodeTensorSliceOp, EncodeTensorFillOp, EncodeTensorUpdateOp,
        EncodeTensorLoadOp, EncodeTensorStoreOp>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<>> createEncodeTensorsPass() {
  return std::make_unique<EncodeTensorsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
