// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compiler/IR/Dialect.h"
#include "compiler/IR/Ops.h"
#include "compiler/IR/Sequencer/HLDialect.h"
#include "compiler/IR/Sequencer/HLOps.h"
#include "compiler/IR/StructureOps.h"
#include "compiler/Transforms/ConversionUtils.h"
#include "compiler/Utils/MemRefUtils.h"
#include "compiler/Utils/OpCreationUtils.h"
#include "compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

// TODO(suderman): tablegen this? or something a bit more flexible.

#define UNARY_OP_LOWERING(XlaOpType, IREEOpType)                 \
  struct XlaOpType##Lowering                                     \
      : public UnaryOpLowering<xla_hlo::XlaOpType, IREEOpType> { \
    using UnaryOpLowering::UnaryOpLowering;                      \
  };

#define TERNARY_OP_LOWERING(XlaOpType, IREEOpType)                 \
  struct XlaOpType##Lowering                                       \
      : public TernaryOpLowering<xla_hlo::XlaOpType, IREEOpType> { \
    using TernaryOpLowering::TernaryOpLowering;                    \
  };

UNARY_OP_LOWERING(CopyOp, IREESeq::HL::CloneOp);

#undef UNARY_OP_LOWERING
#undef TERNARY_OP_LOWERING

template <typename T>
static Operation *createShapeTargetingOp(ConversionPatternRewriter &rewriter,
                                         Location loc, Value *input,
                                         MemRefType targetType) {
  auto shapeOp = createArrayConstant(rewriter, loc, targetType.getShape());
  return rewriter.create<T>(loc, targetType, input, shapeOp);
}

static Value *inputAsMemref(ConversionPatternRewriter &rewriter, Operation *op,
                            Value *tensor) {
  return wrapAsMemRef(loadAccessValue(op->getLoc(), tensor, rewriter), op,
                      rewriter);
}

template <typename SrcOp>
class XlaOpLowering : public OpConversionPattern<SrcOp> {
 public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      SrcOp op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto srcOp = cast<SrcOp>(op);

    SmallVector<Value *, 4> memrefOperands;
    for (auto operand : operands) {
      memrefOperands.push_back(inputAsMemref(rewriter, op, operand));
    }

    auto dstOp = rewriteInternal(&srcOp, memrefOperands, rewriter);
    rewriter.replaceOp(op, wrapAsTensor(dstOp->getResult(0), srcOp, rewriter));
    return this->matchSuccess();
  }

 protected:
  virtual Operation *rewriteInternal(
      SrcOp *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite, did you mean rewriteTerminator?");
  }
};

struct ConcatOpLowering : public XlaOpLowering<xla_hlo::ConcatenateOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::ConcatenateOp *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto finalType = convertTypeToMemRef(*op);

    return rewriter.create<IREESeq::HL::ConcatOp>(
        op->getLoc(), finalType, operands,
        rewriter.getI32IntegerAttr(op->dimension().getZExtValue()));
  }
};

struct DynamicUpdateSliceLowering
    : public XlaOpLowering<xla_hlo::DynamicUpdateSliceOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::DynamicUpdateSliceOp *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operand = operands[0];
    auto update = operands[1];

    auto updateType = update->getType().cast<ShapedType>();
    Value *lengthConstant =
        createArrayConstant(rewriter, op->getLoc(), updateType.getShape());

    auto startIndices = makeArrayRef(operands).drop_front(2);
    const int rank = startIndices.size();
    llvm::SmallVector<Value *, 4> valuesToConcat;
    valuesToConcat.reserve(startIndices.size());
    auto type = getElementTypeOrSelf(startIndices.front());

    // To generate the offset matrix we need to convert the variadic tensors
    // into a reshaped and concated value.
    for (auto index : startIndices) {
      auto reshapedIndex = rewriter.create<IREESeq::HL::ReshapeOp>(
          op->getLoc(), MemRefType::get({1}, type), index,
          createArrayConstant(rewriter, op->getLoc(), {1}));
      valuesToConcat.push_back(reshapedIndex);
    }

    auto dstOffset = rewriter
                         .create<IREESeq::HL::ConcatOp>(
                             op->getLoc(), MemRefType::get({rank}, type),
                             valuesToConcat, rewriter.getI32IntegerAttr(0))
                         .getResult();

    llvm::SmallVector<int64_t, 4> zero_offset;
    zero_offset.resize(updateType.getRank(), 0);
    auto srcOffset = createArrayConstant(rewriter, op->getLoc(), zero_offset);

    auto copiedOperand = rewriter.create<IREESeq::HL::CloneOp>(
        op->getLoc(), operand->getType(), operand);

    rewriter
        .create<IREESeq::HL::CopyOp>(op->getLoc(), update, srcOffset,
                                     copiedOperand, dstOffset, lengthConstant)
        .getOperation();

    return copiedOperand;
  }
};

struct SliceLowering : public XlaOpLowering<xla_hlo::SliceOp> {
  using XlaOpLowering::XlaOpLowering;
  Operation *rewriteInternal(
      xla_hlo::SliceOp *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    // XLA slice has value semantics, whereas the IREE slice creates a view. We
    // lower it to a copy if all strides are one which may be transformed to a
    // slice by later optimizations.
    auto isNotOne = [](APInt stride) { return stride != 1; };
    if (llvm::any_of(op->strides(), isNotOne)) {
      op->emitRemark() << "Could not lower slice op with non-singular strides";
      return nullptr;
    }

    auto finalType = convertTypeToMemRef(*op);

    auto src = operands[0];
    std::vector<Value *> dim_pieces;
    auto dst = rewriter.create<IREESeq::HL::AllocHeapOp>(op->getLoc(),
                                                         finalType, dim_pieces);
    auto srcIndices =
        rewriter.create<IREE::ConstantOp>(op->getLoc(), op->start_indices());
    auto lengths =
        createArrayConstant(rewriter, op->getLoc(), finalType.getShape());

    llvm::SmallVector<int64_t, 4> zero_offset;
    zero_offset.resize(finalType.getRank(), 0);
    auto dstIndices = createArrayConstant(rewriter, op->getLoc(), zero_offset);

    rewriter.create<IREESeq::HL::CopyOp>(op->getLoc(), src, srcIndices, dst,
                                         dstIndices, lengths);
    return dst;
  }
};

struct ReshapeOpLowering : public XlaOpLowering<xla_hlo::ReshapeOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::ReshapeOp *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const override {
    return createShapeTargetingOp<IREESeq::HL::ReshapeOp>(
        rewriter, op->getLoc(), operands[0], convertTypeToMemRef(*op));
  }
};

}  // namespace

void populateLowerXlaToSequencerPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx) {
  patterns.insert<ConcatOpLowering, CopyOpLowering, DynamicUpdateSliceLowering,
                  ReshapeOpLowering, SliceLowering>(ctx);
}

}  // namespace iree_compiler
}  // namespace mlir
