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

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/Interpreter/IR/CommonDialect.h"
#include "iree/compiler/Translation/Interpreter/IR/HLDialect.h"
#include "iree/compiler/Translation/Interpreter/IR/HLOps.h"
#include "iree/compiler/Translation/Interpreter/Transforms/ConversionUtils.h"
#include "iree/compiler/Translation/Interpreter/Utils/MemRefUtils.h"
#include "iree/compiler/Translation/Interpreter/Utils/OpCreationUtils.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

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

UNARY_OP_LOWERING(CopyOp, IREEInterp::HL::CloneOp);
UNARY_OP_LOWERING(ExpOp, IREEInterp::HL::ExpFOp);
UNARY_OP_LOWERING(LogOp, IREEInterp::HL::LogFOp);
UNARY_OP_LOWERING(FloorOp, IREEInterp::HL::FloorFOp);
UNARY_OP_LOWERING(RsqrtOp, IREEInterp::HL::RsqrtFOp);
UNARY_OP_LOWERING(SqrtOp, IREEInterp::HL::SqrtFOp);
UNARY_OP_LOWERING(CosOp, IREEInterp::HL::CosFOp);
UNARY_OP_LOWERING(SinOp, IREEInterp::HL::SinFOp);
UNARY_OP_LOWERING(TanhOp, IREEInterp::HL::TanhFOp);
TERNARY_OP_LOWERING(SelectOp, IREEInterp::HL::SelectOp);

#undef UNARY_OP_LOWERING
#undef TERNARY_OP_LOWERING

template <typename T>
static Operation *createShapeTargetingOp(ConversionPatternRewriter &rewriter,
                                         Location loc, Value input,
                                         MemRefType targetType) {
  auto shapeOp = createArrayConstant(rewriter, loc, targetType.getShape());
  return rewriter.create<T>(loc, targetType, input, shapeOp);
}

static Value inputAsMemref(ConversionPatternRewriter &rewriter, Operation *op,
                           Value tensor) {
  return wrapAsMemRef(loadAccessValue(op->getLoc(), tensor, rewriter), op,
                      rewriter);
}

template <typename SrcOp>
class XlaOpLowering : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      SrcOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> memrefOperands;
    for (auto operand : operands) {
      memrefOperands.push_back(inputAsMemref(rewriter, srcOp, operand));
    }

    if (auto dstOp = rewriteInternal(&srcOp, memrefOperands, rewriter)) {
      rewriter.replaceOp(srcOp,
                         wrapAsTensor(dstOp->getResult(0), srcOp, rewriter));
      return this->matchSuccess();
    }
    return this->matchFailure();
  }

 protected:
  virtual Operation *rewriteInternal(
      SrcOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite, did you mean rewriteTerminator?");
  }
};

struct BroadcastInDimOpLowering
    : public XlaOpLowering<xla_hlo::BroadcastInDimOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::BroadcastInDimOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto inputValue = operands[0];
    auto inputType = inputValue.getType().cast<MemRefType>();
    auto finalType = convertLegacyTypeToMemRef(*op);

    // Reshape to scalar and broadcast.
    auto createFinal = createShapeTargetingOp<IREEInterp::HL::BroadcastOp>;
    llvm::SmallVector<int64_t, 6> intermediateShape{};

    // Or reshape to final rank and tile.
    if (inputType.getNumElements() != 1) {
      createFinal = createShapeTargetingOp<IREEInterp::HL::TileOp>;

      intermediateShape = llvm::SmallVector<int64_t, 6>(finalType.getRank(), 1);
      auto inputShape = inputType.getShape();
      auto dimensions = op->broadcast_dimensions();
      for (size_t i = 0; i < inputType.getRank(); ++i) {
        auto index = dimensions->getValue(i).cast<IntegerAttr>().getInt();
        intermediateShape[index] = inputShape[i];
      }
    }

    auto intermediateType =
        MemRefType::get(intermediateShape, inputType.getElementType());
    auto reshapeOp = createShapeTargetingOp<IREEInterp::HL::ReshapeOp>(
        rewriter, op->getLoc(), inputValue, intermediateType);
    return createFinal(rewriter, op->getLoc(), reshapeOp->getResult(0),
                       finalType);
  }
};

struct ConcatOpLowering : public XlaOpLowering<xla_hlo::ConcatenateOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::ConcatenateOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto finalType = convertLegacyTypeToMemRef(*op);

    return rewriter.create<IREEInterp::HL::ConcatOp>(
        op->getLoc(), finalType, operands,
        rewriter.getI32IntegerAttr(op->dimension().getZExtValue()));
  }
};

struct DotOpLowering : public XlaOpLowering<xla_hlo::DotOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::DotOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto lhsValue = operands[0];
    auto rhsValue = operands[1];

    auto finalType = convertLegacyTypeToMemRef(*op);
    auto elementType = finalType.getElementType();
    if (!elementType.isa<FloatType>()) {
      op->emitOpError("xla_hlo.dot only supports floating point values");
    }

    Operation *matMulOp = rewriter
                              .create<IREEInterp::HL::MatMulFOp>(
                                  op->getLoc(), finalType, lhsValue, rhsValue)
                              .getOperation();
    return matMulOp;
  }
};

struct DynamicUpdateSliceOpLowering
    : public XlaOpLowering<xla_hlo::DynamicUpdateSliceOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::DynamicUpdateSliceOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operand = operands[0];
    auto update = operands[1];

    auto updateType = update.getType().cast<ShapedType>();
    Value lengthConstant =
        createArrayConstant(rewriter, op->getLoc(), updateType.getShape());

    auto startIndices = makeArrayRef(operands).drop_front(2);
    const int rank = startIndices.size();
    llvm::SmallVector<Value, 4> valuesToConcat;
    valuesToConcat.reserve(startIndices.size());
    auto type = getElementTypeOrSelf(startIndices.front());

    // To generate the offset matrix we need to convert the variadic tensors
    // into a reshaped and concated value.
    for (auto index : startIndices) {
      auto reshapedIndex = rewriter.create<IREEInterp::HL::ReshapeOp>(
          op->getLoc(), MemRefType::get({1}, type), index,
          createArrayConstant(rewriter, op->getLoc(), {1}));
      valuesToConcat.push_back(reshapedIndex);
    }

    auto dstOffset = rewriter
                         .create<IREEInterp::HL::ConcatOp>(
                             op->getLoc(), MemRefType::get({rank}, type),
                             valuesToConcat, rewriter.getI32IntegerAttr(0))
                         .getResult();

    llvm::SmallVector<int64_t, 4> zero_offset;
    zero_offset.resize(updateType.getRank(), 0);
    auto srcOffset = createArrayConstant(rewriter, op->getLoc(), zero_offset);

    auto copiedOperand = rewriter.create<IREEInterp::HL::CloneOp>(
        op->getLoc(), operand.getType(), operand);

    rewriter
        .create<IREEInterp::HL::CopyOp>(op->getLoc(), update, srcOffset,
                                        copiedOperand, dstOffset,
                                        lengthConstant)
        .getOperation();

    return copiedOperand;
  }
};

template <typename XlaOpType, typename IreeFloatOpType, typename IreeIntOpType>
struct UnaryFloatIntOpLowering : public XlaOpLowering<XlaOpType> {
  using XlaOpLowering<XlaOpType>::XlaOpLowering;

  Operation *rewriteInternal(
      XlaOpType *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto val = operands[0];
    auto inputType = val.getType().cast<MemRefType>();
    auto elementType = inputType.getElementType();

    if (elementType.isa<FloatType>()) {
      return rewriter.create<IreeFloatOpType>(op->getLoc(), inputType, val);
    }

    return rewriter.create<IreeIntOpType>(op->getLoc(), inputType, val);
  }
};

struct AbsOpLowering
    : public UnaryFloatIntOpLowering<xla_hlo::AbsOp, IREEInterp::HL::AbsFOp,
                                     IREEInterp::HL::AbsIOp> {
  using UnaryFloatIntOpLowering::UnaryFloatIntOpLowering;
};

template <typename XlaOpType, typename IreeFloatOpType, typename IreeIntOpType>
struct BinaryFloatIntOpLowering : public XlaOpLowering<XlaOpType> {
  using XlaOpLowering<XlaOpType>::XlaOpLowering;

  Operation *rewriteInternal(
      XlaOpType *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto lhs = operands[0];
    auto rhs = operands[1];
    auto inputType = lhs.getType().cast<MemRefType>();
    auto elementType = inputType.getElementType();

    if (elementType.isa<FloatType>()) {
      return rewriter.create<IreeFloatOpType>(op->getLoc(), inputType, lhs,
                                              rhs);
    }

    return rewriter.create<IreeIntOpType>(op->getLoc(), inputType, lhs, rhs);
  }
};

struct MaxOpLowering
    : public BinaryFloatIntOpLowering<xla_hlo::MaxOp, IREEInterp::HL::MaxFOp,
                                      IREEInterp::HL::MaxISOp> {
  using BinaryFloatIntOpLowering::BinaryFloatIntOpLowering;
};

struct MinOpLowering
    : public BinaryFloatIntOpLowering<xla_hlo::MinOp, IREEInterp::HL::MinFOp,
                                      IREEInterp::HL::MinISOp> {
  using BinaryFloatIntOpLowering::BinaryFloatIntOpLowering;
};

struct ConvertLowering : public XlaOpLowering<xla_hlo::ConvertOp> {
  using XlaOpLowering<xla_hlo::ConvertOp>::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::ConvertOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operand = operands[0];
    auto result = op->getResult();

    auto operandType = operand.getType().cast<MemRefType>().getElementType();
    auto resultType = result.getType().cast<ShapedType>().getElementType();

    auto newResultType = convertLegacyTypeToMemRef(result);

#define ConvertCase(InType, OutType, NewOp)                                \
  {                                                                        \
    if (operandType.isa<InType>() && resultType.isa<OutType>()) {          \
      return rewriter.create<NewOp>(op->getLoc(), newResultType, operand); \
    }                                                                      \
  }
    ConvertCase(IntegerType, IntegerType, IREEInterp::HL::ConvertSSOp);
    ConvertCase(IntegerType, FloatType, IREEInterp::HL::ConvertSFOp);
    ConvertCase(FloatType, IntegerType, IREEInterp::HL::ConvertFSOp);
    ConvertCase(FloatType, FloatType, IREEInterp::HL::ConvertFFOp);
#undef ConvertCase

    return nullptr;
  }
};

// Lowers a subset of gathers along axis 0 that are really just a slice and
// reshape.
struct GatherOpLowering : public OpConversionPattern<xla_hlo::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  // TODO(gcmn): This only handles a minimal number of cases. When XLA
  // redefines gather to be simpler, lower it properly.
  PatternMatchResult matchAndRewrite(
      xla_hlo::GatherOp gatherOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto dimension_numbers = gatherOp.dimension_numbers();
    if (dimension_numbers.index_vector_dim().getValue().getSExtValue() != 0) {
      gatherOp.emitRemark()
          << "Couldn't lower gather with index_vector_dim != 0";
      return matchFailure();
    }
    if (dimension_numbers.start_index_map().getType().getRank() != 1 ||
        dimension_numbers.start_index_map()
                .getValue(0)
                .cast<IntegerAttr>()
                .getValue() != 0) {
      gatherOp.emitRemark()
          << "Couldn't lower gather with start_index_map != [0]";
      return matchFailure();
    }
    if (dimension_numbers.collapsed_slice_dims().getType().getRank() != 1 ||
        dimension_numbers.collapsed_slice_dims()
                .getValue(0)
                .cast<IntegerAttr>()
                .getValue() != 0) {
      gatherOp.emitRemark()
          << "Couldn't lower gather with collapsed_dims != [0]";
      return matchFailure();
    }

    auto resultType = gatherOp.getResult().getType().cast<RankedTensorType>();
    if (dimension_numbers.offset_dims().getType().getNumElements() !=
        resultType.getRank()) {
      gatherOp.emitRemark() << "Couldn't lower gather with offset_dims != "
                               "[0,...,rank of output]";
      return matchFailure();
    }
    for (auto it : llvm::enumerate(dimension_numbers.offset_dims())) {
      if (it.index() != it.value()) {
        gatherOp.emitRemark() << "Couldn't lower gather with offset_dims != "
                                 "[0,...,rank of output]";
        return matchFailure();
      }
    }

    for (auto it : llvm::enumerate(resultType.getShape())) {
      if (gatherOp.slice_sizes()
              .getValue(it.index() + 1)
              .cast<IntegerAttr>()
              .getValue() != it.value()) {
        gatherOp.emitRemark()
            << "Couldn't lower gather with slice_sizes not [1] + final shape";
        return matchFailure();
      }
    }

    auto inputType = gatherOp.operand().getType().cast<RankedTensorType>();

    auto startIndices =
        inputAsMemref(rewriter, gatherOp, gatherOp.start_indices());
    auto startIndicesType = startIndices.getType().cast<MemRefType>();
    if (startIndicesType.getNumElements() != inputType.getRank()) {
      auto extraDims = inputType.getRank() - startIndicesType.getNumElements();
      auto elementType = startIndicesType.getElementType();

      if (startIndicesType.getRank() != 1) {
        startIndices = createShapeTargetingOp<IREEInterp::HL::ReshapeOp>(
                           rewriter, gatherOp.getLoc(), startIndices,
                           MemRefType::get({1}, elementType))
                           ->getResult(0);
      }

      ElementsAttr elementsAttr;
      if (elementType.isInteger(32)) {
        llvm::SmallVector<int32_t, 4> zeroes(extraDims);
        elementsAttr = DenseIntElementsAttr::get(
            RankedTensorType::get(zeroes.size(), elementType),
            llvm::makeArrayRef(zeroes));
      } else {
        llvm::SmallVector<int64_t, 4> zeroes(extraDims);
        elementsAttr = DenseIntElementsAttr::get(
            RankedTensorType::get(zeroes.size(), elementType),
            llvm::makeArrayRef(zeroes));
      }

      auto extraStartIndices = rewriter.create<IREEInterp::ConstantOp>(
          gatherOp.getLoc(), elementsAttr);

      auto memrefOutputType =
          MemRefType::get({inputType.getRank()}, elementType);

      SmallVector<Value, 2> valuesToConcat = {startIndices, extraStartIndices};
      startIndices = rewriter.create<IREEInterp::HL::ConcatOp>(
          gatherOp.getLoc(), memrefOutputType, valuesToConcat,
          rewriter.getI32IntegerAttr(0));
    }

    auto sliceSizeValues = gatherOp.slice_sizes().getValues<int64_t>();
    std::vector<int64_t> sliceSizes = {sliceSizeValues.begin(),
                                       sliceSizeValues.end()};
    auto dstType = MemRefType::get(sliceSizes, inputType.getElementType());

    auto src = inputAsMemref(rewriter, gatherOp, gatherOp.operand());
    std::vector<Value> dim_pieces;
    auto dst = rewriter.create<IREEInterp::HL::AllocHeapOp>(
        gatherOp.getLoc(), dstType, dim_pieces);
    auto lengths = rewriter.create<IREEInterp::ConstantOp>(
        gatherOp.getLoc(), gatherOp.slice_sizes());
    llvm::SmallVector<int64_t, 4> zero_offset;
    zero_offset.resize(dstType.getRank(), 0);
    auto dstIndices =
        createArrayConstant(rewriter, gatherOp.getLoc(), zero_offset);

    rewriter.create<IREEInterp::HL::CopyOp>(
        gatherOp.getLoc(), src, startIndices, dst, dstIndices, lengths);

    auto reshaped = createShapeTargetingOp<IREEInterp::HL::ReshapeOp>(
        rewriter, gatherOp.getLoc(), dst, convertLegacyTypeToMemRef(gatherOp));
    rewriter.replaceOp(
        gatherOp, wrapAsTensor(reshaped->getResult(0), gatherOp, rewriter));

    return matchSuccess();
  }
};

struct SliceOpLowering : public XlaOpLowering<xla_hlo::SliceOp> {
  using XlaOpLowering<xla_hlo::SliceOp>::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::SliceOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // XLA slice has value semantics, whereas the IREE slice creates a view. We
    // lower it to a copy if all strides are one which may be transformed to a
    // slice by later optimizations.
    auto isNotOne = [](APInt stride) { return stride != 1; };
    if (llvm::any_of(op->strides(), isNotOne)) {
      op->emitRemark() << "Could not lower slice op with non-singular strides";
      return nullptr;
    }

    auto finalType = convertLegacyTypeToMemRef(*op);
    auto src = operands[0];
    std::vector<Value> dim_pieces;
    auto dst = rewriter.create<IREEInterp::HL::AllocHeapOp>(
        op->getLoc(), finalType, dim_pieces);
    auto srcIndices = rewriter.create<IREEInterp::ConstantOp>(
        op->getLoc(), op->start_indices());
    auto lengths =
        createArrayConstant(rewriter, op->getLoc(), finalType.getShape());

    llvm::SmallVector<int64_t, 4> zero_offset;
    zero_offset.resize(finalType.getRank(), 0);
    auto dstIndices = createArrayConstant(rewriter, op->getLoc(), zero_offset);

    rewriter.create<IREEInterp::HL::CopyOp>(op->getLoc(), src, srcIndices, dst,
                                            dstIndices, lengths);
    return dst;
  }
};

struct PadOpLowering : public XlaOpLowering<xla_hlo::PadOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::PadOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto src = operands[0];
    auto paddingValue = operands[1];

    // TODO(b/140836672) Support negative padding
    for (int i = 0; i < op->edge_padding_high().getNumElements(); ++i) {
      if (op->edge_padding_high().getValue<IntegerAttr>(i).getInt() < 0 ||
          op->edge_padding_low().getValue<IntegerAttr>(i).getInt() < 0) {
        op->emitRemark() << "Could not lower pad op with negative padding";
        return nullptr;
      }
    }

    auto edgePaddingLowOp = rewriter.create<IREEInterp::ConstantOp>(
        op->getLoc(), op->edge_padding_low());
    auto edgePaddingHighOp = rewriter.create<IREEInterp::ConstantOp>(
        op->getLoc(), op->edge_padding_high());
    auto interiorPaddingOp = rewriter.create<IREEInterp::ConstantOp>(
        op->getLoc(), op->interior_padding());

    return rewriter.create<IREEInterp::HL::PadOp>(
        op->getLoc(), convertLegacyTypeToMemRef(*op), src, paddingValue,
        edgePaddingLowOp, edgePaddingHighOp, interiorPaddingOp);
  }
};

struct ReshapeOpLowering : public XlaOpLowering<xla_hlo::ReshapeOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::ReshapeOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    return createShapeTargetingOp<IREEInterp::HL::ReshapeOp>(
        rewriter, op->getLoc(), operands[0], convertLegacyTypeToMemRef(*op));
  }
};

struct TransposeOpLowering : public XlaOpLowering<xla_hlo::TransposeOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::TransposeOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto permutationOp = rewriter.create<IREEInterp::ConstantOp>(
        op->getLoc(), op->permutation());

    return rewriter.create<IREEInterp::HL::TransposeOp>(
        op->getLoc(), convertLegacyTypeToMemRef(*op), operands[0],
        permutationOp);
  }
};

struct ReverseOpLowering : public XlaOpLowering<xla_hlo::ReverseOp> {
  using XlaOpLowering::XlaOpLowering;

  Operation *rewriteInternal(
      xla_hlo::ReverseOp *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto reverseOp =
        rewriter.create<IREEInterp::ConstantOp>(op->getLoc(), op->dimensions());

    return rewriter.create<IREEInterp::HL::ReverseOp>(
        op->getLoc(), convertLegacyTypeToMemRef(*op), operands[0], reverseOp);
  }
};

}  // namespace

void populateLowerXlaToInterpreterPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  xla_hlo::PopulateUnfuseBatchNormPatterns(ctx, &patterns);
  patterns
      .insert<AbsOpLowering, BroadcastInDimOpLowering, ConcatOpLowering,
              ConvertLowering, CopyOpLowering, CosOpLowering, DotOpLowering,
              DynamicUpdateSliceOpLowering, ExpOpLowering, FloorOpLowering,
              GatherOpLowering, LogOpLowering, MaxOpLowering, MinOpLowering,
              PadOpLowering, ReshapeOpLowering, ReverseOpLowering,
              RsqrtOpLowering, SinOpLowering, SqrtOpLowering, SelectOpLowering,
              SliceOpLowering, TransposeOpLowering, TanhOpLowering>(ctx);
}

namespace {
// Just for testing these passes.
// TODO(b/141337493) can we get rid of this pass entirely?
class LowerXLAToInterpreterDialectPass
    : public FunctionPass<LowerXLAToInterpreterDialectPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateLowerXlaToInterpreterPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<IREEHLInterpreterDialect, IREEInterpreterDialect,
                           IREEDialect>();

    // Unfuse batch norm into primitive ops.
    target.addIllegalOp<xla_hlo::BatchNormInferenceOp>();

    if (failed(applyPartialConversion(getFunction(), target, patterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

static PassRegistration<LowerXLAToInterpreterDialectPass> pass(
    "lower-xla-to-iree-interpreter",
    "Convert all XLA functions to the IREE dialect");

}  // namespace iree_compiler
}  // namespace mlir
