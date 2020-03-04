// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA/ConvertHLOToVMLA.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/VMLA/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Clones operand[0] and returns the result.
// This models the value semantics of XLA. We expect previous passes to elide
// identity ops when possible and only check for trivial single use ops here.
template <typename SRC>
struct IdentityOpConversion : public OpConversionPattern<SRC> {
  using OpConversionPattern<SRC>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      SRC srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getOperand().hasOneUse()) {
      // Can directly pass through the input buffer as we don't need to clone
      // for other users.
      rewriter.replaceOp(srcOp, operands[0]);
      return this->matchSuccess();
    } else {
      // More than one user of the operand exist and we need to ensure they
      // keep a valid snapshot of the buffer.
      rewriter.replaceOpWithNewOp<IREE::VMLA::BufferCloneOp>(
          srcOp, IREE::VMLA::BufferType::get(rewriter.getContext()),
          operands[0]);
      return this->matchSuccess();
    }
  }
};

// Converts a broadcast_in_dim op to either a broadcast or a tile depending on
// the input shape.
struct BroadcastInDimOpConversion
    : public OpConversionPattern<xla_hlo::BroadcastInDimOp> {
  BroadcastInDimOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  PatternMatchResult matchAndRewrite(
      xla_hlo::BroadcastInDimOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto srcShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.operand(), typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);

    auto tensorType = srcOp.operand().getType().cast<TensorType>();
    if (tensorType.getRank() == 0) {
      // Broadcast of a scalar value.
      rewriter.create<IREE::VMLA::BroadcastOp>(
          srcOp.getLoc(), operands[0], srcShape, dst, dstShape,
          TypeAttr::get(tensorType.getElementType()));
    } else {
      // Tiling a non-scalar value.
      rewriter.create<IREE::VMLA::TileOp>(
          srcOp.getLoc(), operands[0], srcShape, dst, dstShape,
          TypeAttr::get(tensorType.getElementType()));
    }

    rewriter.replaceOp(srcOp, {dst});
    return matchSuccess();
  }

  TypeConverter &typeConverter;
};

// Converts a concat into a set of copies into the destination buffer.
struct ConcatenateOpConversion
    : public OpConversionPattern<xla_hlo::ConcatenateOp> {
  ConcatenateOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  PatternMatchResult matchAndRewrite(
      xla_hlo::ConcatenateOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto zero = rewriter.createOrFold<mlir::ConstantOp>(
        srcOp.getLoc(), rewriter.getI32IntegerAttr(0));

    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);

    auto finalType = srcOp.getResult().getType().cast<TensorType>();
    int rank = finalType.getRank();
    llvm::SmallVector<Value, 4> srcIndices(rank, zero);
    llvm::SmallVector<Value, 4> dstIndices(rank, zero);
    auto concatDimension = srcOp.dimension().getZExtValue();
    for (auto srcDstOperand : llvm::zip(srcOp.val(), operands)) {
      Value tensorOperand, bufferOperand;
      std::tie(tensorOperand, bufferOperand) = srcDstOperand;

      auto srcShape = VMLAConversionTarget::getTensorShape(
          srcOp.getLoc(), tensorOperand, typeConverter, rewriter);
      SmallVector<Value, 4> lengths(rank);
      for (int i = 0; i < rank; ++i) {
        lengths[i] = rewriter.createOrFold<Shape::RankedDimOp>(srcOp.getLoc(),
                                                               srcShape, i);
      }

      rewriter.create<IREE::VMLA::CopyOp>(
          srcOp.getLoc(), bufferOperand, srcShape, srcIndices, dst, dstShape,
          dstIndices, lengths,
          TypeAttr::get(srcOp.getType().cast<ShapedType>().getElementType()));

      dstIndices[concatDimension] = rewriter.createOrFold<mlir::AddIOp>(
          srcOp.getLoc(), dstIndices[concatDimension],
          lengths[concatDimension]);
    }

    rewriter.replaceOp(srcOp, {dst});
    return matchSuccess();
  }

  TypeConverter &typeConverter;
};

// Lowers a subset of gathers along axis 0 that are really just a slice and
// reshape.
struct GatherOpConversion : public OpConversionPattern<xla_hlo::GatherOp> {
  GatherOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  // TODO(gcmn): This only handles a minimal number of cases. When XLA
  // redefines gather to be simpler, lower it properly.
  PatternMatchResult matchAndRewrite(
      xla_hlo::GatherOp gatherOp, ArrayRef<Value> operandValues,
      ConversionPatternRewriter &rewriter) const override {
    xla_hlo::GatherOpOperandAdaptor operands(operandValues);
    auto dimension_numbers = gatherOp.dimension_numbers();
    if (dimension_numbers.index_vector_dim().getValue().getSExtValue() != 0) {
      gatherOp.emitRemark()
          << "couldn't lower gather with index_vector_dim != 0";
      return matchFailure();
    }
    if (dimension_numbers.start_index_map().getType().getRank() != 1 ||
        dimension_numbers.start_index_map()
                .getValue(0)
                .cast<IntegerAttr>()
                .getValue() != 0) {
      gatherOp.emitRemark()
          << "couldn't lower gather with start_index_map != [0]";
      return matchFailure();
    }
    if (dimension_numbers.collapsed_slice_dims().getType().getRank() != 1 ||
        dimension_numbers.collapsed_slice_dims()
                .getValue(0)
                .cast<IntegerAttr>()
                .getValue() != 0) {
      gatherOp.emitRemark()
          << "couldn't lower gather with collapsed_dims != [0]";
      return matchFailure();
    }

    auto resultType = gatherOp.getResult().getType().cast<RankedTensorType>();
    if (dimension_numbers.offset_dims().getType().getNumElements() !=
        resultType.getRank()) {
      gatherOp.emitRemark() << "couldn't lower gather with offset_dims != "
                               "[0,...,rank of output]";
      return matchFailure();
    }
    for (auto it : llvm::enumerate(dimension_numbers.offset_dims())) {
      if (it.index() != it.value()) {
        gatherOp.emitRemark() << "couldn't lower gather with offset_dims != "
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
            << "couldn't lower gather with slice_sizes not [1] + final shape";
        return matchFailure();
      }
    }

    auto srcShape = VMLAConversionTarget::getTensorShape(
        gatherOp.getLoc(), gatherOp.operand(), typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        gatherOp.getLoc(), gatherOp.getResult(), typeConverter, rewriter);

    auto inputType = gatherOp.operand().getType().cast<RankedTensorType>();
    auto startIndicesType =
        gatherOp.start_indices().getType().cast<ShapedType>();
    int rank = inputType.getRank();
    SmallVector<Value, 4> srcIndices(rank);
    SmallVector<Value, 4> dstIndices(rank);
    SmallVector<Value, 4> lengths(rank);
    Value zero = rewriter.createOrFold<mlir::ConstantOp>(
        gatherOp.getLoc(), rewriter.getI32IntegerAttr(0));
    for (int i = 0; i < rank; ++i) {
      if (i < startIndicesType.getNumElements()) {
        auto srcIndexByteOffset = rewriter.createOrFold<mlir::ConstantOp>(
            gatherOp.getLoc(), rewriter.getI32IntegerAttr(i * sizeof(int32_t)));
        srcIndices[i] = rewriter.createOrFold<IREE::VMLA::BufferLoadI32Op>(
            gatherOp.getLoc(), rewriter.getIntegerType(32),
            operands.start_indices(), srcIndexByteOffset);
      } else {
        // Pad missing dimensions to zero offsets.
        srcIndices[i] = zero;
      }
      dstIndices[i] = zero;
      lengths[i] = rewriter.createOrFold<mlir::ConstantOp>(
          gatherOp.getLoc(),
          rewriter.getI32IntegerAttr(gatherOp.slice_sizes().getValue<int64_t>(
              {static_cast<uint64_t>(i)})));
    }

    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        gatherOp.getLoc(), gatherOp.getResult(), typeConverter, rewriter);
    rewriter.create<IREE::VMLA::CopyOp>(
        gatherOp.getLoc(), operands.operand(), srcShape, srcIndices, dst,
        dstShape, dstIndices, lengths,
        TypeAttr::get(inputType.getElementType()));
    rewriter.replaceOp(gatherOp, {dst});
    return matchSuccess();
  }

  TypeConverter &typeConverter;
};

// Converts a static slice op to a copy (if the source must be preserved).
struct SliceOpConversion : public OpConversionPattern<xla_hlo::SliceOp> {
  SliceOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  PatternMatchResult matchAndRewrite(
      xla_hlo::SliceOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto isNotOne = [](APInt stride) { return stride != 1; };
    if (llvm::any_of(srcOp.strides(), isNotOne)) {
      srcOp.emitWarning()
          << "Could not lower slice op with non-singular strides";
      return matchFailure();
    }

    // TODO(benvanik): if the source is only used by this op then replace with
    // a vmla.buffer.view op.

    auto srcShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.operand(), typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);

    int rank = srcOp.operand().getType().cast<ShapedType>().getRank();
    SmallVector<Value, 4> srcIndices(rank);
    SmallVector<Value, 4> dstIndices(rank);
    SmallVector<Value, 4> lengths(rank);
    Value zero = rewriter.createOrFold<mlir::ConstantOp>(
        srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
    for (int i = 0; i < rank; ++i) {
      uint64_t ui = static_cast<uint64_t>(i);
      srcIndices[i] = rewriter.createOrFold<mlir::ConstantOp>(
          srcOp.getLoc(), rewriter.getI32IntegerAttr(
                              srcOp.start_indices().getValue<int64_t>({ui})));
      dstIndices[i] = zero;
      lengths[i] = rewriter.createOrFold<mlir::ConstantOp>(
          srcOp.getLoc(), rewriter.getI32IntegerAttr(
                              srcOp.limit_indices().getValue<int64_t>({ui}) -
                              srcOp.start_indices().getValue<int64_t>({ui})));
    }

    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    rewriter.create<IREE::VMLA::CopyOp>(
        srcOp.getLoc(), operands[0], srcShape, srcIndices, dst, dstShape,
        dstIndices, lengths,
        TypeAttr::get(srcOp.getType().cast<ShapedType>().getElementType()));
    rewriter.replaceOp(srcOp, {dst});
    return matchSuccess();
  }

  TypeConverter &typeConverter;
};

// Converts a dynamic slice op to a copy (if the source must be preserved).
struct DynamicSliceOpConversion
    : public OpConversionPattern<xla_hlo::DynamicSliceOp> {
  DynamicSliceOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  PatternMatchResult matchAndRewrite(
      xla_hlo::DynamicSliceOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(benvanik): if the source is only used by this op then replace with
    // a vmla.buffer.view op.

    auto srcShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.operand(), typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.result(), typeConverter, rewriter);

    int rank = srcOp.operand().getType().cast<ShapedType>().getRank();
    SmallVector<Value, 4> srcIndices(rank);
    SmallVector<Value, 4> dstIndices(rank);
    SmallVector<Value, 4> lengths(rank);
    Value zero = rewriter.createOrFold<mlir::ConstantOp>(
        srcOp.getLoc(), rewriter.getI32IntegerAttr(0));
    for (int i = 0; i < rank; ++i) {
      srcIndices[i] = rewriter.createOrFold<IREE::VMLA::BufferLoadI32Op>(
          srcOp.getLoc(), rewriter.getIntegerType(32), operands[1],
          rewriter.createOrFold<mlir::ConstantOp>(
              srcOp.getLoc(), rewriter.getI32IntegerAttr(i * sizeof(int32_t))));
      dstIndices[i] = zero;
      lengths[i] = rewriter.createOrFold<mlir::ConstantOp>(
          srcOp.getLoc(),
          rewriter.getI32IntegerAttr(srcOp.slice_sizes().getValue<int64_t>(
              {static_cast<uint64_t>(i)})));
    }

    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    rewriter.create<IREE::VMLA::CopyOp>(
        srcOp.getLoc(), operands[0], srcShape, srcIndices, dst, dstShape,
        dstIndices, lengths,
        TypeAttr::get(srcOp.getType().cast<ShapedType>().getElementType()));
    rewriter.replaceOp(srcOp, {dst});
    return matchSuccess();
  }

  TypeConverter &typeConverter;
};

}  // namespace

void populateHLOToVMLAPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter) {
  // We rely on some additional HLO->HLO/HLO->std patterns and assume they
  // have been run already. In case they haven't we provide them here (useful
  // for standalone conversion testing).
  xla_hlo::PopulateXlaToStdPatterns(&patterns, context);
  xla_hlo::PopulateUnfuseBatchNormPatterns(context, &patterns);

  // Simple 1:1 conversion patterns using the automated trait-based converter.
  // Used for HLO ops that have equivalent VMLA ops such as most arithmetic ops.
  patterns.insert<VMLAOpConversion<xla_hlo::AddOp, IREE::VMLA::AddOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::SubOp, IREE::VMLA::SubOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::DivOp, IREE::VMLA::DivOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::MulOp, IREE::VMLA::MulOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::PowOp, IREE::VMLA::PowOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::RemOp, IREE::VMLA::RemOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::ShiftLeftOp, IREE::VMLA::ShlOp>>(
      context, typeConverter);
  patterns.insert<
      VMLAOpConversion<xla_hlo::ShiftRightArithmeticOp, IREE::VMLA::ShrOp>>(
      context, typeConverter);
  patterns
      .insert<VMLAOpConversion<xla_hlo::ShiftRightLogicalOp, IREE::VMLA::ShrOp,
                               VMLAOpSemantics::kForceUnsigned>>(context,
                                                                 typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::AndOp, IREE::VMLA::AndOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::OrOp, IREE::VMLA::OrOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::XorOp, IREE::VMLA::XorOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::ExpOp, IREE::VMLA::ExpOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::LogOp, IREE::VMLA::LogOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::FloorOp, IREE::VMLA::FloorOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::RsqrtOp, IREE::VMLA::RsqrtOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::SqrtOp, IREE::VMLA::SqrtOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::CosOp, IREE::VMLA::CosOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::SinOp, IREE::VMLA::SinOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::TanhOp, IREE::VMLA::TanhOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::Atan2Op, IREE::VMLA::Atan2Op>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::SelectOp, IREE::VMLA::SelectOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::ConvertOp, IREE::VMLA::ConvertOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::ReverseOp, IREE::VMLA::ReverseOp>>(
      context, typeConverter);
  patterns
      .insert<VMLAOpConversion<xla_hlo::TransposeOp, IREE::VMLA::TransposeOp>>(
          context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::PadOp, IREE::VMLA::PadOp>>(
      context, typeConverter);
  patterns
      .insert<VMLAOpConversion<xla_hlo::BroadcastOp, IREE::VMLA::BroadcastOp>>(
          context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::AbsOp, IREE::VMLA::AbsOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::MaxOp, IREE::VMLA::MaxOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::MinOp, IREE::VMLA::MinOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::ClampOp, IREE::VMLA::ClampOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<xla_hlo::DotOp, IREE::VMLA::MatMulOp>>(
      context, typeConverter);

  // Ops that are only used for type information that we erase. We can elide
  // these entirely by just passing on their input values.
  patterns.insert<IdentityOpConversion<xla_hlo::BitcastConvertOp>>(context);
  patterns.insert<IdentityOpConversion<xla_hlo::CopyOp>>(context);
  patterns.insert<IdentityOpConversion<xla_hlo::ReshapeOp>>(context);

  // Conversions that don't have a 1:1 mapping, mostly involving buffer views
  // or transfers.
  patterns.insert<BroadcastInDimOpConversion>(context, typeConverter);
  patterns.insert<ConcatenateOpConversion>(context, typeConverter);
  patterns.insert<GatherOpConversion>(context, typeConverter);
  patterns.insert<SliceOpConversion>(context, typeConverter);
  patterns.insert<DynamicSliceOpConversion>(context, typeConverter);

  // TODO(benvanik): add missing ops:
  // - ConvOp
}

}  // namespace iree_compiler
}  // namespace mlir
