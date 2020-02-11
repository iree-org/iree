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
#include "mlir/Dialect/StandardOps/Ops.h"
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
          srcOp,
          IREE::RefPtrType::get(
              IREE::VMLA::BufferType::get(rewriter.getContext())),
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
  patterns.insert<VMLAOpConversion<xla_hlo::CopyOp, IREE::VMLA::BufferCloneOp>>(
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
  patterns.insert<IdentityOpConversion<xla_hlo::ReshapeOp>>(context);

  // Conversions that don't have a 1:1 mapping, mostly involving buffer views
  // or transfers.
  patterns.insert<BroadcastInDimOpConversion>(context, typeConverter);

  // TODO(benvanik): add missing ops:
  // - ConvOp
}

}  // namespace iree_compiler
}  // namespace mlir
