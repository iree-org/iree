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

#include "iree/compiler/Dialect/VMLA/Conversion/StandardToVMLA/ConvertStandardToVMLA.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConstantOpConversion
    : public VMLAOpConversion<mlir::ConstantOp, IREE::VMLA::BufferConstOp> {
  using VMLAOpConversion::VMLAOpConversion;

  PatternMatchResult matchAndRewrite(
      mlir::ConstantOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto value = srcOp.value().dyn_cast<ElementsAttr>();
    if (!value) return matchFailure();
    rewriter.replaceOpWithNewOp<IREE::VMLA::ConstantOp>(srcOp, value);
    return matchSuccess();
  }
};

struct CmpIOpConversion
    : public VMLAOpConversion<mlir::CmpIOp, IREE::VMLA::CmpOp> {
  using VMLAOpConversion::VMLAOpConversion;

  PatternMatchResult matchAndRewrite(
      mlir::CmpIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = srcOp.lhs().getType().dyn_cast<ShapedType>();
    if (!inputType) return matchFailure();

    IREE::VMLA::CmpPredicate predicate = IREE::VMLA::CmpPredicate::EQ;
    bool forceUnsigned = false;
    switch (srcOp.predicate()) {
      case CmpIPredicate::eq:
        predicate = IREE::VMLA::CmpPredicate::EQ;
        break;
      case CmpIPredicate::ne:
        predicate = IREE::VMLA::CmpPredicate::NE;
        break;
      case CmpIPredicate::slt:
        predicate = IREE::VMLA::CmpPredicate::LT;
        break;
      case CmpIPredicate::sle:
        predicate = IREE::VMLA::CmpPredicate::LE;
        break;
      case CmpIPredicate::sgt:
        predicate = IREE::VMLA::CmpPredicate::GT;
        break;
      case CmpIPredicate::sge:
        predicate = IREE::VMLA::CmpPredicate::GE;
        break;
      case CmpIPredicate::ult:
        predicate = IREE::VMLA::CmpPredicate::LT;
        forceUnsigned = true;
        break;
      case CmpIPredicate::ule:
        predicate = IREE::VMLA::CmpPredicate::LE;
        forceUnsigned = true;
        break;
      case CmpIPredicate::ugt:
        predicate = IREE::VMLA::CmpPredicate::GT;
        forceUnsigned = true;
        break;
      case CmpIPredicate::uge:
        predicate = IREE::VMLA::CmpPredicate::GE;
        forceUnsigned = true;
        break;
      default:
        llvm_unreachable("unhandled comparison predicate");
        return matchFailure();
    }

    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    auto newOp = rewriter.create<IREE::VMLA::CmpOp>(
        srcOp.getLoc(), static_cast<uint32_t>(predicate), operands[0],
        operands[1], dst, TypeAttr::get(inputType.getElementType()));
    if (forceUnsigned) {
      newOp.setAttr("force_unsigned", UnitAttr::get(rewriter.getContext()));
    }
    rewriter.replaceOp(srcOp, newOp.dst());
    return matchSuccess();
  }
};

class CmpFOpConversion
    : public VMLAOpConversion<mlir::CmpFOp, IREE::VMLA::CmpOp> {
 public:
  using VMLAOpConversion::VMLAOpConversion;

  PatternMatchResult matchAndRewrite(
      mlir::CmpFOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto inputType = srcOp.lhs().getType().dyn_cast<ShapedType>();
    if (!inputType) return matchFailure();

    // NOTE: the std.cmpf semantics are practically undefined. We explicitly
    // match the HLO semantics (that get lowered to the expected case values
    // here). In the future as new ML-focused intermediate dialects are built we
    // can reevaluate what we support here.
    //
    // Rules:
    // https://stackoverflow.com/questions/8627331/what-does-ordered-unordered-comparison-mean
    IREE::VMLA::CmpPredicate predicate = IREE::VMLA::CmpPredicate::EQ;
    switch (srcOp.getPredicate()) {
      case CmpFPredicate::OEQ:
        predicate = IREE::VMLA::CmpPredicate::EQ;
        break;
      case CmpFPredicate::UNE:
        predicate = IREE::VMLA::CmpPredicate::NE;
        break;
      case CmpFPredicate::OLT:
        predicate = IREE::VMLA::CmpPredicate::LT;
        break;
      case CmpFPredicate::OLE:
        predicate = IREE::VMLA::CmpPredicate::LE;
        break;
      case CmpFPredicate::OGT:
        predicate = IREE::VMLA::CmpPredicate::GT;
        break;
      case CmpFPredicate::OGE:
        predicate = IREE::VMLA::CmpPredicate::GE;
        break;
      default:
        llvm_unreachable("unhandled comparison predicate");
        return matchFailure();
    }

    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    auto newOp = rewriter.create<IREE::VMLA::CmpOp>(
        srcOp.getLoc(), static_cast<uint32_t>(predicate), operands[0],
        operands[1], dst, TypeAttr::get(inputType.getElementType()));
    rewriter.replaceOp(srcOp, newOp.dst());
    return matchSuccess();
  }
};

}  // namespace

void populateStandardToVMLAPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter) {
  patterns.insert<ConstantOpConversion>(context, typeConverter);
  patterns.insert<CmpIOpConversion>(context, typeConverter);
  patterns.insert<CmpFOpConversion>(context, typeConverter);

  patterns.insert<VMLAOpConversion<mlir::AddIOp, IREE::VMLA::AddOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::AddFOp, IREE::VMLA::AddOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::SubIOp, IREE::VMLA::SubOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::SubFOp, IREE::VMLA::SubOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::MulIOp, IREE::VMLA::MulOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::MulFOp, IREE::VMLA::MulOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::SignedDivIOp, IREE::VMLA::DivOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::UnsignedDivIOp, IREE::VMLA::DivOp,
                                   VMLAOpSemantics::kForceUnsigned>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::DivFOp, IREE::VMLA::DivOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::AbsFOp, IREE::VMLA::AbsOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::SignedRemIOp, IREE::VMLA::RemOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::UnsignedRemIOp, IREE::VMLA::RemOp,
                                   VMLAOpSemantics::kForceUnsigned>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::RemFOp, IREE::VMLA::RemOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::LogOp, IREE::VMLA::LogOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::ExpOp, IREE::VMLA::ExpOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::SqrtOp, IREE::VMLA::SqrtOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::CosOp, IREE::VMLA::CosOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::TanhOp, IREE::VMLA::TanhOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::NegFOp, IREE::VMLA::NegOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::AndOp, IREE::VMLA::AndOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::OrOp, IREE::VMLA::OrOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::XOrOp, IREE::VMLA::XorOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::ShiftLeftOp, IREE::VMLA::ShlOp>>(
      context, typeConverter);
  patterns
      .insert<VMLAOpConversion<mlir::SignedShiftRightOp, IREE::VMLA::ShrOp>>(
          context, typeConverter);
  patterns
      .insert<VMLAOpConversion<mlir::UnsignedShiftRightOp, IREE::VMLA::ShrOp,
                               VMLAOpSemantics::kForceUnsigned>>(context,
                                                                 typeConverter);
  patterns.insert<VMLAOpConversion<mlir::CeilFOp, IREE::VMLA::CeilOp>>(
      context, typeConverter);
  patterns.insert<VMLAOpConversion<mlir::SelectOp, IREE::VMLA::SelectOp>>(
      context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
