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

#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

class ModuleOpConversion : public OpConversionPattern<ModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModuleOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert the top level module.
    // This mechanism can only support rewriting non top-level modules.
    if (!srcOp->getParentOp() || !isa<ModuleOp>(srcOp->getParentOp())) {
      return failure();
    }

    StringRef name = srcOp.getName() ? *srcOp.getName() : "module";
    auto newModuleOp =
        rewriter.create<IREE::VM::ModuleOp>(srcOp.getLoc(), name);
    assert(!newModuleOp.getBodyRegion().empty());
    Block *firstCreatedBlock = &newModuleOp.getBodyRegion().front();
    rewriter.inlineRegionBefore(srcOp.getBodyRegion(), firstCreatedBlock);
    auto blockRange = llvm::make_range(Region::iterator(firstCreatedBlock),
                                       newModuleOp.getBodyRegion().end());
    for (Block &block : llvm::make_early_inc_range(blockRange)) {
      rewriter.eraseBlock(&block);
    }
    rewriter.replaceOp(srcOp, {});
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(&newModuleOp.getBodyRegion().front());
    rewriter.create<IREE::VM::ModuleTerminatorOp>(srcOp.getLoc());
    return success();
  }
};

// Allowlist of function attributes to retain when converting to vm.func.
constexpr const char *kRetainedAttributes[] = {
    "iree.reflection",
    "sym_visibility",
    "noinline",
};

class FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FuncOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getType();
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
      if (failed(getTypeConverter()->convertSignatureArg(
              i, srcFuncType.getInput(i), signatureConversion))) {
        return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
      }
    }

    // Convert function results.
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(srcFuncType.getResults(),
                                                convertedResultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);
    auto newFuncOp = rewriter.create<IREE::VM::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), newFuncType);
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Retain function attributes in the allowlist.
    auto retainedAttributes = ArrayRef<const char *>(
        kRetainedAttributes,
        sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
    for (auto retainAttrName : retainedAttributes) {
      StringRef attrName(retainAttrName);
      Attribute attr = srcOp->getAttr(attrName);
      if (attr) {
        newFuncOp->setAttr(attrName, attr);
      }
    }

    // Tell the rewriter to convert the region signature.
    TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    // Also add an export for the "raw" form of this function, which operates
    // on low level VM types and does no verification. A later pass will
    // materialize high level API-friendly wrappers.
    if (auto exportAttr = srcOp->getAttr("iree.module.export")) {
      StringRef exportName = newFuncOp.getName();
      if (auto exportStrAttr = exportAttr.dyn_cast<StringAttr>()) {
        exportName = exportStrAttr.getValue();
      } else {
        assert(exportAttr.isa<UnitAttr>());
      }

      rewriter.create<IREE::VM::ExportOp>(srcOp.getLoc(), newFuncOp,
                                          exportName);
    }

    rewriter.replaceOp(srcOp, llvm::None);
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<mlir::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::ReturnOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(srcOp, operands);
    return success();
  }
};

class ConstantOpConversion : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(#2878): use getTypeConverter() when we pass it upon creation.
    IREE::VM::TypeConverter typeConverter(
        IREE::VM::getTargetOptionsFromFlags());
    auto targetType = typeConverter.convertType(srcOp.getType());
    if (targetType.isa<IntegerType>()) {
      auto integerAttr = srcOp.getValue().dyn_cast<IntegerAttr>();
      if (!integerAttr) {
        return srcOp.emitRemark() << "unsupported const type for dialect";
      }
      switch (targetType.getIntOrFloatBitWidth()) {
        case 1:
        case 32:
          if (integerAttr.getInt()) {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(
                srcOp, integerAttr.getInt());
          } else {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
          }
          break;
        case 64:
          if (integerAttr.getInt()) {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstI64Op>(
                srcOp, integerAttr.getInt());
          } else {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstI64ZeroOp>(srcOp);
          }
          break;
        default:
          return srcOp.emitRemark()
                 << "unsupported const integer bit width for dialect";
      }
    } else if (targetType.isa<FloatType>()) {
      auto floatAttr = srcOp.getValue().dyn_cast<FloatAttr>();
      if (!floatAttr) {
        return srcOp.emitRemark() << "unsupported const type for dialect";
      }
      switch (targetType.getIntOrFloatBitWidth()) {
        case 32:
          if (floatAttr.getValue().isZero()) {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstF32ZeroOp>(srcOp);
          } else {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstF32Op>(srcOp, floatAttr);
          }
          break;
        case 64:
          if (floatAttr.getValue().isZero()) {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstF64ZeroOp>(srcOp);
          } else {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstF64Op>(srcOp, floatAttr);
          }
          break;
        default:
          return srcOp.emitRemark()
                 << "unsupported const floating-point bit width for dialect";
      }
    }
    return success();
  }
};

class CmpIOpConversion : public OpConversionPattern<CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpIOp::Adaptor srcAdaptor(operands);
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case CmpIPredicate::eq:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI32Op>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ne:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI32Op>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::slt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::sle:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::sgt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::sge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32SOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ult:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ule:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::ugt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      case CmpIPredicate::uge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        return success();
      default:
        return failure();
    }
  }
};

class CmpFOpConversion : public OpConversionPattern<CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpFOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpFOp::Adaptor srcAdaptor(operands);
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case CmpFPredicate::AlwaysFalse:  // 0
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
        break;
      case CmpFPredicate::AlwaysTrue:  // 1
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(srcOp, 1);
        break;
      case CmpFPredicate::UNO:  // isnan(lhs) || isnan(rhs)
        rewriter.replaceOpWithNewOp<IREE::VM::OrI32Op>(
            srcOp, returnType,
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, srcAdaptor.lhs()),
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, srcAdaptor.rhs()));
        break;
      case CmpFPredicate::ORD:  // !(isnan(lhs) || isnan(rhs))
        rewriter.replaceOpWithNewOp<IREE::VM::XorI32Op>(
            srcOp, returnType,
            rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1),
            rewriter.createOrFold<IREE::VM::AndI32Op>(
                srcOp.getLoc(), returnType,
                rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                    srcOp.getLoc(), returnType, srcAdaptor.lhs()),
                rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                    srcOp.getLoc(), returnType, srcAdaptor.rhs())));
        break;
      case CmpFPredicate::OEQ:  // ordered and equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OGT:  // ordered and greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OGE:  // ordered and greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OLT:  // ordered and less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::OLE:  // ordered and less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::ONE:  // ordered and not equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32OOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UEQ:  // unordered or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UGT:  // unordered or greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UGE:  // unordered or greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::ULT:  // unordered or less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::ULE:  // unordered or less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      case CmpFPredicate::UNE:  // unordered or not equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32UOp>(
            srcOp, returnType, srcAdaptor.lhs(), srcAdaptor.rhs());
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unhandled CmpFPredicate");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class UnaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(operands);
    switch (srcAdaptor.operand().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, srcAdaptor.operand().getType(), srcAdaptor.operand());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, srcAdaptor.operand().getType(), srcAdaptor.operand());
        break;
      default:
        llvm_unreachable("invalid target type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(operands);
    switch (srcAdaptor.lhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, srcAdaptor.lhs().getType(), srcAdaptor.lhs(),
            srcAdaptor.rhs());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, srcAdaptor.lhs().getType(), srcAdaptor.lhs(),
            srcAdaptor.rhs());
        break;
      default:
        llvm_unreachable("invalid target type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(operands);
    Value amount = srcAdaptor.rhs();
    if (amount.getType().getIntOrFloatBitWidth() > 32) {
      // Shift amounts are always 32-bit in the VM.
      amount = rewriter.createOrFold<TruncateIOp>(
          srcOp.getLoc(), rewriter.getI32Type(), amount);
    }
    switch (srcAdaptor.lhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(srcOp, srcOp.getType(),
                                               srcAdaptor.lhs(), amount);
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(srcOp, srcOp.getType(),
                                               srcAdaptor.lhs(), amount);
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename StdOp>
class CastingOpConversion : public OpConversionPattern<StdOp> {
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StdOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(srcOp, operands);
    return success();
  }
};

class SIToFPOpConversion : public OpConversionPattern<SIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SIToFPOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SIToFPOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isSignlessInteger(32) || srcType.isSignedInteger(32)) {
      if (dstType.isF32()) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastSI32F32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class UIToFPOpConversion : public OpConversionPattern<UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UIToFPOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    UIToFPOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isUnsignedInteger(32)) {
      if (dstType.isF32()) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastUI32F32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class FPToSIOpConversion : public OpConversionPattern<FPToSIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FPToSIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FPToSIOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32()) {
      if (dstType.isSignlessInteger(32) || dstType.isSignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32SI32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class FPToUIOpConversion : public OpConversionPattern<FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FPToUIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FPToUIOpAdaptor srcAdaptor(operands);
    auto srcType = operands[0].getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32()) {
      if (srcType.isUnsignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32UI32Op>(srcOp, dstType,
                                                             operands[0]);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class SelectOpConversion : public OpConversionPattern<SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SelectOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SelectOp::Adaptor srcAdaptor(operands);
    auto valueType = srcAdaptor.true_value().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI32Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI64Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF32Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF64Op>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else if (valueType.isa<IREE::VM::RefType>()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectRefOp>(
          srcOp, valueType, srcAdaptor.condition(), srcAdaptor.true_value(),
          srcAdaptor.false_value());
      return success();
    } else {
      return rewriter.notifyMatchFailure(srcOp,
                                         "unsupported select element type");
    }
  }
};

class BranchOpConversion : public OpConversionPattern<BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BranchOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(srcOp, srcOp.getDest(),
                                                    operands);
    return success();
  }
};

class CondBranchOpConversion : public OpConversionPattern<CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CondBranchOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Block *trueDest = srcOp.getTrueDest();
    rewriter.replaceOpWithNewOp<IREE::VM::CondBranchOp>(
        srcOp, operands[0], trueDest,
        operands.slice(1, trueDest->getNumArguments()), srcOp.getFalseDest(),
        operands.slice(1 + trueDest->getNumArguments()));
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CallOp::Adaptor srcAdaptor(operands);
    // Convert function result types. The conversion framework will ensure
    // that the callee has been equivalently converted.
    SmallVector<Type, 4> resultTypes;
    for (auto resultType : srcOp.getResultTypes()) {
      resultType = getTypeConverter()->convertType(resultType);
      if (!resultType) {
        return failure();
      }
      resultTypes.push_back(resultType);
    }
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        srcOp, srcOp.getCallee(), resultTypes, srcAdaptor.operands());

    return success();
  }
};

}  // namespace

void populateStandardToVMPatterns(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  OwningRewritePatternList &patterns) {
  patterns.insert<BranchOpConversion, CallOpConversion, CmpIOpConversion,
                  CmpFOpConversion, CondBranchOpConversion, ModuleOpConversion,
                  FuncOpConversion, ReturnOpConversion,
                  CastingOpConversion<IndexCastOp>,
                  CastingOpConversion<TruncateIOp>,
                  CastingOpConversion<ZeroExtendIOp>, SelectOpConversion>(
      typeConverter, context);
  // TODO(#2878): pass typeConverter here.
  patterns.insert<ConstantOpConversion>(context);

  // Integer arithmetic ops.
  patterns.insert<
      BinaryArithmeticOpConversion<AddIOp, IREE::VM::AddI32Op,
                                   IREE::VM::AddI64Op>,
      BinaryArithmeticOpConversion<SignedDivIOp, IREE::VM::DivI32SOp,
                                   IREE::VM::DivI64SOp>,
      BinaryArithmeticOpConversion<UnsignedDivIOp, IREE::VM::DivI32UOp,
                                   IREE::VM::DivI64UOp>,
      BinaryArithmeticOpConversion<MulIOp, IREE::VM::MulI32Op,
                                   IREE::VM::MulI64Op>,
      BinaryArithmeticOpConversion<SignedRemIOp, IREE::VM::RemI32SOp,
                                   IREE::VM::RemI64SOp>,
      BinaryArithmeticOpConversion<UnsignedRemIOp, IREE::VM::RemI32UOp,
                                   IREE::VM::RemI64UOp>,
      BinaryArithmeticOpConversion<SubIOp, IREE::VM::SubI32Op,
                                   IREE::VM::SubI64Op>,
      BinaryArithmeticOpConversion<AndOp, IREE::VM::AndI32Op,
                                   IREE::VM::AndI64Op>,
      BinaryArithmeticOpConversion<OrOp, IREE::VM::OrI32Op, IREE::VM::OrI64Op>,
      BinaryArithmeticOpConversion<XOrOp, IREE::VM::XorI32Op,
                                   IREE::VM::XorI64Op>>(typeConverter, context);

  // Floating-point arithmetic ops.
  patterns.insert<BinaryArithmeticOpConversion<AddFOp, IREE::VM::AddF32Op,
                                               IREE::VM::AddF64Op>,
                  BinaryArithmeticOpConversion<DivFOp, IREE::VM::DivF32Op,
                                               IREE::VM::DivF64Op>,
                  BinaryArithmeticOpConversion<MulFOp, IREE::VM::MulF32Op,
                                               IREE::VM::MulF64Op>,
                  BinaryArithmeticOpConversion<RemFOp, IREE::VM::RemF32Op,
                                               IREE::VM::RemF64Op>,
                  BinaryArithmeticOpConversion<SubFOp, IREE::VM::SubF32Op,
                                               IREE::VM::SubF64Op>>(
      typeConverter, context);

  // Floating-point conversion ops.
  patterns.insert<SIToFPOpConversion, UIToFPOpConversion, FPToUIOpConversion,
                  FPToUIOpConversion>(typeConverter, context);

  // Shift ops.
  patterns.insert<
      ShiftArithmeticOpConversion<ShiftLeftOp, IREE::VM::ShlI32Op,
                                  IREE::VM::ShlI64Op>,
      ShiftArithmeticOpConversion<SignedShiftRightOp, IREE::VM::ShrI32SOp,
                                  IREE::VM::ShrI64SOp>,
      ShiftArithmeticOpConversion<UnsignedShiftRightOp, IREE::VM::ShrI32UOp,
                                  IREE::VM::ShrI64UOp>>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
