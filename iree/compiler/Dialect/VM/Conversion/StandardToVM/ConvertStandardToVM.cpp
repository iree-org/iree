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
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
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
    if (!srcOp.getParentOp() || !isa<ModuleOp>(srcOp.getParentOp())) {
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
    return success();
  }
};

class ModuleTerminatorOpConversion
    : public OpConversionPattern<ModuleTerminatorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ModuleTerminatorOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert the top level module's terminator.
    // This mechanism can only support rewriting non top-level modules.
    if (!isa<IREE::VM::ModuleOp>(srcOp.getParentOp())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::VM::ModuleTerminatorOp>(srcOp);
    return success();
  }
};

// Allowlist of function attributes to retain when converting to vm.func.
constexpr const char *kRetainedAttributes[] = {
    "iree.reflection",
    "sym_visibility",
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
      if (failed(typeConverter.convertSignatureArg(i, srcFuncType.getInput(i),
                                                   signatureConversion))) {
        return failure();
      }
    }

    // Convert function results.
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(typeConverter.convertTypes(srcFuncType.getResults(),
                                          convertedResultTypes))) {
      return failure();
    }

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncType =
        mlir::FunctionType::get(signatureConversion.getConvertedTypes(),
                                convertedResultTypes, srcOp.getContext());
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
      Attribute attr = srcOp.getAttr(attrName);
      if (attr) {
        newFuncOp.setAttr(attrName, attr);
      }
    }

    // Tell the rewriter to convert the region signature.
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    // Also add an export for the "raw" form of this function, which operates
    // on low level VM types and does no verification. A later pass will
    // materialize high level API-friendly wrappers.
    if (auto exportAttr = srcOp.getAttr("iree.module.export")) {
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

  mutable VMTypeConverter typeConverter;
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
    auto integerAttr = srcOp.getValue().dyn_cast<IntegerAttr>();
    // Only 32bit integer supported for now.
    if (!integerAttr) {
      srcOp.emitRemark() << "unsupported const type for dialect";
      return failure();
    }
    if (integerAttr.getType().isIntOrFloat()) {
      int numBits = integerAttr.getType().getIntOrFloatBitWidth();
      if (numBits != 1 && numBits != 32) {
        srcOp.emitRemark() << "unsupported bit width for dialect constant";
        return failure();
      }
    }

    auto intValue = integerAttr.getInt();
    if (intValue == 0) {
      rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
    } else {
      rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(srcOp, intValue);
    }
    return success();
  }
};

class CmpIOpConversion : public OpConversionPattern<CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpIOp::Adaptor srcAdapter(operands);
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case CmpIPredicate::eq:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI32Op>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::ne:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI32Op>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::slt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::sle:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::sgt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::sge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::ult:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::ule:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::ugt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      case CmpIPredicate::uge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return success();
      default:
        return failure();
    }
  }
};

template <typename SrcOpTy, typename DstOpTy>
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdapter(operands);

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcAdapter.lhs().getType(),
                                         srcAdapter.lhs(), srcAdapter.rhs());
    return success();
  }
};

template <typename SrcOpTy, typename DstOpTy, unsigned kBits = 32>
class ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::Adaptor srcAdaptor(operands);
    auto type = srcOp.getType();
    if (!type.isSignlessInteger() || type.getIntOrFloatBitWidth() != kBits) {
      return failure();
    }
    APInt amount;
    if (!matchPattern(srcAdaptor.rhs(), m_ConstantInt(&amount))) {
      return failure();
    }
    uint64_t amountRaw = amount.getZExtValue();
    if (amountRaw > kBits) return failure();
    IntegerAttr amountAttr =
        IntegerAttr::get(IntegerType::get(8, srcOp.getContext()), amountRaw);
    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(),
                                         srcAdaptor.lhs(), amountAttr);
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

class SelectI32OpConversion : public OpConversionPattern<SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SelectOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SelectOp::Adaptor srcAdaptor(operands);
    IntegerType requiredType = IntegerType::get(32, srcOp.getContext());
    // Note: This check can correctly just be a verification that
    // actualType == requiredType, but since the VM type conversion also
    // maps Indextype to this type, widening the check here reduces red-herrings
    // when other conversions fail to properly match/rewrite index related ops.
    // (Otherwise, the dialect converter may report the error as a failure to
    // legalize the select op depending on order of resolution).
    auto actualType = srcAdaptor.true_value().getType();
    if (actualType != requiredType && actualType.isa<IndexType>())
      return failure();

    rewriter.replaceOpWithNewOp<IREE::VM::SelectI32Op>(
        srcOp, requiredType, srcAdaptor.condition(), srcAdaptor.true_value(),
        srcAdaptor.false_value());
    return success();
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
    VMTypeConverter typeConverter;
    SmallVector<Type, 4> resultTypes;
    for (auto resultType : srcOp.getResultTypes()) {
      resultType = typeConverter.convertType(resultType);
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
                                  OwningRewritePatternList &patterns) {
  patterns
      .insert<BranchOpConversion, CallOpConversion, CmpIOpConversion,
              CondBranchOpConversion, ConstantOpConversion, ModuleOpConversion,
              ModuleTerminatorOpConversion, FuncOpConversion,
              ReturnOpConversion, CastingOpConversion<IndexCastOp>,
              CastingOpConversion<TruncateIOp>, SelectI32OpConversion>(context);

  // Binary arithmetic ops
  patterns
      .insert<BinaryArithmeticOpConversion<AddIOp, IREE::VM::AddI32Op>,
              BinaryArithmeticOpConversion<SignedDivIOp, IREE::VM::DivI32SOp>,
              BinaryArithmeticOpConversion<UnsignedDivIOp, IREE::VM::DivI32UOp>,
              BinaryArithmeticOpConversion<MulIOp, IREE::VM::MulI32Op>,
              BinaryArithmeticOpConversion<SignedRemIOp, IREE::VM::RemI32SOp>,
              BinaryArithmeticOpConversion<UnsignedRemIOp, IREE::VM::RemI32UOp>,
              BinaryArithmeticOpConversion<SubIOp, IREE::VM::SubI32Op>,
              BinaryArithmeticOpConversion<AndOp, IREE::VM::AndI32Op>,
              BinaryArithmeticOpConversion<OrOp, IREE::VM::OrI32Op>,
              BinaryArithmeticOpConversion<XOrOp, IREE::VM::XorI32Op>>(context);

  // Shift ops
  // TODO(laurenzo): The standard dialect is missing shr ops. Add once in place.
  patterns.insert<ShiftArithmeticOpConversion<ShiftLeftOp, IREE::VM::ShlI32Op>>(
      context);
}

}  // namespace iree_compiler
}  // namespace mlir
