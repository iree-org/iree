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
#include "mlir/Dialect/StandardOps/Ops.h"
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

  PatternMatchResult matchAndRewrite(
      ModuleOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert the top level module.
    // This mechanism can only support rewriting non top-level modules.
    if (!srcOp.getParentOp()) {
      return matchFailure();
    }

    StringRef name = srcOp.getName() ? *srcOp.getName() : "module";
    auto newModuleOp =
        rewriter.create<IREE::VM::ModuleOp>(srcOp.getLoc(), name);
    newModuleOp.getBodyRegion().takeBody(srcOp.getBodyRegion());

    // Replace the terminator.
    Operation *srcTerminator =
        newModuleOp.getBodyRegion().back().getTerminator();
    rewriter.setInsertionPointToEnd(&newModuleOp.getBodyRegion().back());
    rewriter.replaceOpWithNewOp<IREE::VM::ModuleEndOp>(srcTerminator);

    rewriter.eraseOp(srcOp);
    return matchSuccess();
  }
};

// Whitelist of function attributes to retain when converting to vm.func.
constexpr std::array<const char *, 1> kRetainedAttributes = {
    "iree.reflection",
};

class FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      FuncOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getType();
    VMTypeConverter typeConverter;
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
      if (failed(typeConverter.convertSignatureArg(i, srcFuncType.getInput(i),
                                                   signatureConversion))) {
        return matchFailure();
      }
    }

    // Convert function results.
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(typeConverter.convertTypes(srcFuncType.getResults(),
                                          convertedResultTypes))) {
      return matchFailure();
    }

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncType =
        mlir::FunctionType::get(signatureConversion.getConvertedTypes(),
                                convertedResultTypes, srcOp.getContext());
    auto newFuncOp = rewriter.create<IREE::VM::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), newFuncType);

    // Retain function attributes in the whitelist.
    for (auto retainAttrName : kRetainedAttributes) {
      StringRef attrName(retainAttrName);
      Attribute attr = srcOp.getAttr(attrName);
      if (attr) {
        newFuncOp.setAttr(attrName, attr);
      }
    }

    // Move the body region from src -> new.
    auto &srcRegion = srcOp.getOperation()->getRegion(0);
    auto &newRegion = newFuncOp.getOperation()->getRegion(0);
    newRegion.takeBody(srcRegion);

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(),
                                      signatureConversion);

    // Also add an export if the function is tagged. Ideally this would be
    // replaced with river sym_vis magic.
    if (srcOp.getAttr("iree.module.export")) {
      rewriter.create<IREE::VM::ExportOp>(srcOp.getLoc(), newFuncOp);
    }

    rewriter.replaceOp(srcOp, llvm::None);
    return matchSuccess();
  }
};

class ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      ReturnOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(srcOp, operands);
    return matchSuccess();
  }
};

class ConstantOpConversion : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      ConstantOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto integerAttr = srcOp.getValue().dyn_cast<IntegerAttr>();
    // Only 32bit integer supported for now.
    if (!integerAttr) {
      srcOp.emitRemark() << "unsupported const type for dialect";
      return matchFailure();
    }
    int numBits = integerAttr.getType().getIntOrFloatBitWidth();
    if (numBits != 1 && numBits != 32) {
      srcOp.emitRemark() << "unsupported bit width for dialect constant";
      return matchFailure();
    }

    auto intValue = integerAttr.getInt();
    if (intValue == 0) {
      rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
    } else {
      rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(srcOp, intValue);
    }
    return matchSuccess();
  }
};

class CmpIOpConversion : public OpConversionPattern<CmpIOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      CmpIOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CmpIOpOperandAdaptor srcAdapter(operands);
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case CmpIPredicate::eq:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI32Op>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::ne:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI32Op>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::slt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::sle:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::sgt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::sge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32SOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::ult:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::ule:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::ugt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
      case CmpIPredicate::uge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32UOp>(
            srcOp, returnType, srcAdapter.lhs(), srcAdapter.rhs());
        return matchSuccess();
    }
  }
};

template <typename SrcOpTy, typename DstOpTy>
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpConversionPattern<SrcOpTy>::matchSuccess;

  PatternMatchResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::OperandAdaptor srcAdapter(operands);

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(),
                                         srcAdapter.lhs(), srcAdapter.rhs());
    return matchSuccess();
  }
};

template <typename SrcOpTy, typename DstOpTy, unsigned kBits = 32>
class ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpConversionPattern<SrcOpTy>::matchFailure;
  using OpConversionPattern<SrcOpTy>::matchSuccess;

  PatternMatchResult matchAndRewrite(
      SrcOpTy srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    typename SrcOpTy::OperandAdaptor srcAdaptor(operands);
    auto type = srcOp.getType();
    if (!type.template isa<IntegerType>() ||
        type.getIntOrFloatBitWidth() != kBits) {
      return matchFailure();
    }
    APInt amount;
    if (!matchPattern(srcAdaptor.rhs(), m_ConstantInt(&amount))) {
      return matchFailure();
    }
    uint64_t amountRaw = amount.getZExtValue();
    if (amountRaw > kBits) return matchFailure();
    IntegerAttr amountAttr =
        IntegerAttr::get(IntegerType::get(8, srcOp.getContext()), amountRaw);
    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(),
                                         srcAdaptor.lhs(), amountAttr);
    return matchSuccess();
  }
};

class SelectI32OpConversion : public OpConversionPattern<SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  static constexpr unsigned kBits = 32;

  PatternMatchResult matchAndRewrite(
      SelectOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    SelectOpOperandAdaptor srcAdaptor(operands);
    IntegerType requiredType = IntegerType::get(kBits, srcOp.getContext());
    if (srcAdaptor.true_value().getType() != requiredType)
      return matchFailure();

    rewriter.replaceOpWithNewOp<IREE::VM::SelectI32Op>(
        srcOp, requiredType, srcAdaptor.condition(), srcAdaptor.true_value(),
        srcAdaptor.false_value());
    return matchSuccess();
  }
};

class BranchOpConversion : public OpConversionPattern<BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      BranchOp srcOp, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    assert(destinations.size() == 1 && operands.size() == 1);
    rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(srcOp, destinations[0],
                                                    operands[0]);
    return matchSuccess();
  }
};

class CondBranchOpConversion : public OpConversionPattern<CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      CondBranchOp srcOp, ArrayRef<Value> properOperands,
      ArrayRef<Block *> destinations, ArrayRef<ArrayRef<Value>> operands,
      ConversionPatternRewriter &rewriter) const override {
    assert(destinations.size() == 2 && operands.size() == 2);
    CondBranchOpOperandAdaptor srcAdaptor(properOperands);
    rewriter.replaceOpWithNewOp<IREE::VM::CondBranchOp>(
        srcOp, srcAdaptor.condition(), destinations[0], operands[0],  // true
        destinations[1], operands[1]);                                // false;
    return matchSuccess();
  }
};

class CallOpConversion : public OpConversionPattern<CallOp> {
  using OpConversionPattern::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      CallOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    CallOpOperandAdaptor srcAdaptor(operands);
    // Convert function result types. The conversion framework will ensure
    // that the callee has been equivalently converted.
    VMTypeConverter typeConverter;
    SmallVector<Type, 4> resultTypes;
    for (auto resultType : srcOp.getResultTypes()) {
      resultType = typeConverter.convertType(resultType);
      if (!resultType) {
        return matchFailure();
      }
      resultTypes.push_back(resultType);
    }
    rewriter.replaceOpWithNewOp<IREE::VM::CallOp>(
        srcOp, srcOp.getCallee(), resultTypes, srcAdaptor.operands());

    return matchSuccess();
  }
};

}  // namespace

void populateStandardToVMPatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns) {
  patterns
      .insert<BranchOpConversion, CallOpConversion, CmpIOpConversion,
              CondBranchOpConversion, ConstantOpConversion, ModuleOpConversion,
              FuncOpConversion, ReturnOpConversion, SelectI32OpConversion>(
          context);

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
