// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/ConvertStandardToVM.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
      ModuleOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Do not attempt to convert the top level module.
    // This mechanism can only support rewriting non top-level modules.
    if (VMConversionTarget::isTopLevelModule(srcOp)) {
      return failure();
    }

    StringRef name = srcOp.getName() ? *srcOp.getName() : "module";
    auto newModuleOp =
        rewriter.create<IREE::VM::ModuleOp>(srcOp.getLoc(), name);
    assert(!newModuleOp.getBodyRegion().empty());
    if (auto version = srcOp->getAttrOfType<IntegerAttr>("vm.version")) {
      newModuleOp.setVersionAttr(version);
    }
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

// Converts a function signature with the given |signatureConversion| util.
static FailureOr<FunctionType> convertFuncSignature(
    func::FuncOp srcOp, TypeConverter &typeConverter,
    TypeConverter::SignatureConversion &signatureConversion,
    ConversionPatternRewriter &rewriter) {
  FunctionType srcFuncType = srcOp.getFunctionType();
  for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
    if (failed(typeConverter.convertSignatureArg(i, srcFuncType.getInput(i),
                                                 signatureConversion))) {
      return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
    }
  }
  SmallVector<Type, 1> convertedResultTypes;
  if (failed(typeConverter.convertTypes(srcFuncType.getResults(),
                                        convertedResultTypes))) {
    return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
  }
  return mlir::FunctionType::get(srcOp.getContext(),
                                 signatureConversion.getConvertedTypes(),
                                 convertedResultTypes);
}

// Copies attributes from |srcOp| to |dstOp| that we preserve during conversion.
// There may be any number of attributes on srcOp but we don't always want to
// preserve them as they may come from dialects we are removing.
static void copyFuncAttrs(func::FuncOp srcOp, Operation *dstOp) {
  constexpr const char *kRetainedAttributes[] = {
      "iree.reflection",
      "sym_visibility",
      "noinline",
      "nosideeffects",
  };
  auto retainedAttributes = ArrayRef<const char *>(
      kRetainedAttributes,
      sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
  for (auto retainAttrName : retainedAttributes) {
    StringRef attrName(retainAttrName);
    Attribute attr = srcOp->getAttr(attrName);
    if (attr) {
      dstOp->setAttr(attrName, attr);
    }
  }
}

class FuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::FuncOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Handled by import-specific conversion.
    if (srcOp.isExternal()) return failure();

    // Convert function signature.
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());
    auto newFuncType = convertFuncSignature(srcOp, *getTypeConverter(),
                                            signatureConversion, rewriter);
    if (failed(newFuncType)) return failure();

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncOp = rewriter.create<IREE::VM::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), *newFuncType);
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getFunctionBody(),
                                newFuncOp.end());

    // Tell the rewriter to convert the region signature.
    TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    // Retain function attributes in the allowlist.
    copyFuncAttrs(srcOp, newFuncOp);

    // Also add an export for the "raw" form of this function, which operates
    // on low level VM types and does no verification. A later pass will
    // materialize high level API-friendly wrappers.
    if (srcOp.isPublic()) {
      StringRef exportName = newFuncOp.getName();
      auto exportOp = rewriter.create<IREE::VM::ExportOp>(
          srcOp.getLoc(), newFuncOp, exportName);
      exportOp->setDialectAttrs(srcOp->getDialectAttrs());
    }

    // VM functions are private by default and exported via the dedicated
    // vm.export ops.
    newFuncOp.setPrivate();

    rewriter.replaceOp(srcOp, std::nullopt);
    return success();
  }
};

// Copies attributes from |srcOp| to |dstOp| that we preserve during conversion.
// We allow external functions to have some special vm-specific attributes that
// override behavior during conversion and don't want to propagate them.
static void copyImportAttrs(func::FuncOp srcOp, IREE::VM::ImportOp dstOp) {
  constexpr const char *kRetainedAttributes[] = {
      "noinline",
      "nosideeffects",
      "vm.fallback",
  };
  auto retainedAttributes = ArrayRef<const char *>(
      kRetainedAttributes,
      sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
  for (auto retainAttrName : retainedAttributes) {
    StringRef attrName(retainAttrName);
    Attribute attr = srcOp->getAttr(attrName);
    if (attr) {
      dstOp->setAttr(attrName, attr);
    }
  }
}

class ExternalFuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::FuncOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Handled by internal-specific conversion.
    if (!srcOp.isExternal()) return failure();

    // If the user declared an intended signature then we can use that instead
    // of running conversion ourselves. This can be used in cases where the
    // types of the import differ from the expected conversion results like
    // index -> i32 or i64 depending on VM mode.
    FunctionType newSignature;
    auto signatureAttr = srcOp->getAttrOfType<TypeAttr>("vm.signature");
    if (signatureAttr) {
      // Directly use the signature from the user.
      newSignature = llvm::dyn_cast<FunctionType>(signatureAttr.getValue());
      if (!newSignature) {
        return rewriter.notifyMatchFailure(srcOp, "invalid vm.signature");
      }
    } else {
      // Convert function signature.
      TypeConverter::SignatureConversion signatureConversion(
          srcOp.getNumArguments());
      auto convertedSignature = convertFuncSignature(
          srcOp, *getTypeConverter(), signatureConversion, rewriter);
      if (failed(convertedSignature)) return failure();
      newSignature = *convertedSignature;
    }

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto importOp = rewriter.create<IREE::VM::ImportOp>(
        srcOp.getLoc(), srcOp.getName(), newSignature);
    importOp.setSymVisibilityAttr(srcOp.getSymVisibilityAttr());

    // If there is a fallback then the import is optional.
    if (srcOp->hasAttr("vm.fallback")) {
      importOp.setIsOptionalAttr(rewriter.getUnitAttr());
    }

    // By default imports are unversioned but we allow the user to specify one.
    if (auto minimumVersion = srcOp->getAttrOfType<IntegerAttr>("vm.version")) {
      importOp.setMinimumVersionAttr(minimumVersion);
    }

    // Retain function attributes in the allowlist.
    copyImportAttrs(srcOp, importOp);

    rewriter.replaceOp(srcOp, std::nullopt);
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      func::CallOp callOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Regardless of what the call (or import) does we need to match the
    // result types defined by the type converter. We will insert casts as
    // needed to these types before replacing the op.
    SmallVector<Type> resultTypes;
    for (auto resultType : callOp.getResultTypes()) {
      resultType = getTypeConverter()->convertType(resultType);
      if (!resultType) {
        return rewriter.notifyMatchFailure(callOp, "unsupported result type");
      }
      resultTypes.push_back(resultType);
    }

    // We change the call behavior based on whether we are targeting an import
    // or an internal function. This may include recursively calling the
    // conversion if imports have fallbacks that are themselves imports.
    auto callResults = convertCallOp(
        callOp->getParentOfType<IREE::VM::ModuleOp>(), callOp.getLoc(),
        callOp.getCallee(), adaptor.getOperands(), resultTypes, rewriter);
    if (failed(callResults)) {
      return rewriter.notifyMatchFailure(
          callOp, "unable to convert call (results mismatch)");
    }

    rewriter.replaceOp(callOp, *callResults);
    return success();
  }

  // Converts a call to some function which may be internal or an import.
  // Returns the new converted call results.
  FailureOr<SmallVector<Value>> convertCallOp(
      Operation *rootOp, Location loc, StringRef calleeName,
      ValueRange operands, TypeRange resultTypes,
      ConversionPatternRewriter &rewriter) const {
    // (Slow) lookup of the target function, which may be an import that we need
    // to perform type conversion for.
    auto calleeOp = SymbolTable::lookupSymbolIn(rootOp, calleeName);
    if (auto funcOp = dyn_cast_or_null<func::FuncOp>(calleeOp)) {
      if (funcOp.isExternal()) {
        // Import that may require conversion.
        // This case handles when funcs are declared after the call.
        FunctionType convertedSignature;
        if (auto signatureAttr =
                funcOp->getAttrOfType<TypeAttr>("vm.signature")) {
          if (auto importSignature =
                  llvm::dyn_cast<FunctionType>(signatureAttr.getValue())) {
            convertedSignature = importSignature;
          }
        }
        if (!convertedSignature) {
          convertedSignature =
              rewriter.getFunctionType(TypeRange(operands), resultTypes);
        }
        return convertImportCallOp(rootOp, loc, calleeName, operands,
                                   resultTypes, convertedSignature, funcOp,
                                   rewriter);
      }
    } else if (auto importOp = dyn_cast_or_null<IREE::VM::ImportOp>(calleeOp)) {
      // Calling an import.
      // This case handles when funcs are declared before the call and have
      // already been converted.
      return convertImportCallOp(rootOp, loc, calleeName, operands, resultTypes,
                                 importOp.getFunctionType(), importOp,
                                 rewriter);
    }

    // Otherwise this is a direct call to an internal function.
    auto newOp = rewriter.create<IREE::VM::CallOp>(loc, calleeName, resultTypes,
                                                   operands);
    return SmallVector<Value>(newOp.result_begin(), newOp.result_end());
  }

  // Converts a call to an import that may be optional.
  // Returns the new converted call results.
  FailureOr<SmallVector<Value>> convertImportCallOp(
      Operation *rootOp, Location loc, StringRef calleeName,
      ValueRange operands, TypeRange resultTypes, FunctionType importSignature,
      Operation *calleeOp, ConversionPatternRewriter &rewriter) const {
    auto fallbackAttr = calleeOp->getAttrOfType<SymbolRefAttr>("vm.fallback");
    return fallbackAttr
               ? convertOptionalImportCallOp(
                     rootOp, loc, calleeName, operands, resultTypes,
                     importSignature,
                     fallbackAttr.getLeafReference().getValue(), rewriter)
               : convertMandatoryImportCallOp(rootOp, loc, calleeName, operands,
                                              resultTypes, importSignature,
                                              rewriter);
  }

  // Converts a call to an optional import by adding logic to check whether it
  // resolves at runtime and otherwise calling a fallback.
  // Returns the new converted call results.
  FailureOr<SmallVector<Value>> convertOptionalImportCallOp(
      Operation *rootOp, Location loc, StringRef calleeName,
      ValueRange operands, TypeRange resultTypes, FunctionType importSignature,
      StringRef fallbackName, ConversionPatternRewriter &rewriter) const {
    // Check whether the import resolved and if so call it. Otherwise we call
    // the fallback which should not require any conversion.
    Value resolved = rewriter.create<IREE::VM::ImportResolvedOp>(
        loc, rewriter.getI32Type(), calleeName);

    // We'll be making the call via two blocks and then joining again on a block
    // that takes the results in target form.
    SmallVector<Location> resultLocs(resultTypes.size(), loc);
    auto *exitBlock = rewriter.splitBlock(rewriter.getInsertionBlock(),
                                          rewriter.getInsertionPoint());
    auto exitResults = llvm::to_vector(
        llvm::map_range(exitBlock->addArguments(resultTypes, resultLocs),
                        [](BlockArgument arg) -> Value { return arg; }));

    auto *resolvedBlock = rewriter.createBlock(exitBlock);
    auto *fallbackBlock = rewriter.createBlock(exitBlock);

    // Insert the branch to each block.
    rewriter.setInsertionPointAfterValue(resolved);
    rewriter.create<IREE::VM::CondBranchOp>(loc, resolved, resolvedBlock,
                                            ValueRange{}, fallbackBlock,
                                            ValueRange{});

    // Resolved: make call to the import as normal.
    rewriter.setInsertionPointToStart(resolvedBlock);
    auto importResults =
        convertMandatoryImportCallOp(rootOp, loc, calleeName, operands,
                                     resultTypes, importSignature, rewriter);
    rewriter.create<IREE::VM::BranchOp>(loc, exitBlock, importResults);

    // Not resolved: call fallback as a normal function.
    rewriter.setInsertionPointToStart(fallbackBlock);
    auto fallbackResults = convertCallOp(rootOp, loc, fallbackName, operands,
                                         resultTypes, rewriter);
    if (failed(fallbackResults)) return failure();
    rewriter.create<IREE::VM::BranchOp>(loc, exitBlock, *fallbackResults);

    return exitResults;
  }

  // Converts a call to a mandatory import that is guaranteed to be resolved at
  // runtime. Handles potential import signature overrides that change type
  // conversion behavior.
  // Returns the new converted call results.
  SmallVector<Value> convertMandatoryImportCallOp(
      Operation *rootOp, Location loc, StringRef calleeName,
      ValueRange operands, TypeRange resultTypes, FunctionType importSignature,
      ConversionPatternRewriter &rewriter) const {
    // Marshal arguments to import types.
    SmallVector<Value> importArgs;
    for (auto [operand, importType] :
         llvm::zip_equal(operands, importSignature.getInputs())) {
      importArgs.push_back(castToImportType(operand, importType, rewriter));
    }

    // Direct call to mandatory import.
    auto newOp = rewriter.create<IREE::VM::CallOp>(
        loc, calleeName, importSignature.getResults(), importArgs);

    // Marshal results from import types.
    SmallVector<Value> callResults;
    for (auto [importResult, resultType] :
         llvm::zip_equal(newOp.getResults(), resultTypes)) {
      callResults.push_back(
          castFromImportType(importResult, resultType, rewriter));
    }
    return callResults;
  }
};

class ReturnOpConversion : public OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::func::ReturnOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(srcOp,
                                                    adaptor.getOperands());
    return success();
  }
};

struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  TypeConverter &typeConverter;
  ConstantOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}
  LogicalResult matchAndRewrite(
      arith::ConstantOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto targetType = typeConverter.convertType(srcOp.getType());
    if (!targetType) {
      return srcOp.emitError() << "could not convert type: " << srcOp.getType()
                               << " (check -iree-vm-target-* options)";
    }
    if (llvm::isa<IntegerType>(targetType)) {
      auto integerAttr = llvm::dyn_cast<IntegerAttr>(srcOp.getValue());
      if (!integerAttr) {
        return srcOp.emitRemark() << "unsupported const type for dialect";
      }
      switch (targetType.getIntOrFloatBitWidth()) {
        case 1:
        case 32:
          if (integerAttr.getInt()) {
            rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(
                srcOp,
                integerAttr.getType().isInteger(1) ? 1 : integerAttr.getInt());
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
    } else if (llvm::isa<FloatType>(targetType)) {
      auto floatAttr = llvm::dyn_cast<FloatAttr>(srcOp.getValue());
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
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

struct CmpI32OpConversion : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::CmpIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLhs().getType().isInteger(32)) return failure();
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI32Op>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ne:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI32Op>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::slt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::sle:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::sgt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::sge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ult:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ule:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ugt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::uge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      default:
        return failure();
    }
  }
};

struct CmpI64OpConversion : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::CmpIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLhs().getType().isInteger(64)) return failure();
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQI64Op>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ne:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEI64Op>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::slt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI64SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::sle:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI64SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::sgt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI64SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::sge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI64SOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ult:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTI64UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ule:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEI64UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::ugt:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTI64UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      case arith::CmpIPredicate::uge:
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEI64UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        return success();
      default:
        return failure();
    }
  }
};

struct CmpF32OpConversion : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::CmpFOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getLhs().getType().isF32()) return failure();
    auto returnType = rewriter.getIntegerType(32);
    switch (srcOp.getPredicate()) {
      case arith::CmpFPredicate::AlwaysFalse:  // 0
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32ZeroOp>(srcOp);
        break;
      case arith::CmpFPredicate::AlwaysTrue:  // 1
        rewriter.replaceOpWithNewOp<IREE::VM::ConstI32Op>(srcOp, 1);
        break;
      case arith::CmpFPredicate::UNO:  // isnan(lhs) || isnan(rhs)
        rewriter.replaceOpWithNewOp<IREE::VM::OrI32Op>(
            srcOp, returnType,
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, adaptor.getLhs()),
            rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                srcOp.getLoc(), returnType, adaptor.getRhs()));
        break;
      case arith::CmpFPredicate::ORD:  // !(isnan(lhs) || isnan(rhs))
        rewriter.replaceOpWithNewOp<IREE::VM::XorI32Op>(
            srcOp, returnType,
            rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1),
            rewriter.createOrFold<IREE::VM::AndI32Op>(
                srcOp.getLoc(), returnType,
                rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                    srcOp.getLoc(), returnType, adaptor.getLhs()),
                rewriter.createOrFold<IREE::VM::CmpNaNF32Op>(
                    srcOp.getLoc(), returnType, adaptor.getRhs())));
        break;
      case arith::CmpFPredicate::OEQ:  // ordered and equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OGT:  // ordered and greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OGE:  // ordered and greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OLT:  // ordered and less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::OLE:  // ordered and less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::ONE:  // ordered and not equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32OOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UEQ:  // unordered or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpEQF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UGT:  // unordered or greater than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UGE:  // unordered or greater than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpGTEF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::ULT:  // unordered or less than
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::ULE:  // unordered or less than or equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpLTEF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      case arith::CmpFPredicate::UNE:  // unordered or not equal
        rewriter.replaceOpWithNewOp<IREE::VM::CmpNEF32UOp>(
            srcOp, returnType, adaptor.getLhs(), adaptor.getRhs());
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp,
                                           "unhandled arith::CmpFPredicate");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class UnaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    switch (adaptor.getOperand().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, adaptor.getOperand().getType(), adaptor.getOperand());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, adaptor.getOperand().getType(), adaptor.getOperand());
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class BinaryArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    switch (adaptor.getLhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(
            srcOp, adaptor.getLhs().getType(), adaptor.getLhs(),
            adaptor.getRhs());
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(
            srcOp, adaptor.getLhs().getType(), adaptor.getLhs(),
            adaptor.getRhs());
        break;
      default:
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    return success();
  }
};

template <typename SrcOpTy, typename Dst32OpTy, typename Dst64OpTy>
class ShiftArithmeticOpConversion : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      SrcOpTy srcOp, typename SrcOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value amount = adaptor.getRhs();
    if (amount.getType().getIntOrFloatBitWidth() > 32) {
      // Shift amounts are always 32-bit in the VM.
      amount = rewriter.createOrFold<arith::TruncIOp>(
          srcOp.getLoc(), rewriter.getI32Type(), amount);
    }
    switch (adaptor.getLhs().getType().getIntOrFloatBitWidth()) {
      case 32:
        rewriter.replaceOpWithNewOp<Dst32OpTy>(srcOp, rewriter.getI32Type(),
                                               adaptor.getLhs(), amount);
        break;
      case 64:
        rewriter.replaceOpWithNewOp<Dst64OpTy>(srcOp, rewriter.getI64Type(),
                                               adaptor.getLhs(), amount);
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
      StdOp srcOp, typename StdOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(srcOp, adaptor.getOperands());
    return success();
  }
};

template <typename OpTy, typename ExtOpTy>
class IndexCastOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy srcOp, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(jpienaar): Audit and fix if needed.
    auto srcType = adaptor.getIn().getType();
    auto dstType =
        this->getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType == dstType) {
      rewriter.replaceOp(srcOp, adaptor.getOperands());
    } else if (srcType.getIntOrFloatBitWidth() <
               dstType.getIntOrFloatBitWidth()) {
      rewriter.replaceOpWithNewOp<ExtOpTy>(srcOp, dstType, adaptor.getIn());
    } else {
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(srcOp, dstType,
                                                   adaptor.getIn());
    }
    return success();
  }
};

class ZeroExtendIOpConversion : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::ExtUIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(1) && dstType.isInteger(32)) {
      // This may not be needed but ensures that the input was treated as a
      // single bit.
      // NOTE: this may not be required - if we know that the i1 is never able
      // to have more than bit 0 manipulated then this is wasted work.
      rewriter.replaceOpWithNewOp<IREE::VM::AndI32Op>(
          srcOp, dstType, adaptor.getIn(),
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    } else if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32UOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(8) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I64UOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32UOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64UOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported zero extension");
    }
    return success();
  }
};

class SignExtendIOpConversion : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::ExtSIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isInteger(8) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I32SOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(8) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI8I64SOp>(srcOp, dstType,
                                                         adaptor.getIn());
    } else if (srcType.isInteger(16) && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI16I32SOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::ExtI32I64SOp>(srcOp, dstType,
                                                          adaptor.getIn());
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported sign extension");
    }
    return success();
  }
};

class TruncateIOpConversion : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::TruncIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto resultType = srcOp.getResult().getType();
    auto dstType = getTypeConverter()->convertType(resultType);
    if (resultType.isInteger(1)) {
      // i1 is represented as i32, so just mask off the bit and truncate as
      // normal. Note that if we started as i64 we need to first get that into
      // an i32 that we can work with.
      auto value = adaptor.getIn();
      if (srcType.isInteger(64)) {
        value = rewriter.createOrFold<IREE::VM::TruncI64I32Op>(srcOp.getLoc(),
                                                               dstType, value);
      }
      rewriter.replaceOpWithNewOp<IREE::VM::AndI32Op>(
          srcOp, dstType, value,
          rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    } else if (srcType.isInteger(32) && resultType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI32I8Op>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(32) && resultType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI32I16Op>(srcOp, dstType,
                                                           adaptor.getIn());
    } else if (srcType.isInteger(64) && resultType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I8Op>(srcOp, dstType,
                                                          adaptor.getIn());
    } else if (srcType.isInteger(64) && resultType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I16Op>(srcOp, dstType,
                                                           adaptor.getIn());
    } else if (srcType.isInteger(64) && resultType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::TruncI64I32Op>(srcOp, dstType,
                                                           adaptor.getIn());
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported truncation");
    }
    return success();
  }
};

template <typename OpTy, typename ExtOpTy, typename CastOpTy>
class IntToFPOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy srcOp, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    if (!dstType.isF32() ||
        !(srcType.isSignedInteger() || srcType.isSignlessInteger())) {
      return rewriter.notifyMatchFailure(srcOp, "unsupported type");
    }
    Value input = srcOp.getIn();
    if (!(srcType.isSignlessInteger(32) || srcType.isSignedInteger(32))) {
      if (srcType.getIntOrFloatBitWidth() < 32) {
        input = rewriter.create<ExtOpTy>(
            srcOp.getLoc(), IntegerType::get(this->getContext(), 32), input);
      } else {
        return rewriter.notifyMatchFailure(srcOp, "unsupported type");
      }
    }

    auto resultType = this->getTypeConverter()->convertType(dstType);
    rewriter.replaceOpWithNewOp<CastOpTy>(srcOp, resultType, input);
    return success();
  }
};

class FPToSIOpConversion : public OpConversionPattern<arith::FPToSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::FPToSIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    auto resultType = getTypeConverter()->convertType(dstType);
    if (srcType.isF32()) {
      // This uses the resultType rather than dstType as any truncation
      // required will be handled via interpretation by consumer.
      if (resultType.isSignlessInteger(32) || resultType.isSignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32SI32Op>(srcOp, resultType,
                                                             adaptor.getIn());
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class FPToUIOpConversion : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::FPToUIOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    auto resultType = getTypeConverter()->convertType(dstType);
    if (srcType.isF32()) {
      if (dstType.isSignlessInteger(32) || dstType.isUnsignedInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::CastF32UI32Op>(srcOp, resultType,
                                                             adaptor.getIn());
        return success();
      }
    }
    return rewriter.notifyMatchFailure(srcOp, "unsupported type");
  }
};

class BitcastOpConversion : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::BitcastOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto srcType = srcOp.getIn().getType();
    auto dstType = srcOp.getResult().getType();
    auto resultType =
        getTypeConverter()->convertType(srcOp.getResult().getType());
    if (srcType.isF32() && dstType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastF32I32Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else if (srcType.isInteger(32) && dstType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastI32F32Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else if (srcType.isF64() && dstType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastF64I64Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else if (srcType.isInteger(64) && dstType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BitcastI64F64Op>(
          srcOp, resultType, adaptor.getOperands()[0]);
    } else {
      return rewriter.notifyMatchFailure(srcOp, "unsupported bitcast");
    }
    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      arith::SelectOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto valueType = adaptor.getTrueValue().getType();
    if (valueType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI32Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (valueType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectI64Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (valueType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF32Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (valueType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectF64Op>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else if (llvm::isa<IREE::VM::RefType>(valueType)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SelectRefOp>(
          srcOp, valueType, adaptor.getCondition(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    } else {
      return rewriter.notifyMatchFailure(srcOp,
                                         "unsupported select element type");
    }
  }
};

class AssertOpConversion : public OpConversionPattern<cf::AssertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cf::AssertOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto status = rewriter.create<IREE::VM::ConstI32Op>(
        srcOp.getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(32),
            static_cast<int32_t>(IREE::Util::StatusCode::FailedPrecondition)));
    // TODO(benvanik): invert cond_fail instead.
    auto invertedCondition = rewriter.createOrFold<IREE::VM::XorI32Op>(
        srcOp.getLoc(), adaptor.getArg().getType(), adaptor.getArg(),
        rewriter.createOrFold<IREE::VM::ConstI32Op>(srcOp.getLoc(), 1));
    rewriter.replaceOpWithNewOp<IREE::VM::CondFailOp>(srcOp, invertedCondition,
                                                      status, adaptor.getMsg());
    return success();
  }
};

class BranchOpConversion : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cf::BranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(srcOp, srcOp.getDest(),
                                                    adaptor.getOperands());
    return success();
  }
};

class CondBranchOpConversion : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      cf::CondBranchOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Block *trueDest = srcOp.getTrueDest();
    rewriter.replaceOpWithNewOp<IREE::VM::CondBranchOp>(
        srcOp, adaptor.getCondition(), trueDest, adaptor.getTrueDestOperands(),
        srcOp.getFalseDest(), adaptor.getFalseDestOperands());
    return success();
  }
};

}  // namespace

void populateStandardToVMPatterns(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns
      .insert<AssertOpConversion, BranchOpConversion, CallOpConversion,
              CmpI32OpConversion, CmpI64OpConversion, CmpF32OpConversion,
              CondBranchOpConversion, ModuleOpConversion, FuncOpConversion,
              ExternalFuncOpConversion, ReturnOpConversion, SelectOpConversion>(
          typeConverter, context);

  // TODO(#2878): figure out how to pass the type converter in a supported way.
  // Right now if we pass the type converter as the first argument - triggering
  // the ConversionPattern stuff - it'll do weird things.
  patterns.insert<ConstantOpConversion>(context, typeConverter);

  patterns.insert<CastingOpConversion<UnrealizedConversionCastOp>,
                  IndexCastOpConversion<arith::IndexCastOp, arith::ExtSIOp>,
                  IndexCastOpConversion<arith::IndexCastUIOp, arith::ExtUIOp>,
                  ZeroExtendIOpConversion, SignExtendIOpConversion,
                  TruncateIOpConversion>(typeConverter, context);

  // Integer arithmetic ops.
  patterns
      .insert<BinaryArithmeticOpConversion<arith::AddIOp, IREE::VM::AddI32Op,
                                           IREE::VM::AddI64Op>,
              BinaryArithmeticOpConversion<arith::DivSIOp, IREE::VM::DivI32SOp,
                                           IREE::VM::DivI64SOp>,
              BinaryArithmeticOpConversion<arith::DivUIOp, IREE::VM::DivI32UOp,
                                           IREE::VM::DivI64UOp>,
              BinaryArithmeticOpConversion<arith::MulIOp, IREE::VM::MulI32Op,
                                           IREE::VM::MulI64Op>,
              BinaryArithmeticOpConversion<arith::RemSIOp, IREE::VM::RemI32SOp,
                                           IREE::VM::RemI64SOp>,
              BinaryArithmeticOpConversion<arith::RemUIOp, IREE::VM::RemI32UOp,
                                           IREE::VM::RemI64UOp>,
              BinaryArithmeticOpConversion<arith::MinSIOp, IREE::VM::MinI32SOp,
                                           IREE::VM::MinI64SOp>,
              BinaryArithmeticOpConversion<arith::MinUIOp, IREE::VM::MinI32UOp,
                                           IREE::VM::MinI64UOp>,
              BinaryArithmeticOpConversion<arith::MaxSIOp, IREE::VM::MaxI32SOp,
                                           IREE::VM::MaxI64SOp>,
              BinaryArithmeticOpConversion<arith::MaxUIOp, IREE::VM::MaxI32UOp,
                                           IREE::VM::MaxI64UOp>,
              BinaryArithmeticOpConversion<arith::SubIOp, IREE::VM::SubI32Op,
                                           IREE::VM::SubI64Op>,
              BinaryArithmeticOpConversion<arith::AndIOp, IREE::VM::AndI32Op,
                                           IREE::VM::AndI64Op>,
              BinaryArithmeticOpConversion<arith::OrIOp, IREE::VM::OrI32Op,
                                           IREE::VM::OrI64Op>,
              BinaryArithmeticOpConversion<arith::XOrIOp, IREE::VM::XorI32Op,
                                           IREE::VM::XorI64Op>>(typeConverter,
                                                                context);

  // Floating-point arithmetic ops.
  patterns
      .insert<UnaryArithmeticOpConversion<math::AbsFOp, IREE::VM::AbsF32Op,
                                          IREE::VM::AbsF64Op>,
              BinaryArithmeticOpConversion<arith::AddFOp, IREE::VM::AddF32Op,
                                           IREE::VM::AddF64Op>,
              UnaryArithmeticOpConversion<math::CeilOp, IREE::VM::CeilF32Op,
                                          IREE::VM::CeilF64Op>,
              UnaryArithmeticOpConversion<math::FloorOp, IREE::VM::FloorF32Op,
                                          IREE::VM::FloorF64Op>,
              BinaryArithmeticOpConversion<arith::DivFOp, IREE::VM::DivF32Op,
                                           IREE::VM::DivF64Op>,
              BinaryArithmeticOpConversion<arith::MulFOp, IREE::VM::MulF32Op,
                                           IREE::VM::MulF64Op>,
              UnaryArithmeticOpConversion<arith::NegFOp, IREE::VM::NegF32Op,
                                          IREE::VM::NegF64Op>,
              BinaryArithmeticOpConversion<arith::RemFOp, IREE::VM::RemF32Op,
                                           IREE::VM::RemF64Op>,
              BinaryArithmeticOpConversion<arith::SubFOp, IREE::VM::SubF32Op,
                                           IREE::VM::SubF64Op>,
              BinaryArithmeticOpConversion<arith::MinFOp, IREE::VM::MinF32Op,
                                           IREE::VM::MinF64Op>,
              BinaryArithmeticOpConversion<arith::MaxFOp, IREE::VM::MaxF32Op,
                                           IREE::VM::MaxF64Op>>(typeConverter,
                                                                context);

  // Floating-point conversion ops.
  patterns.insert<IntToFPOpConversion<arith::SIToFPOp, arith::ExtSIOp,
                                      IREE::VM::CastSI32F32Op>,
                  IntToFPOpConversion<arith::UIToFPOp, arith::ExtUIOp,
                                      IREE::VM::CastUI32F32Op>,
                  FPToSIOpConversion, FPToUIOpConversion, BitcastOpConversion>(
      typeConverter, context);

  // Shift ops.
  patterns
      .insert<ShiftArithmeticOpConversion<arith::ShLIOp, IREE::VM::ShlI32Op,
                                          IREE::VM::ShlI64Op>,
              ShiftArithmeticOpConversion<arith::ShRSIOp, IREE::VM::ShrI32SOp,
                                          IREE::VM::ShrI64SOp>,
              ShiftArithmeticOpConversion<arith::ShRUIOp, IREE::VM::ShrI32UOp,
                                          IREE::VM::ShrI64UOp>>(typeConverter,
                                                                context);
}

}  // namespace iree_compiler
}  // namespace mlir
