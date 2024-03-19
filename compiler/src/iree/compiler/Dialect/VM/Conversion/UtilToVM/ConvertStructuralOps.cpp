// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct InitializerOpConversion
    : public OpConversionPattern<IREE::Util::InitializerOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Util::InitializerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<IREE::VM::InitializerOp>(op.getLoc());
    rewriter.cloneRegionBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody().begin());

    // Tell the rewriter to convert the region signature.
    const TypeConverter &typeConverter = *getTypeConverter();
    TypeConverter::SignatureConversion signatureConversion(0);
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return rewriter.notifyMatchFailure(op, "failed to convert region types");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Converts a function signature with the given |signatureConversion| util.
static FailureOr<FunctionType>
convertFuncSignature(IREE::Util::FuncOp srcOp,
                     const TypeConverter &typeConverter,
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
static void copyFuncAttrs(IREE::Util::FuncOp srcOp, Operation *dstOp) {
  constexpr const char *kRetainedAttributes[] = {
      "iree.reflection",
      "sym_visibility",
      "inlining_policy",
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

class FuncOpConversion : public OpConversionPattern<IREE::Util::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::FuncOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Handled by import-specific conversion.
    if (srcOp.isExternal())
      return failure();

    // Convert function signature.
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());
    auto newFuncType = convertFuncSignature(srcOp, *getTypeConverter(),
                                            signatureConversion, rewriter);
    if (failed(newFuncType))
      return failure();

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncOp = rewriter.create<IREE::VM::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), *newFuncType);
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getFunctionBody(),
                                newFuncOp.end());

    // Tell the rewriter to convert the region signature.
    const TypeConverter &typeConverter = *getTypeConverter();
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

    rewriter.eraseOp(srcOp);
    return success();
  }
};

// Copies attributes from |srcOp| to |dstOp| that we preserve during conversion.
// We allow external functions to have some special vm-specific attributes that
// override behavior during conversion and don't want to propagate them.
static void copyImportAttrs(IREE::Util::FuncOp srcOp,
                            IREE::VM::ImportOp dstOp) {
  constexpr const char *kRetainedAttributes[] = {
      "nosideeffects",
      "vm.fallback",
      "vm.signature",
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

class ExternalFuncOpConversion
    : public OpConversionPattern<IREE::Util::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::FuncOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Handled by internal-specific conversion.
    if (!srcOp.isExternal())
      return failure();

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
      if (failed(convertedSignature))
        return failure();
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

    rewriter.eraseOp(srcOp);
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<IREE::Util::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::CallOp callOp, OpAdaptor adaptor,
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
  FailureOr<SmallVector<Value>>
  convertCallOp(Operation *rootOp, Location loc, StringRef calleeName,
                ValueRange operands, TypeRange resultTypes,
                ConversionPatternRewriter &rewriter) const {
    // (Slow) lookup of the target function, which may be an import that we need
    // to perform type conversion for.
    auto calleeOp = SymbolTable::lookupSymbolIn(rootOp, calleeName);
    if (auto funcOp = dyn_cast_or_null<IREE::Util::FuncOp>(calleeOp)) {
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
  FailureOr<SmallVector<Value>>
  convertImportCallOp(Operation *rootOp, Location loc, StringRef calleeName,
                      ValueRange operands, TypeRange resultTypes,
                      FunctionType importSignature, Operation *calleeOp,
                      ConversionPatternRewriter &rewriter) const {
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
    auto exitResults =
        llvm::map_to_vector(exitBlock->addArguments(resultTypes, resultLocs),
                            [](BlockArgument arg) -> Value { return arg; });

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
    if (failed(fallbackResults))
      return failure();
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

struct ReturnOpConversion : public OpConversionPattern<IREE::Util::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::Util::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

} // namespace

void populateUtilStructuralToVMPatterns(MLIRContext *context,
                                        ConversionTarget &conversionTarget,
                                        TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  conversionTarget.addIllegalOp<IREE::Util::InitializerOp, IREE::Util::FuncOp,
                                IREE::Util::CallOp, IREE::Util::ReturnOp>();
  patterns
      .insert<InitializerOpConversion, FuncOpConversion,
              ExternalFuncOpConversion, CallOpConversion, ReturnOpConversion>(
          typeConverter, context);
}

} // namespace mlir::iree_compiler
