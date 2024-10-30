// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/StandardToVM/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ModuleOpConversion : public OpConversionPattern<ModuleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ModuleOp srcOp, OpAdaptor adaptor,
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
    if (auto reflectionAttr =
            srcOp->getAttrOfType<DictionaryAttr>("iree.reflection")) {
      newModuleOp->setAttr("iree.reflection", reflectionAttr);
    }
    Block *firstCreatedBlock = &newModuleOp.getBodyRegion().front();
    rewriter.inlineRegionBefore(srcOp.getBodyRegion(), firstCreatedBlock);
    auto blockRange = llvm::make_range(Region::iterator(firstCreatedBlock),
                                       newModuleOp.getBodyRegion().end());
    for (Block &block : llvm::make_early_inc_range(blockRange)) {
      rewriter.eraseBlock(&block);
    }
    rewriter.eraseOp(srcOp);
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(&newModuleOp.getBodyRegion().front());
    rewriter.create<IREE::VM::ModuleTerminatorOp>(srcOp.getLoc());
    return success();
  }
};

// Converts a function signature with the given |signatureConversion| util.
static FailureOr<FunctionType>
convertFuncSignature(func::FuncOp srcOp, const TypeConverter &typeConverter,
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
  if (srcOp->hasAttr("noinline")) {
    dstOp->setAttr("inlining_policy",
                   IREE::Util::InlineNeverAttr::get(dstOp->getContext()));
  }
}

struct FuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp srcOp, OpAdaptor adaptor,
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
static void copyImportAttrs(func::FuncOp srcOp, IREE::VM::ImportOp dstOp) {
  constexpr const char *kRetainedAttributes[] = {
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

struct ExternalFuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp srcOp, OpAdaptor adaptor,
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

struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  ImportTable &importTable;
  CallOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                   ImportTable &importTable, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        importTable(importTable) {}
  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
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
        callOp.getCallee(), adaptor.getOperands(), resultTypes, importTable,
        rewriter);
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
                ImportTable &importTable,
                ConversionPatternRewriter &rewriter) const {
    // Lookup the target and detect if it is an import.
    auto import = importTable.find(calleeName);
    if (import.has_value()) {
      return convertImportCallOp(rootOp, loc, *import, operands, resultTypes,
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
  convertImportCallOp(Operation *rootOp, Location loc,
                      ImportTable::Import &import, ValueRange operands,
                      TypeRange resultTypes,
                      ConversionPatternRewriter &rewriter) const {
    if (import.fallback) {
      return convertOptionalImportCallOp(
          rootOp, loc, import.name, operands, resultTypes, import.signature,
          import.fallback.getLeafReference().getValue(), rewriter);
    } else {
      return convertMandatoryImportCallOp(rootOp, loc, import.name, operands,
                                          resultTypes, import.signature,
                                          rewriter);
    }
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
                                         resultTypes, importTable, rewriter);
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

struct ReturnOpConversion : public OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::func::ReturnOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::ReturnOp>(srcOp,
                                                    adaptor.getOperands());
    return success();
  }
};

template <typename StdOp>
struct CastingOpConversion : public OpConversionPattern<StdOp> {
  using OpConversionPattern<StdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(StdOp srcOp, typename StdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(srcOp, adaptor.getOperands());
    return success();
  }
};

struct AssertOpConversion : public OpConversionPattern<cf::AssertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cf::AssertOp srcOp, OpAdaptor adaptor,
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

struct BranchOpConversion : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cf::BranchOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(srcOp, srcOp.getDest(),
                                                    adaptor.getOperands());
    return success();
  }
};

struct CondBranchOpConversion : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cf::CondBranchOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *trueDest = srcOp.getTrueDest();
    rewriter.replaceOpWithNewOp<IREE::VM::CondBranchOp>(
        srcOp, adaptor.getCondition(), trueDest, adaptor.getTrueDestOperands(),
        srcOp.getFalseDest(), adaptor.getFalseDestOperands());
    return success();
  }
};

struct SwitchOpConversion : public OpConversionPattern<cf::SwitchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cf::SwitchOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Special handling for default only ops: just jump to default.
    if (srcOp.getCaseDestinations().empty()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BranchOp>(
          srcOp, srcOp.getDefaultDestination(), adaptor.getDefaultOperands());
      return success();
    }

    // NOTE: cf.switch can have sparse indices but we cannot; instead we fill
    // any gaps with branches to the default block. This is wasteful but keeps
    // the runtime super simple - if we have offset or really sparse tables
    // (default + case 10000 + case 400000) we can optimize those in the
    // compiler by using multiple branch tables, inverse lookups via a lookup
    // op, etc.
    //
    // To make this simple here we get all cases, sort them, and then walk in
    // order while filling gaps as need.
    SmallVector<std::pair<int, int64_t>> caseValues;
    for (auto [i, value] : llvm::enumerate(srcOp.getCaseValues().value())) {
      caseValues.push_back(std::make_pair(i, value.getSExtValue()));
    }
    llvm::stable_sort(caseValues,
                      [](std::pair<int, int64_t> a, std::pair<int, int64_t> b) {
                        return a.second < b.second;
                      });

    // Sanity check negative values, which are tricky.
    int64_t minValue = caseValues.front().second;
    if (minValue < 0) {
      return rewriter.notifyMatchFailure(
          srcOp, "negative case indices are not supported by the VM (today); "
                 "needs positive offsetting");
    }

    // If the first branch is offset from 0 then we can subtract that out to
    // avoid holes at the start of the table.
    Value index = adaptor.getFlag();
    if (minValue > 0) {
      index = rewriter.create<IREE::VM::SubI32Op>(
          srcOp.getLoc(), rewriter.getI32Type(), index,
          rewriter.create<IREE::VM::ConstI32Op>(
              srcOp.getLoc(), static_cast<int32_t>(minValue)));
      for (auto &[i, value] : caseValues) {
        value -= minValue;
      }
    }

    // Emit each dense case, filling interior holes as needed.
    SmallVector<ValueRange> adaptedCaseOperands = adaptor.getCaseOperands();
    SmallVector<Block *> caseDestinations;
    SmallVector<ValueRange> caseOperands;
    int64_t lastValue = 0;
    for (auto [i, value] : caseValues) {
      while (value != lastValue && value - lastValue != 1) {
        // Hole to fill.
        caseDestinations.push_back(srcOp.getDefaultDestination());
        caseOperands.push_back(adaptor.getDefaultOperands());
        ++lastValue;
      }
      caseDestinations.push_back(srcOp.getCaseDestinations()[i]);
      caseOperands.push_back(adaptedCaseOperands[i]);
      lastValue = value;
    }

    rewriter.replaceOpWithNewOp<IREE::VM::BranchTableOp>(
        srcOp, index, adaptor.getDefaultOperands(), caseOperands,
        srcOp.getDefaultDestination(), caseDestinations);
    return success();
  }
};

} // namespace

void populateStandardToVMPatterns(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  ImportTable &importTable,
                                  RewritePatternSet &patterns) {
  patterns
      .insert<AssertOpConversion, BranchOpConversion, CondBranchOpConversion,
              SwitchOpConversion, ModuleOpConversion, FuncOpConversion,
              ExternalFuncOpConversion, ReturnOpConversion>(typeConverter,
                                                            context);
  patterns.insert<CallOpConversion>(typeConverter, context, importTable);
  patterns.insert<CastingOpConversion<mlir::UnrealizedConversionCastOp>>(
      typeConverter, context);
}

} // namespace mlir::iree_compiler
