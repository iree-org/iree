// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ImportMLProgramPass : public ImportMLProgramBase<ImportMLProgramPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, func::FuncDialect>();
  }
  void runOnOperation() override;
};

class IREETypeConverter : public TypeConverter {
public:
  IREETypeConverter();
};

// Generic 1:1 conversion pattern which effectively just renames an op.
// It does not support regions or ops with successors.
class OneToOneConversionPattern : public ConversionPattern {
public:
  OneToOneConversionPattern(TypeConverter &converter, StringRef srcName,
                            StringRef targetName, MLIRContext *context,
                            PatternBenefit benefit)
      : ConversionPattern(converter, srcName, benefit, context),
        targetName(targetName) {}
  LogicalResult
  matchAndRewrite(Operation *srcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(srcOp->getResultTypes(),
                                           resultTypes))) {
      return srcOp->emitError()
             << "could not convert result types to IREE internal types";
    }

    OperationState state(srcOp->getLoc(), targetName, operands, resultTypes,
                         srcOp->getAttrs());
    Operation *targetOp = rewriter.create(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }

private:
  StringRef targetName;
};

// Components to construct globals.
struct GlobalComponents {
  GlobalComponents(StringRef name, Type newType)
      : name(name), newType(newType) {}
  StringRef name;
  Type newType;
};

class MLProgramGlobalOpPattern
    : public OpConversionPattern<ml_program::GlobalOp> {
public:
  MLProgramGlobalOpPattern(TypeConverter &typeConverter, MLIRContext *context,
                           PatternBenefit benefit,
                           SmallVector<GlobalComponents> &externGlobals)
      : OpConversionPattern<ml_program::GlobalOp>(typeConverter, context,
                                                  benefit),
        externGlobals(externGlobals) {}

  LogicalResult
  matchAndRewrite(ml_program::GlobalOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newType = typeConverter->convertType(srcOp.getType());
    if (!newType)
      return failure();

    std::map<StringRef, ml_program::ExternAttr> externs;

    auto srcOpAttr = srcOp.getValue();
    bool isExtern = srcOpAttr && isa<ml_program::ExternAttr>(*srcOpAttr);
    auto srcOpTypedAttr =
        (srcOpAttr && !isExtern)
            ? std::optional<TypedAttr>(llvm::cast<TypedAttr>(srcOpAttr.value()))
            : std::nullopt;
    const SymbolTable::Visibility visibility = srcOp.getVisibility();
    // Create util Global which is mutable if the ML program global was or if
    // extern to enable initialization method.
    bool isMutable = srcOp.getIsMutable() || isExtern;
    auto globalOp = rewriter.replaceOpWithNewOp<IREE::Util::GlobalOp>(
        srcOp, srcOp.getName(), isMutable | isExtern, newType, srcOpTypedAttr);
    globalOp.setVisibility(SymbolTable::Visibility::Private);

    if (isExtern)
      externGlobals.emplace_back(srcOp.getName(), newType);

    // No more work needed if not public global.
    if (visibility != SymbolTable::Visibility::Public)
      return success();

    ModuleOp module = srcOp->getParentOfType<ModuleOp>();

    // Generate accessor methods for global variables. For a given global
    // variable generate using the format provided on the Module.  The format
    // dict requires single "{0}" which will be replaced by the global variable
    // name. E.g., if the format dict is `{get: "get{0}", set: "set{0}"}` then
    // for a variable X
    //   * `getX` will be generated to read the variable;
    //   * `setX` will be generated to set the variable (if X is mutable);
    //
    // If unset the default is `global${0}$get` and `global${0}$set`.

    std::string getterName, setterName;

    auto verifyFormat = [](const std::string &format) {
      StringRef s = format;
      // Verify only single replacement of 0th index.
      s = s.drop_until([](char c) { return c == '{'; });
      if (s.empty() || !s.consume_front("{"))
        return failure();
      if (!s.consume_front("0"))
        return failure();
      if (!s.consume_front("}"))
        return failure();
      s = s.drop_until([](char c) { return c == '{'; });
      return success(s.empty());
    };

    auto v = module->getAttrOfType<DictionaryAttr>(
        "ml_program.public_global_accessors");
    // TODO(jpienaar): The attribute should be verified before here.
    StringAttr get =
        v ? llvm::dyn_cast_if_present<StringAttr>(v.get("get")) : nullptr;
    {
      const std::string getFormat = get ? get.str() : "global${0}$get";
      if (failed(verifyFormat(getFormat)))
        return failure();
      getterName = llvm::formatv(getFormat.c_str(), globalOp.getSymName());
    }
    auto set =
        v ? llvm::dyn_cast_if_present<StringAttr>(v.get("set")) : nullptr;
    {
      const std::string setFormat = set ? set.str() : "global${0}$set";
      if (failed(verifyFormat(setFormat)))
        return failure();
      setterName = llvm::formatv(setFormat.c_str(), globalOp.getSymName());
    }

    // Add public getter function.
    if (!getterName.empty()) {
      FunctionType funcType =
          rewriter.getFunctionType(/*input=*/TypeRange{}, /*outputs=*/newType);
      ImplicitLocOpBuilder b(globalOp.getLoc(), rewriter);
      auto funcOp = b.create<func::FuncOp>(getterName, funcType);
      funcOp.setPublic();
      b.setInsertionPointToStart(funcOp.addEntryBlock());
      auto val = b.create<IREE::Util::GlobalLoadOp>(
          newType, SymbolRefAttr::get(globalOp.getSymNameAttr()));
      b.create<func::ReturnOp>(val.getResult());
    }

    if (!setterName.empty() && isMutable) {
      // Add public setter function.
      FunctionType funcType =
          rewriter.getFunctionType(/*input=*/newType, /*outputs=*/TypeRange{});
      ImplicitLocOpBuilder b(globalOp.getLoc(), rewriter);
      auto funcOp = b.create<func::FuncOp>(setterName, funcType);
      funcOp.setPublic();
      b.setInsertionPointToStart(funcOp.addEntryBlock());
      b.create<IREE::Util::GlobalStoreOp>(funcOp.getArgument(0),
                                          globalOp.getSymNameAttr());
      b.create<func::ReturnOp>();
    }

    return success();
  }

  SmallVector<GlobalComponents> &externGlobals;
};

LogicalResult
createExternInitFunction(ModuleOp module,
                         SmallVector<GlobalComponents> &externGlobals) {
  std::sort(externGlobals.begin(), externGlobals.end(),
            [](const GlobalComponents &lhs, const GlobalComponents &rhs) {
              return lhs.name < rhs.name;
            });
  auto *context = module.getContext();
  ImplicitLocOpBuilder b(module.getLoc(), context);
  b.setInsertionPointToEnd(&module.getBodyRegion().back());
  FunctionType funcType = b.getFunctionType(
      /*input=*/TypeRange{IREE::Util::ListType::get(
          IREE::Util::VariantType::get(context))},
      /*outputs=*/{});
  auto funcOp = b.create<func::FuncOp>("ireeMlProgramGlobalsInit", funcType);
  funcOp.setPublic();
  b.setInsertionPointToStart(funcOp.addEntryBlock());

  for (auto it : llvm::enumerate(externGlobals)) {
    auto val = b.create<IREE::Util::ListGetOp>(
        it.value().newType, funcOp.getArgument(0),
        b.create<arith::ConstantIndexOp>(it.index()));
    b.create<IREE::Util::GlobalStoreOp>(val, it.value().name);
  }

  b.create<func::ReturnOp>();

  return success();
}

} // namespace

IREETypeConverter::IREETypeConverter() {
  addConversion([](Type t) { return t; });
}

void ImportMLProgramPass::runOnOperation() {
  auto &context = getContext();
  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  target.addLegalDialect<IREE::Util::UtilDialect>();
  target.addIllegalDialect<ml_program::MLProgramDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

  IREETypeConverter typeConverter;
  SmallVector<GlobalComponents> externGlobals;
  patterns.insert<MLProgramGlobalOpPattern>(typeConverter, &getContext(), 0,
                                            externGlobals);

  PatternBenefit specific_benefit = 100;
#define ONE_TO_ONE(SrcOpTy, TargetOpTy)                                        \
  patterns.insert<OneToOneConversionPattern>(                                  \
      typeConverter, SrcOpTy::getOperationName(),                              \
      TargetOpTy::getOperationName(), &context, specific_benefit)

  ONE_TO_ONE(ml_program::GlobalLoadOp, IREE::Util::GlobalLoadOp);
  ONE_TO_ONE(ml_program::GlobalLoadConstOp, IREE::Util::GlobalLoadOp);
  ONE_TO_ONE(ml_program::GlobalStoreOp, IREE::Util::GlobalStoreOp);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  if (!externGlobals.empty() &&
      failed(createExternInitFunction(getOperation(), externGlobals)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createImportMLProgramPass() {
  return std::make_unique<ImportMLProgramPass>();
}

} // namespace mlir::iree_compiler
