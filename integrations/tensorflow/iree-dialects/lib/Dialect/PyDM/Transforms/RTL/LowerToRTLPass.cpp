// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

namespace {

class RtlFunc {
protected:
  FunctionType makeRaisingSignature(Builder b, ArrayRef<Type> inputs,
                                    Type output) {
    return b.getType<FunctionType>(
        inputs, TypeRange{b.getType<PYDM::ExceptionResultType>(), output});
  }
};

template <typename RtlFuncTy>
Operation *importRtlFunc(SymbolTable &symbolTable, RtlFuncTy rtlFunc) {
  OpBuilder builder(symbolTable.getOp()->getContext());
  auto name = builder.getStringAttr(rtlFunc.getRtlName());
  auto *existing = symbolTable.lookup(name);
  if (existing)
    return existing;

  // Does not exist - create detached and insert.
  FunctionType signature = rtlFunc.getRtlSignature(builder);
  OperationState state(symbolTable.getOp()->getLoc(),
                       PYDM::FuncOp::getOperationName());
  PYDM::FuncOp::build(builder, state, name, signature);
  auto funcOp = Operation::create(state);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  symbolTable.insert(funcOp);
  return funcOp;
}

struct ObjectAsBoolFunc : public RtlFunc {
  StringRef getRtlName() { return "pydmrtl$object_as_bool"; }
  FunctionType getRtlSignature(Builder b) {
    return makeRaisingSignature(b, {b.getType<PYDM::ObjectType>(nullptr)},
                                b.getType<PYDM::BoolType>());
  }
};

struct DynamicBinaryPromoteFunc : public RtlFunc {
  StringRef getRtlName() { return "pydmrtl$dynamic_binary_promote"; }
  FunctionType getRtlSignature(Builder b) {
    return makeRaisingSignature(b,
                                {b.getType<PYDM::ObjectType>(nullptr),
                                 b.getType<PYDM::ObjectType>(nullptr)},
                                b.getType<PYDM::TupleType>());
  }
};

/// pydmrtl$apply_binary_${dunderName} RTL func.
class ApplyBinaryFunc : public RtlFunc {
public:
  ApplyBinaryFunc(StringRef dunderName) : rtlName("pydmrtl$apply_binary_") {
    rtlName.append(dunderName.begin(), dunderName.end());
  }
  StringRef getRtlName() { return rtlName; }
  FunctionType getRtlSignature(Builder b) {
    Type objectType = b.getType<PYDM::ObjectType>(nullptr);
    return makeRaisingSignature(b, {objectType, objectType}, objectType);
  }

private:
  std::string rtlName;
};

/// pydmrtl$apply_compare_${dunderName} RTL func.
class ApplyCompareFunc : public RtlFunc {
public:
  ApplyCompareFunc(StringRef dunderName) : rtlName("pydmrtl$apply_compare_") {
    rtlName.append(dunderName.begin(), dunderName.end());
  }
  StringRef getRtlName() { return rtlName; }
  FunctionType getRtlSignature(Builder b) {
    Type objectType = b.getType<PYDM::ObjectType>(nullptr);
    Type boolType = b.getType<PYDM::BoolType>();
    return makeRaisingSignature(b, {objectType, objectType}, boolType);
  }

private:
  std::string rtlName;
};

template <typename RtlFuncTy, typename OpTy>
class EmitImportCallBase : public OpRewritePattern<OpTy> {
public:
  EmitImportCallBase(SymbolTable &symbolTable, PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>::OpRewritePattern(
            symbolTable.getOp()->getContext(), benefit),
        symbolTable(symbolTable) {}

protected:
  Value emitImportCall(Location loc, ValueRange inputs, RtlFuncTy rtlFunc,
                       PatternRewriter &rewriter) const {
    auto rtlName = rtlFunc.getRtlName();
    importRtlFunc<RtlFuncTy>(symbolTable, rtlFunc);
    FunctionType signature = rtlFunc.getRtlSignature(rewriter);
    auto symbolRef = rewriter.getType<FlatSymbolRefAttr>(rtlName);
    // Perform simple conversions on inputs.
    SmallVector<Value> convertedInputs;
    for (auto it : zip(inputs, signature.getInputs())) {
      Value input = std::get<0>(it);
      Type expectedType = std::get<1>(it);
      // Detect boxing.
      if (expectedType.isa<PYDM::ObjectType>() &&
          !input.getType().isa<ObjectType>()) {
        input = rewriter.create<PYDM::BoxOp>(loc, expectedType, input);
      }
      convertedInputs.push_back(input);
    }

    auto callOp = rewriter.create<PYDM::CallOp>(loc, signature.getResults(),
                                                symbolRef, convertedInputs);
    rewriter.create<PYDM::RaiseOnFailureOp>(loc, callOp.exc_result());
    return callOp.result();
  }

  void replaceOpWithCall(Operation *op, ValueRange inputs, RtlFuncTy rtlFunc,
                         PatternRewriter &rewriter) const {
    Value callResult = emitImportCall(op->getLoc(), inputs, rtlFunc, rewriter);
    assert(op->getNumResults() != 0 && "expected op with results");
    if (op->getNumResults() == 1) {
      // No unpack.
      rewriter.replaceOp(op, {callResult});
    } else {
      // Unpack 1 -> N.
      SmallVector<Type> unpackTypes = {
          rewriter.getType<PYDM::ExceptionResultType>()};
      unpackTypes.append(op->getResultTypes().begin(),
                         op->getResultTypes().end());
      auto unpackOp = rewriter.create<PYDM::DynamicUnpackOp>(
          op->getLoc(), unpackTypes, callResult);
      rewriter.create<PYDM::RaiseOnFailureOp>(op->getLoc(),
                                              unpackOp.exc_result());
      rewriter.replaceOp(op, unpackOp.slots());
    }
  }

private:
  SymbolTable &symbolTable;
};

struct ApplyBinaryPattern
    : public EmitImportCallBase<ApplyBinaryFunc, PYDM::ApplyBinaryOp> {
  using EmitImportCallBase::EmitImportCallBase;

  LogicalResult matchAndRewrite(PYDM::ApplyBinaryOp srcOp,
                                PatternRewriter &rewriter) const override {
    // Only match object-object binary apply.
    if (!srcOp.left().getType().isa<PYDM::ObjectType>() ||
        !srcOp.right().getType().isa<PYDM::ObjectType>())
      return rewriter.notifyMatchFailure(srcOp, "not (object, object) variant");

    ApplyBinaryFunc f(srcOp.dunder_name());
    replaceOpWithCall(srcOp, {srcOp.left(), srcOp.right()}, std::move(f),
                      rewriter);
    return success();
  }
};

struct ApplyComparePattern
    : public EmitImportCallBase<ApplyCompareFunc, PYDM::ApplyCompareOp> {
  using EmitImportCallBase::EmitImportCallBase;

  LogicalResult matchAndRewrite(PYDM::ApplyCompareOp srcOp,
                                PatternRewriter &rewriter) const override {
    // Only match object-object binary apply.
    if (!srcOp.left().getType().isa<PYDM::ObjectType>() ||
        !srcOp.right().getType().isa<PYDM::ObjectType>())
      return rewriter.notifyMatchFailure(srcOp, "not (object, object) variant");

    ApplyCompareFunc f(srcOp.dunder_name());
    replaceOpWithCall(srcOp, {srcOp.left(), srcOp.right()}, std::move(f),
                      rewriter);
    return success();
  }
};

struct DynamicBinaryPromotePattern
    : public EmitImportCallBase<DynamicBinaryPromoteFunc,
                                PYDM::DynamicBinaryPromoteOp> {
  using EmitImportCallBase::EmitImportCallBase;

  LogicalResult matchAndRewrite(PYDM::DynamicBinaryPromoteOp srcOp,
                                PatternRewriter &rewriter) const override {
    replaceOpWithCall(srcOp, {srcOp.left(), srcOp.right()}, {}, rewriter);
    return success();
  }
};

struct ObjectAsBoolPattern
    : public EmitImportCallBase<ObjectAsBoolFunc, PYDM::AsBoolOp> {
  using EmitImportCallBase::EmitImportCallBase;

  LogicalResult matchAndRewrite(PYDM::AsBoolOp srcOp,
                                PatternRewriter &rewriter) const override {
    auto valueType = srcOp.value().getType().dyn_cast<PYDM::ObjectType>();
    if (!valueType)
      return rewriter.notifyMatchFailure(srcOp, "not an !object<>");
    replaceOpWithCall(srcOp, {srcOp.value()}, {}, rewriter);
    return success();
  }
};

struct LowerIREEPyDMToRTLPass
    : public LowerIREEPyDMToRTLBase<LowerIREEPyDMToRTLPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::iree_compiler::IREE::Input::IREEInputDialect,
                    BuiltinDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    RewritePatternSet patterns(context);
    patterns.insert<ApplyBinaryPattern, ApplyComparePattern,
                    DynamicBinaryPromotePattern, ObjectAsBoolPattern>(
        symbolTable);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns),
                                            config))) {
      emitError(getOperation().getLoc())
          << "did not converge while lowering to rtl";
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> PYDM::createLowerIREEPyDMToRTLPass() {
  return std::make_unique<LowerIREEPyDMToRTLPass>();
}
