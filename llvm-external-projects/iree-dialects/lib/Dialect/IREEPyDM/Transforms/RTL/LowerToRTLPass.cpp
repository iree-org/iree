// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_pydm;

namespace pydm_d = mlir::iree_pydm;

namespace {

class RtlFunc {
 protected:
  static FunctionType makeRaisingSignature(Builder b, ArrayRef<Type> inputs,
                                           Type output) {
    return b.getType<FunctionType>(
        inputs, TypeRange{b.getType<pydm_d::ExceptionResultType>(), output});
  }
};

template <typename RtlFuncTy>
Operation *importRtlFunc(SymbolTable &symbolTable) {
  OpBuilder builder(symbolTable.getOp()->getContext());
  auto name = builder.getStringAttr(RtlFuncTy::getRtlName());
  auto *existing = symbolTable.lookup(name);
  if (existing) return existing;

  // Does not exist - create detached and insert.
  FunctionType signature = RtlFuncTy::getRtlSignature(builder);
  OperationState state(symbolTable.getOp()->getLoc(),
                       pydm_d::FuncOp::getOperationName());
  pydm_d::FuncOp::build(builder, state, name, signature);
  auto funcOp = Operation::create(state);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  symbolTable.insert(funcOp);
  return funcOp;
}

struct ObjectAsBoolFunc : public RtlFunc {
  static StringRef getRtlName() { return "pydmrtl$object_as_bool"; }
  static FunctionType getRtlSignature(Builder b) {
    return makeRaisingSignature(b, {b.getType<pydm_d::ObjectType>(nullptr)},
                                b.getType<pydm_d::BoolType>());
  }
};

struct DynamicBinaryPromoteFunc : public RtlFunc {
  static StringRef getRtlName() { return "pydmrtl$dynamic_binary_promote"; }
  static FunctionType getRtlSignature(Builder b) {
    return makeRaisingSignature(b,
                                {b.getType<pydm_d::ObjectType>(nullptr),
                                 b.getType<pydm_d::ObjectType>(nullptr)},
                                b.getType<pydm_d::TupleType>());
  }
};

template <typename RtlFuncTy, typename OpTy>
class EmitImportCallBase : public OpRewritePattern<OpTy> {
 public:
  EmitImportCallBase(SymbolTable &symbolTable, PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>::OpRewritePattern(
            symbolTable.getOp()->getContext(), benefit),
        symbolTable(symbolTable) {}

 protected:
  Value emitImportCall(Location loc, ValueRange inputs,
                       PatternRewriter &rewriter) const {
    auto rtlName = RtlFuncTy::getRtlName();
    importRtlFunc<RtlFuncTy>(symbolTable);
    FunctionType signature = RtlFuncTy::getRtlSignature(rewriter);
    auto symbolRef = rewriter.getType<FlatSymbolRefAttr>(rtlName);
    auto callOp = rewriter.create<pydm_d::CallOp>(loc, signature.getResults(),
                                                  symbolRef, inputs);
    rewriter.create<pydm_d::RaiseOnFailureOp>(loc, callOp.exc_result());
    return callOp.result();
  }

  void replaceOpWithCall(Operation *op, ValueRange inputs,
                         PatternRewriter &rewriter) const {
    Value callResult = emitImportCall(op->getLoc(), inputs, rewriter);
    assert(op->getNumResults() != 0 && "expected op with results");
    if (op->getNumResults() == 1) {
      // No unpack.
      rewriter.replaceOp(op, {callResult});
    } else {
      // Unpack 1 -> N.
      SmallVector<Type> unpackTypes = {
          rewriter.getType<pydm_d::ExceptionResultType>()};
      unpackTypes.append(op->getResultTypes().begin(),
                         op->getResultTypes().end());
      auto unpackOp = rewriter.create<pydm_d::DynamicUnpackOp>(
          op->getLoc(), unpackTypes, callResult);
      rewriter.create<pydm_d::RaiseOnFailureOp>(op->getLoc(),
                                                unpackOp.exc_result());
      rewriter.replaceOp(op, unpackOp.slots());
    }
  }

 private:
  SymbolTable &symbolTable;
};

struct ObjectAsBoolPattern
    : public EmitImportCallBase<ObjectAsBoolFunc, pydm_d::AsBoolOp> {
  using EmitImportCallBase::EmitImportCallBase;

  LogicalResult matchAndRewrite(pydm_d::AsBoolOp srcOp,
                                PatternRewriter &rewriter) const override {
    auto valueType = srcOp.value().getType().dyn_cast<pydm_d::ObjectType>();
    if (!valueType)
      return rewriter.notifyMatchFailure(srcOp, "not an !object<>");
    replaceOpWithCall(srcOp, {srcOp.value()}, rewriter);
    return success();
  }
};

struct DynamicBinaryPromotePattern
    : public EmitImportCallBase<DynamicBinaryPromoteFunc,
                                pydm_d::DynamicBinaryPromoteOp> {
  using EmitImportCallBase::EmitImportCallBase;

  LogicalResult matchAndRewrite(pydm_d::DynamicBinaryPromoteOp srcOp,
                                PatternRewriter &rewriter) const override {
    replaceOpWithCall(srcOp, {srcOp.left(), srcOp.right()}, rewriter);
    return success();
  }
};

struct LowerIREEPyDMToRTLPass
    : public LowerIREEPyDMToRTLBase<LowerIREEPyDMToRTLPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<mlir::iree::IREEDialect, BuiltinDialect, StandardOpsDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    RewritePatternSet patterns(context);
    patterns.insert<DynamicBinaryPromotePattern, ObjectAsBoolPattern>(
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

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::iree_pydm::createLowerIREEPyDMToRTLPass() {
  return std::make_unique<LowerIREEPyDMToRTLPass>();
}
