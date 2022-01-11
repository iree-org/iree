// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace mlir {
namespace iree_compiler {

/// Return the function symbol associated with `name` or create it if it doesn't
/// exist.
static FlatSymbolRefAttr getOrCreateFunc(ModuleOp module, Location loc,
                                         StringRef name, Type result,
                                         ValueRange operands) {
  MLIRContext *context = module.getContext();
  auto func = module.lookupSymbol<FuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder
        .create<FuncOp>(loc, name,
                        FunctionType::get(context, operands.getTypes(), result))
        .setPrivate();
  }
  return SymbolRefAttr::get(context, name);
}

namespace {

constexpr static StringRef kNewSparseFuncName = "newSparseTensor";

/// This pass convert `newSparseTensor` builtin from tensor to memref world. It
/// inserts bufferization.to_tensor_op at the boundary.
struct SparsePreBufferizationPass
    : public SparsePreBufferizationBase<SparsePreBufferizationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getOperation();
    SmallVector<CallOp> callToReplace;
    moduleOp.walk([&callToReplace](CallOp callOp) {
      if (callOp.getCallee() == kNewSparseFuncName) {
        callToReplace.push_back(callOp);
      }
    });
    for (auto callOp : callToReplace) {
      OpBuilder b(callOp);
      Location loc = callOp.getLoc();
      StringRef name = kNewSparseFuncName;
      SmallVector<Value> args;
      for (Value operand : callOp->getOperands()) {
        auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
        if (!tensorType) {
          args.push_back(operand);
          continue;
        }
        Type newType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        Value newarg =
            b.create<bufferization::ToTensorOp>(loc, newType, operand);
        args.push_back(newarg);
      }
      Type ptrType = callOp.getResult(0).getType();
      Operation *newCall = b.create<CallOp>(
          loc, ptrType, getOrCreateFunc(moduleOp, loc, name, ptrType, args),
          args);
      callOp.replaceAllUsesWith(newCall);
      callOp.erase();
      auto module = newCall->getParentOfType<ModuleOp>();
      auto func = module.lookupSymbol<FuncOp>(name);
      func.setType(
          FunctionType::get(context, ValueRange(args).getTypes(), ptrType));
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createSparsePreBufferize() {
  return std::make_unique<SparsePreBufferizationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
