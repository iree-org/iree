// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler {

std::optional<llvm::json::Value> mapTypeToJsonTypeRecord(Type type) {
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    auto typeValue = mapTypeToJsonTypeRecord(shapedType.getElementType());
    llvm::json::Array record({
        llvm::json::Value("ndarray"),
        typeValue ? *typeValue : llvm::json::Value(nullptr),
        shapedType.hasRank() ? llvm::json::Value(shapedType.getRank())
                             : llvm::json::Value(nullptr),
    });
    if (shapedType.hasRank()) {
      for (auto dim : shapedType.getShape()) {
        record.push_back(dim == ShapedType::kDynamic
                             ? llvm::json::Value(nullptr)
                             : llvm::json::Value(dim));
      }
    }
    return llvm::json::Value(std::move(record));
  }

  // Primitives.
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    std::string name = (Twine("i") + Twine(integerType.getWidth())).str();
    return llvm::json::Value(std::move(name));
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    if (floatType == FloatType::getBF16(floatType.getContext())) {
      return llvm::json::Value("bf16");
    }
    std::string name = (Twine("f") + Twine(floatType.getWidth())).str();
    return llvm::json::Value(std::move(name));
  }

  return std::nullopt;
}

void appendDefaultABI(func::FuncOp funcOp) {
  llvm::json::Array refArgs;
  for (Type t : funcOp.getArgumentTypes()) {
    auto descriptor = mapTypeToJsonTypeRecord(t);
    if (!descriptor) {
      funcOp.emitWarning()
          << "unable to generate reflection descriptor for argument type " << t;
      return;
    }
    refArgs.push_back(*descriptor);
  }

  llvm::json::Array refReturns;
  for (Type t : funcOp.getCallableResults()) {
    auto descriptor = mapTypeToJsonTypeRecord(t);
    if (!descriptor) {
      funcOp.emitWarning()
          << "unable to generate reflection descriptor for result type " << t;
      return;
    }
    refReturns.push_back(*descriptor);
  }

  Builder builder(funcOp);
  llvm::json::Object refDict;
  refDict["v"] = llvm::json::Value(1);
  refDict["a"] = llvm::json::Value(std::move(refArgs));
  refDict["r"] = llvm::json::Value(std::move(refReturns));
  llvm::json::Value refDictValue(std::move(refDict));
  std::string refStr;
  llvm::raw_string_ostream refOut(refStr);
  refOut << refDictValue;
  refOut.flush();
  funcOp->setAttr("iree.abi", builder.getStringAttr(refStr));
}

struct EmitDefaultIREEABIPass
    : public EmitDefaultIREEABIBase<EmitDefaultIREEABIPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (SymbolTable::getSymbolVisibility(funcOp) ==
            SymbolTable::Visibility::Public ||
        !funcOp->hasAttr("iree.abi")) {
      appendDefaultABI(funcOp);
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createEmitDefaultIREEABIPass() {
  return std::make_unique<EmitDefaultIREEABIPass>();
}

}  // namespace mlir::iree_compiler
