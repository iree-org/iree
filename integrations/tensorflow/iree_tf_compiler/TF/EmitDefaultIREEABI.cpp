// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace json = llvm::json;

namespace mlir {
namespace iree_integrations {
namespace TF {

class EmitDefaultIREEABIPass
    : public PassWrapper<EmitDefaultIREEABIPass, OperationPass<FuncOp>> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (SymbolTable::getSymbolVisibility(funcOp) !=
        SymbolTable::Visibility::Public) {
      return;
    }
    if (funcOp->hasAttr("iree.abi")) {
      return;
    }

    json::Array refArgs;
    for (Type t : funcOp.getArgumentTypes()) {
      auto descriptor = mapTypeToJsonTypeRecord(t);
      if (!descriptor) {
        funcOp.emitWarning()
            << "unable to generate reflection descriptor for argument type "
            << t;
        return;
      }
      refArgs.push_back(*descriptor);
    }

    json::Array refReturns;
    for (Type t : funcOp.getCallableResults()) {
      auto descriptor = mapTypeToJsonTypeRecord(t);
      if (!descriptor) {
        funcOp.emitWarning()
            << "unable to generate reflection descriptor for result type " << t;
        return;
      }
      refReturns.push_back(*descriptor);
    }

    Builder builder(&getContext());
    json::Object refDict;
    refDict["v"] = json::Value(1);
    refDict["a"] = json::Value(std::move(refArgs));
    refDict["r"] = json::Value(std::move(refReturns));
    json::Value refDictValue(std::move(refDict));
    std::string refStr;
    llvm::raw_string_ostream refOut(refStr);
    refOut << refDictValue;
    refOut.flush();
    funcOp->setAttr("iree.abi", builder.getStringAttr(refStr));
  }

  llvm::Optional<json::Value> mapTypeToJsonTypeRecord(Type type) {
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      json::Array record({
          json::Value("ndarray"),
          mapTypeToJsonTypeRecord(shapedType.getElementType()),
          shapedType.hasRank() ? json::Value(shapedType.getRank())
                               : json::Value(nullptr),
      });
      if (shapedType.hasRank()) {
        for (auto dim : shapedType.getShape()) {
          record.push_back(dim == ShapedType::kDynamicSize
                               ? json::Value(nullptr)
                               : json::Value(dim));
        }
      }
      return json::Value(std::move(record));
    }

    // Primitives.
    if (auto integerType = type.dyn_cast<IntegerType>()) {
      std::string name = (Twine("i") + Twine(integerType.getWidth())).str();
      return json::Value(std::move(name));
    }
    if (auto floatType = type.dyn_cast<FloatType>()) {
      if (floatType == FloatType::getBF16(floatType.getContext())) {
        // Why Google?
        return json::Value("bf16");
      }
      std::string name = (Twine("f") + Twine(floatType.getWidth())).str();
      return json::Value(std::move(name));
    }

    return llvm::None;
  }
};

std::unique_ptr<OperationPass<FuncOp>> createEmitDefaultIREEABIPass() {
  return std::make_unique<EmitDefaultIREEABIPass>();
}

static PassRegistration<EmitDefaultIREEABIPass> funcPass(
    "iree-tf-emit-default-iree-abi", "Emits simple default ABI metadata");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
