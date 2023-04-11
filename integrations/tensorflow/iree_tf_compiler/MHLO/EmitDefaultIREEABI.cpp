// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/MHLO/Passes.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace json = llvm::json;

namespace mlir {
namespace iree_integrations {
namespace MHLO {

class EmitDefaultIREEABIPass
    : public PassWrapper<EmitDefaultIREEABIPass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-mhlo-emit-default-iree-abi";
  }

  StringRef getDescription() const override {
    return "Emits simple default ABI metadata";
  }

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
    SmallVector<Type> argTypes = flattenTypes(funcOp.getArgumentTypes());
    for (Type t : argTypes) {
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
    SmallVector<Type> resultTypes = flattenTypes(funcOp.getCallableResults());
    for (Type t : resultTypes) {
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

  SmallVector<Type> flattenTypes(ArrayRef<Type> types) {
    SmallVector<Type> flattened;
    std::function<void(ArrayRef<Type>)> helper =
        [&](ArrayRef<Type> types) -> void {
      for (Type t : types) {
        if (auto tt = t.dyn_cast<TupleType>()) {
          helper(tt.getTypes());
        } else {
          flattened.push_back(t);
        }
      }
    };
    helper(types);
    return flattened;
  }

  std::optional<json::Value> mapTypeToJsonTypeRecord(Type type) {
    if (auto shapedType = type.dyn_cast<ShapedType>()) {
      auto typeValue = mapTypeToJsonTypeRecord(shapedType.getElementType());
      json::Array record({
          json::Value("ndarray"),
          typeValue ? *typeValue : json::Value(nullptr),
          shapedType.hasRank() ? json::Value(shapedType.getRank())
                               : json::Value(nullptr),
      });
      if (shapedType.hasRank()) {
        for (auto dim : shapedType.getShape()) {
          record.push_back(dim == ShapedType::kDynamic ? json::Value(nullptr)
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

    return std::nullopt;
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createEmitDefaultIREEABIPass() {
  return std::make_unique<EmitDefaultIREEABIPass>();
}

static PassRegistration<EmitDefaultIREEABIPass> funcPass;

}  // namespace MHLO
}  // namespace iree_integrations
}  // namespace mlir
