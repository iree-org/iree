// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace json = llvm::json;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace ABI {

// Generates default ABI metadata for entry points.
class EmitDefaultABIPass
    : public PassWrapper<EmitDefaultABIPass, OperationPass<ModuleOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {}

  StringRef getArgument() const override { return "iree-abi-emit-default-abi"; }

  StringRef getDescription() const override {
    return "Generates default reflection metadata for entry points.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<func::FuncOp> entryFuncOps;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (funcOp.isPublic() && !funcOp->hasAttr("iree.abi.stub")) {
        entryFuncOps.push_back(funcOp);
      }
    }

    auto hasIREEABI = [](func::FuncOp func) {
      return func->hasAttr("iree.abi");
    };

    // Respect any existing iree.abi attributes. Frontends may have already
    // populated these, so we don't have any work to do in that case.
    // However, we treat it as an error if some, but not all, entry points have
    // iree.abi attributes, since then the frontend won't be able to call the
    // functions (or will be relying on some implicit contract)
    if (llvm::any_of(entryFuncOps, hasIREEABI)) {
      if (llvm::all_of(entryFuncOps, hasIREEABI)) {
        return;
      } else {
        auto diag = moduleOp->emitError(
            "some, but not all, entry points have iree.abi attributes -- it "
            "will not be possible to consistently call all entry points");
        for (auto funcOp : entryFuncOps) {
          if (!hasIREEABI(funcOp)) {
            diag.attachNote(funcOp.getLoc())
                .append("this function is missing iree.abi attributes");
          }
        }
        return signalPassFailure();
      }
    }

    for (auto funcOp : entryFuncOps) {
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
              << "unable to generate reflection descriptor for result type "
              << t;
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

std::unique_ptr<OperationPass<ModuleOp>> createEmitDefaultABIPass() {
  return std::make_unique<EmitDefaultABIPass>();
}

static PassRegistration<EmitDefaultABIPass> pass;

}  // namespace ABI
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
