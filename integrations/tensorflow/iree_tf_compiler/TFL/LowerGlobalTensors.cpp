// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/Input/InputOps.h"
#include "iree_tf_compiler/TFL/PassDetail.h"
#include "iree_tf_compiler/TFL/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {
namespace {

class LowerGlobalTensorsPass
    : public LowerGlobalTensorsBase<LowerGlobalTensorsPass> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect, tosa::TosaDialect,
                    iree_compiler::IREE::Input::IREEInputDialect>();
  }

  // Converts TFLite state operations to the IREE equivalent.
  void runOnOperation() override {
    auto* context = &getContext();
    auto moduleOp = getOperation();
    mlir::OpBuilder builder(moduleOp.body());

    DenseMap<StringRef, FuncOp> symNameToFunction;
    for (auto func : moduleOp.getOps<FuncOp>()) {
      symNameToFunction[func.sym_name()] = func;
    }

    DenseMap<StringRef, DenseElementsAttr> sharedNameToConstant;
    DenseMap<StringRef, LocationAttr> sharedNameToLoc;

    llvm::SmallVector<mlir::TFL::VarHandleOp, 6> handleOps;
    llvm::SmallVector<mlir::TFL::AssignVariableOp, 6> assignOps;
    llvm::SmallVector<mlir::TFL::ReadVariableOp, 6> readOps;
    for (auto it : symNameToFunction) {
      auto func = std::get<1>(it);
      // Look through the initialization functions and find the assigned values
      // for each handle, save out the constant value.
      for (auto init : func.getOps<mlir::TFL::CallOnceOp>()) {
        auto findInitFunc =
            symNameToFunction.find(init.session_init_function());
        if (findInitFunc == symNameToFunction.end()) {
          init.emitError("Unable to find initialization function: " +
                         init.session_init_function());
          continue;
        }
        FuncOp initFunc = std::get<1>(*findInitFunc);
        for (auto assign : initFunc.getOps<mlir::TFL::AssignVariableOp>()) {
          auto handle = dyn_cast<mlir::TFL::VarHandleOp>(
              assign.resource_id().getDefiningOp());
          if (!handle) continue;

          DenseElementsAttr constant;
          if (!matchPattern(assign.value(), m_Constant(&constant))) {
            // Quantized types we can not use the m_Constant matcher.
            if (auto constOp = dyn_cast<mlir::TFL::QConstOp>(
                    assign.value().getDefiningOp())) {
              constant = constOp.value().cast<DenseElementsAttr>();
            }
          }
          if (!constant) continue;

          auto name = handle.shared_name();
          sharedNameToConstant[name] = constant;
          sharedNameToLoc[name] = handle.getLoc();
        }
      }

      // We also want to grab the list of operations to replace.
      for (auto& op : func.getOps()) {
        if (auto handle = dyn_cast<mlir::TFL::VarHandleOp>(op))
          handleOps.push_back(handle);
        if (auto assign = dyn_cast<mlir::TFL::AssignVariableOp>(op))
          assignOps.push_back(assign);
        if (auto read = dyn_cast<mlir::TFL::ReadVariableOp>(op))
          readOps.push_back(read);
      }
    }

    // TF::CallOnceOps are no longer needed as we have already extracted their
    // state.
    SmallVector<mlir::TFL::CallOnceOp> callOnceOps;
    for (auto func : moduleOp.getOps<FuncOp>()) {
      for (auto init : func.getOps<mlir::TFL::CallOnceOp>()) {
        callOnceOps.push_back(init);
      }
    }
    for (auto op : callOnceOps) op.erase();

    // Create the Util::GlobalOps to store our new global variables.
    DenseMap<StringRef, std::string> sharedNameToFlowName;
    for (auto it : sharedNameToConstant) {
      auto name = std::get<0>(it);
      auto attribute = std::get<1>(it);
      auto locIt = sharedNameToLoc.find(name);
      LocationAttr loc = mlir::UnknownLoc();
      if (locIt != sharedNameToLoc.end()) {
        loc = std::get<1>(*locIt);
      }

      std::string flowSymName = "__iree_flow_" + name.str();

      // TODO(suderman): Determine the global type based on all store
      // operations.
      auto global = builder.create<iree_compiler::IREE::Input::GlobalOp>(
          loc, flowSymName, /*is_mutable=*/true, attribute.getType(),
          attribute);
      global.setPrivate();
      sharedNameToFlowName[name] = std::move(flowSymName);
    }

    // Replace handles with global addresses.
    for (auto handle : handleOps) {
      auto name = handle.shared_name();
      auto flowName = sharedNameToFlowName[name];
      auto constIt = sharedNameToConstant.find(name);
      if (constIt == sharedNameToConstant.end()) continue;

      auto attribute = std::get<1>(*constIt);

      builder.setInsertionPoint(handle);
      auto address =
          builder.create<iree_compiler::IREE::Input::GlobalAddressOp>(
              handle.getLoc(),
              iree_compiler::IREE::Input::PtrType::get(context,
                                                       attribute.getType()),
              SymbolRefAttr::get(builder.getContext(), flowName));
      handle.getResult().replaceAllUsesWith(address.getResult());
      handle.erase();
    }

    // Replace the assign ops with a global store operation.
    for (auto assign : assignOps) {
      auto address = dyn_cast<iree_compiler::IREE::Input::GlobalAddressOp>(
          assign.resource_id().getDefiningOp());
      if (!address) continue;

      builder.setInsertionPoint(assign);
      Value value = assign.value();
      Type storageType = address.getType()
                             .cast<iree_compiler::IREE::Input::PtrType>()
                             .getTargetType();
      if (storageType != value.getType()) {
        value = builder
                    .create<UnrealizedConversionCastOp>(assign.getLoc(),
                                                        storageType, value)
                    .getResult(0);
      }

      builder.create<iree_compiler::IREE::Input::GlobalStoreIndirectOp>(
          assign.getLoc(), value, assign.resource_id());
      assign.erase();
    }

    // Replace the read ops with a global load operation.
    for (auto read : readOps) {
      auto address = dyn_cast<iree_compiler::IREE::Input::GlobalAddressOp>(
          read.resource_id().getDefiningOp());
      if (!address) continue;

      auto ptrType =
          address.getType().dyn_cast<iree_compiler::IREE::Input::PtrType>();
      if (!ptrType) continue;

      auto type = ptrType.getTargetType();

      builder.setInsertionPoint(read);
      Value load =
          builder.create<iree_compiler::IREE::Input::GlobalLoadIndirectOp>(
              read.getLoc(), type, read.resource_id());
      if (type != read.getResult().getType()) {
        load = builder
                   .create<UnrealizedConversionCastOp>(
                       read.getLoc(), read.getResult().getType(), load)
                   .getResult(0);
      }
      read.getResult().replaceAllUsesWith(load);
      read.erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass() {
  return std::make_unique<LowerGlobalTensorsPass>();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
