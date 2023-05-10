// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

namespace {
struct ConvertMHLOCustomCallsToCallsPass
    : public ConvertMHLOCustomCallsToCallsBase<
          ConvertMHLOCustomCallsToCallsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    mhlo::MhloDialect, shape::ShapeDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    auto context = &getContext();
    auto moduleOp = getOperation();
    IREE::HAL::FenceType fenceType = IREE::HAL::FenceType::get(context);
    OpBuilder b(context);

    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!funcOp.empty()) {
        for (auto customCallOp :
                  funcOp.getFunctionBody().getOps<mhlo::CustomCallOp>()) {
          mlir::StringRef callTargetName
            = customCallOp.getCallTargetName();
          bool hasSideEffect = customCallOp.getHasSideEffect();
          mlir::ArrayAttr calledComputations
            = customCallOp.getCalledComputations();
          if (calledComputations.size() == 1) {
            callTargetName
              = calledComputations[0].cast<mlir::StringAttr>().getValue();
          }
          SmallVector<Type> inputTypes, outputTypes;
          auto inputs = customCallOp.getInputs();
          auto actualInputTypes = inputs.getTypes();
          inputTypes.append(actualInputTypes.begin(), actualInputTypes.end());
          inputTypes.push_back(fenceType);
          inputTypes.push_back(fenceType);
          // TODO: Fix aliasing.
          auto outputs = customCallOp.getResults();
          auto actualOutputTypes = outputs.getTypes();
          inputTypes.append(actualOutputTypes.begin(),
                            actualOutputTypes.end());
          outputTypes.append(actualOutputTypes.begin(),
                             actualOutputTypes.end());
          auto funcType = FunctionType::get(context, {} /*inputTypes*/, outputTypes);
          auto newFuncOp = func::FuncOp::create(customCallOp.getLoc(),
                                                callTargetName, funcType);
          newFuncOp.setPrivate();
          newFuncOp->setAttr("iree.abi.model", 
                             StringAttr::get(context, "coarse-fences"));
          if (!hasSideEffect)
            newFuncOp->setAttr("nosideeffects",  UnitAttr::get(context));
          if (!moduleOp.lookupSymbol<func::FuncOp>(callTargetName)) {
            b.setInsertionPointToStart(moduleOp.getBody());
            b.insert(newFuncOp.getOperation());
          }
          b.setInsertionPoint(customCallOp);
          auto callOp = b.create<func::CallOp>(customCallOp.getLoc(), newFuncOp);
          customCallOp.replaceAllUsesWith(callOp.getResults());
          customCallOp.erase();
        }
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMHLOCustomCallsToCallsPass() {
  return std::make_unique<ConvertMHLOCustomCallsToCallsPass>();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
