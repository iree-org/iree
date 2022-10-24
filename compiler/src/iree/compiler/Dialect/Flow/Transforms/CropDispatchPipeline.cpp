// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static SmallVector<Value, 4> filterTensorValues(ValueRange&& range) {
  SmallVector<Value, 4> result;
  for (auto value : range) {
    if (value.getType().isa<TensorType>()) result.push_back(value);
  }
  return result;
}

class CropDispatchPipelinePass
    : public CropDispatchPipelineBase<CropDispatchPipelinePass> {
 public:
  CropDispatchPipelinePass(int index) : dispatchIndex(index){};

  void runOnOperation() override {
    auto funcOp = getOperation();

    int index = 0;
    DispatchOp targetDispatch;
    for (auto dispatchOp : funcOp.getFunctionBody().getOps<DispatchOp>()) {
      targetDispatch = dispatchOp;
      if (++index > dispatchIndex) break;
    }
    if (index <= dispatchIndex) return;

    OpBuilder builder(targetDispatch);
    auto bufferType = funcOp.getResultTypes()[0];
    builder.setInsertionPointAfter(targetDispatch);
    auto rVal = filterTensorValues(targetDispatch.getResults())[0];
    auto retVal = builder
                      .create<HAL::TensorExportOp>(targetDispatch.getLoc(),
                                                   bufferType, rVal)
                      .getTarget();
    auto terminator =
        *funcOp.getFunctionBody().getOps<func::ReturnOp>().begin();
    builder.setInsertionPointAfter(terminator);
    builder.create<func::ReturnOp>(targetDispatch.getLoc(), retVal);
    terminator.erase();
  }

 private:
  int dispatchIndex;
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCropDispatchPipelinePass(int dispatchIndex) {
  return std::make_unique<CropDispatchPipelinePass>(dispatchIndex);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
