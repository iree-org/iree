// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::IREE::Flow {

static std::string inferTraceKey(Operation *op) {
  return TypeSwitch<Operation *, std::string>(op)
      .Case<IREE::Flow::DispatchOp>(
          [&](auto op) { return op.getEntryPointName(); })
      .Case<IREE::Util::CallOp>([&](auto op) { return op.getCallee().str(); })
      .Default([](auto *op) { return op->getName().getStringRef().str(); });
}

static SmallVector<Value> filterTensorValues(ValueRange &&range) {
  SmallVector<Value> result;
  for (auto value : range) {
    if (llvm::isa<TensorType>(value.getType()))
      result.push_back(value);
  }
  return result;
}

static SmallVector<Value> getTensorOperands(Operation *op) {
  if (auto dispatchRegionOp = dyn_cast<IREE::Flow::DispatchRegionOp>(op)) {
    llvm::SetVector<Value> argumentsSet;
    mlir::getUsedValuesDefinedAbove(dispatchRegionOp.getBody(), argumentsSet);
    return filterTensorValues(argumentsSet.takeVector());
  }
  return filterTensorValues(op->getOperands());
}

static void injectTracingOnOp(Operation *op, StringRef traceKey) {
  OpBuilder builder(op);
  auto inputTensors = getTensorOperands(op);
  if (!inputTensors.empty()) {
    builder.create<IREE::Flow::TensorTraceOp>(
        op->getLoc(), builder.getStringAttr(traceKey + " inputs"),
        inputTensors);
  }

  builder.setInsertionPointAfter(op);
  auto outputTensors = filterTensorValues(op->getResults());
  if (!outputTensors.empty()) {
    builder.create<IREE::Flow::TensorTraceOp>(
        op->getLoc(), builder.getStringAttr(traceKey + " outputs"),
        outputTensors);
  }
}

class InjectTensorTracingPass
    : public InjectTensorTracingBase<InjectTensorTracingPass> {
public:
  InjectTensorTracingPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, IREE::Flow::FlowDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto attrName = StringAttr::get(&getContext(), "iree.tensor.trace");
    auto funcOp = getOperation();
    funcOp.walk([&](Operation *op) {
      if (auto attr = op->getAttr(attrName)) {
        std::string traceKey;
        if (auto stringAttr = dyn_cast<StringAttr>(attr))
          traceKey = stringAttr.getValue().str();
        else
          traceKey = inferTraceKey(op);
        injectTracingOnOp(op, traceKey);
        op->removeAttr(attrName);
      }
    });
  }
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createInjectTensorTracingPass() {
  return std::make_unique<InjectTensorTracingPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
