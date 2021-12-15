// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

class InjectDispatchTracingPass
    : public InjectDispatchTracingBase<InjectDispatchTracingPass> {
 public:
  InjectDispatchTracingPass() = default;

  void runOnOperation() override {
    for (auto dispatchOp : function_like_impl::getFunctionBody(getOperation())
                               .getOps<DispatchOp>()) {
      std::string entryPointName =
          dispatchOp.entry_point().getRootReference().getValue().str();
      for (FlatSymbolRefAttr nestedRef :
           dispatchOp.entry_point().getNestedReferences()) {
        entryPointName = (entryPointName + "::" + nestedRef.getValue()).str();
      }

      // Input tensors:
      OpBuilder builder(dispatchOp);
      builder.create<TensorTraceOp>(
          dispatchOp.getLoc(),
          builder.getStringAttr(entryPointName + " inputs"),
          filterTensorValues(dispatchOp.operands()));

      // Output tensors:
      builder.setInsertionPointAfter(dispatchOp);
      builder.create<TensorTraceOp>(
          dispatchOp.getLoc(),
          builder.getStringAttr(entryPointName + " outputs"),
          filterTensorValues(dispatchOp.results()));
    }
  }
};

std::unique_ptr<Pass> createInjectDispatchTracingPass() {
  return std::make_unique<InjectDispatchTracingPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
