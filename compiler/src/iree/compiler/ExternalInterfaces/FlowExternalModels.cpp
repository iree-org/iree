// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/FlowExternalModels.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {
namespace {

struct DispatchTensorLoadOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          DispatchTensorLoadOpInterface, IREE::Flow::DispatchTensorLoadOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    assert(value == loadOp.getResult() && "invalid value");
    cstr.bound(value)[dim] == loadOp.getMixedSizes()[dim];
  }
};

} // namespace

void registerFlowExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Flow::FlowDialect *dialect) {
        IREE::Flow::DispatchTensorLoadOp::attachInterface<
            DispatchTensorLoadOpInterface>(*ctx);
      });
}

} // namespace mlir::iree_compiler
