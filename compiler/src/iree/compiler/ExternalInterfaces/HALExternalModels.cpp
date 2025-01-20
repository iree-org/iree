// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/HALExternalModels.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// ValueBoundsOpInterface
//===----------------------------------------------------------------------===//

template <typename IDOp>
struct IDOpValueBoundsInterface : public ValueBoundsOpInterface::ExternalModel<
                                      IDOpValueBoundsInterface<IDOp>, IDOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto boundOp = cast<IDOp>(op);
    assert(value == boundOp.getResult() && "value must be op result");
    cstr.bound(value) >= 0;
    if (boundOp.getUpperBound()) {
      cstr.bound(value) < boundOp.getUpperBound()->getSExtValue();
    }
  }
};

template <typename CountOp>
struct CountOpValueBoundsInterface
    : public ValueBoundsOpInterface::ExternalModel<
          CountOpValueBoundsInterface<CountOp>, CountOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto boundOp = cast<CountOp>(op);
    assert(value == boundOp.getResult() && "value must be op result");
    cstr.bound(value) >= 1;
    if (boundOp.getUpperBound()) {
      cstr.bound(value) <= boundOp.getUpperBound()->getSExtValue();
    }
  }
};

} // namespace

void registerHALExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            IREE::HAL::HALDialect *dialect) {
    IREE::HAL::InterfaceWorkgroupIDOp::attachInterface<
        IDOpValueBoundsInterface<IREE::HAL::InterfaceWorkgroupIDOp>>(*context);

    IREE::HAL::InterfaceWorkgroupSizeOp::attachInterface<
        CountOpValueBoundsInterface<IREE::HAL::InterfaceWorkgroupSizeOp>>(
        *context);
    IREE::HAL::InterfaceWorkgroupCountOp::attachInterface<
        CountOpValueBoundsInterface<IREE::HAL::InterfaceWorkgroupCountOp>>(
        *context);
  });
}
} // namespace mlir::iree_compiler
