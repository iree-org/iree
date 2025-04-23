// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/TensorExtExternalModels.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// Op Interfaces
//===----------------------------------------------------------------------===//

struct DispatchTensorLoadOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          DispatchTensorLoadOpInterface,
          IREE::TensorExt::DispatchTensorLoadOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto loadOp = cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    assert(value == loadOp.getResult() && "invalid value");
    cstr.bound(value)[dim] == loadOp.getMixedSizes()[dim];
  }
};

struct WorkloadOrdinalOpInterface
    : public ValueBoundsOpInterface::ExternalModel<
          WorkloadOrdinalOpInterface,
          IREE::TensorExt::DispatchWorkloadOrdinalOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto ordinalOp = cast<IREE::TensorExt::DispatchWorkloadOrdinalOp>(op);
    assert(value == ordinalOp.getResult() && "value must be op result");
    cstr.bound(value) == ordinalOp.getOperand();
  }
};

//===----------------------------------------------------------------------===//
// Type Interfaces
//===----------------------------------------------------------------------===//

struct EncodingTypeExternalModel
    : public IREE::Encoding::EncodingTypeInterface::ExternalModel<
          EncodingTypeExternalModel, IREE::TensorExt::DispatchTensorType> {

  Type getEncodingType(Type type) const {
    auto dispatchTensorType = cast<IREE::TensorExt::DispatchTensorType>(type);
    return dispatchTensorType.getBoundType();
  }

  Type updateEncoding(Type type, Attribute encoding) const {
    auto dispatchTensorType = cast<IREE::TensorExt::DispatchTensorType>(type);
    return IREE::TensorExt::DispatchTensorType::get(
        dispatchTensorType.getAccess(), dispatchTensorType.getShape(),
        dispatchTensorType.getBoundElementType(), encoding);
  }
};

} // namespace

void registerTensorExtExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::TensorExt::IREETensorExtDialect *dialect) {
        IREE::TensorExt::DispatchTensorLoadOp::attachInterface<
            DispatchTensorLoadOpInterface>(*ctx);
        IREE::TensorExt::DispatchWorkloadOrdinalOp::attachInterface<
            WorkloadOrdinalOpInterface>(*ctx);
        IREE::TensorExt::DispatchTensorType::attachInterface<
            EncodingTypeExternalModel>(*ctx);
      });
}

} // namespace mlir::iree_compiler
