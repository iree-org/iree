// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/TensorExtExternalModels.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"
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

struct TensorLikeTypeExternalModel
    : bufferization::TensorLikeType::ExternalModel<
          TensorLikeTypeExternalModel, IREE::TensorExt::DispatchTensorType> {
  FailureOr<bufferization::BufferLikeType> getBufferType(
      Type type, const bufferization::BufferizationOptions &options,
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
    auto dispatchTensorType = cast<IREE::TensorExt::DispatchTensorType>(type);
    auto tensorType = cast<TensorType>(dispatchTensorType.asRankedTensorType());
    auto memSpace = options.defaultMemorySpaceFn(tensorType);
    if (!memSpace.has_value()) {
      return emitError() << "could not infer memory space";
    }
    return cast<bufferization::BufferLikeType>(
        getMemRefType(tensorType, options, /*layout=*/{}, *memSpace));
  }

  LogicalResult verifyCompatibleBufferType(
      Type type, bufferization::BufferLikeType bufferType,
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
    auto dispatchTensorType = cast<IREE::TensorExt::DispatchTensorType>(type);
    assert(isa<BaseMemRefType>(bufferType) && "expected memref type");
    auto memrefType = cast<ShapedType>(bufferType);
    if (dispatchTensorType.getShape() != memrefType.getShape()) {
      return emitError() << "shapes do not match";
    }
    if (dispatchTensorType.getBoundElementType() !=
        memrefType.getElementType()) {
      return emitError() << "element types do not match";
    }
    return success();
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
            EncodingTypeExternalModel, TensorLikeTypeExternalModel>(*ctx);
      });
}

} // namespace mlir::iree_compiler
