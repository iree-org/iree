// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Indexing/IR/IndexingExternalModels.h"

#include "iree/compiler/Dialect/Indexing/IR/IndexingInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler::IREE::Indexing {

namespace {} // namespace

struct AddIOpStaticBoundsOpInterface
    : public StaticBoundsOpInterface::ExternalModel<
          AddIOpStaticBoundsOpInterface, arith::AddIOp> {
  SaturatedIndexRange
  getIndexRange(Operation *op, Value target,
                ArrayRef<SaturatedValueRange> operandRanges) const {
    assert(operandRanges.size() == 2 && "invalid addi operand ranges");
    return std::get<SaturatedIndexRange>(operandRanges[0]) +
           std::get<SaturatedIndexRange>(operandRanges[1]);
  }
};

struct MulIOpStaticBoundsOpInterface
    : public StaticBoundsOpInterface::ExternalModel<
          MulIOpStaticBoundsOpInterface, arith::MulIOp> {
  SaturatedIndexRange
  getIndexRange(Operation *op, Value target,
                ArrayRef<SaturatedValueRange> operandRanges) const {
    assert(operandRanges.size() == 2 && "invalid muli operand ranges");
    return std::get<SaturatedIndexRange>(operandRanges[0]) *
           std::get<SaturatedIndexRange>(operandRanges[1]);
  }
};

struct SubIOpStaticBoundsOpInterface
    : public StaticBoundsOpInterface::ExternalModel<
          SubIOpStaticBoundsOpInterface, arith::SubIOp> {
  SaturatedIndexRange
  getIndexRange(Operation *op, Value target,
                ArrayRef<SaturatedValueRange> operandRanges) const {
    assert(operandRanges.size() == 2 && "invalid subi operand ranges");
    return std::get<SaturatedIndexRange>(operandRanges[0]) -
           std::get<SaturatedIndexRange>(operandRanges[1]);
  }
};

struct ConstantOpStaticBoundsOpInterface
    : public StaticBoundsOpInterface::ExternalModel<
          ConstantOpStaticBoundsOpInterface, arith::ConstantOp> {
  std::optional<SaturatedIndexRange>
  initializeRange(Operation *op, Value target, bool &isFixedPoint) const {
    auto constantOp = llvm::cast<arith::ConstantOp>(op);
    if (!constantOp.getType().isIntOrIndex()) {
      return std::nullopt;
    }
    IntegerAttr value = llvm::cast<IntegerAttr>(constantOp.getValue());
    if (constantOp.getType().isUnsignedInteger()) {
      return SaturatedIndexRange::getConstantRange(value.getUInt());
    }
    if (constantOp.getType().isSignedInteger()) {
      return SaturatedIndexRange::getConstantRange(value.getSInt());
    }
    // It is the responsibility of propagation patterns to be aware of potential
    // signless integer pitfalls. Unfortunately signless integers limit range
    // analysis as we must simultaneously assume a value could be interpreted as
    // signed and unsigned. Addition with a negative integer is an overflow were
    // the value interpreted as an unsigned integer, and thus should invalidate
    // the range.
    return SaturatedIndexRange::getConstantRange(value.getInt());
  }
};

void registerIndexingExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<arith::ArithDialect, tensor::TensorDialect>();

  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::AddIOp::attachInterface<AddIOpStaticBoundsOpInterface>(*ctx);
    arith::ConstantOp::attachInterface<ConstantOpStaticBoundsOpInterface>(*ctx);
    arith::MulIOp::attachInterface<MulIOpStaticBoundsOpInterface>(*ctx);
    arith::SubIOp::attachInterface<SubIOpStaticBoundsOpInterface>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::Indexing
