// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- BufferizationExternalModels.cpp -----------------------------------===//
//
// This file implements bufferization interfaces for PCF ops.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/PCF/ExternalInterfaces/BufferizationExternalModels.h"

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace mlir::iree_compiler::IREE::PCF {

using namespace mlir::bufferization;

namespace {

struct GenericOpInterface
    : BufferizableOpInterface::ExternalModel<GenericOpInterface,
                                             PCF::GenericOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Parallel ops can be treated as though they never read.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Generic ops must always be assumed to write to a tensor (init) operand.
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto genericOp = cast<PCF::GenericOp>(op);
    OpResult tiedResult = genericOp.getTiedResult(opOperand);
    if (!tiedResult) {
      return {};
    }

    return {{tiedResult, BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto genericOp = cast<PCF::GenericOp>(op);
    Location loc = genericOp.getLoc();

    SmallVector<Value> newInits;
    newInits.reserve(genericOp.getInits().size());
    for (Value init : genericOp.getInits()) {
      if (isa<RankedTensorType>(init.getType())) {
        FailureOr<Value> newInit = getBuffer(rewriter, init, options, state);
        if (failed(newInit)) {
          return op->emitOpError("failed to get init buffer");
        }
        newInits.push_back(*newInit);
      } else {
        newInits.push_back(init);
      }
    }

    SmallVector<Type> newResultTypes;
    for (Value result : genericOp.getResults()) {
      if (isa<TensorType>(result.getType())) {
        FailureOr<BufferLikeType> resultType =
            bufferization::getBufferType(result, options, state);
        if (failed(resultType)) {
          return failure();
        }
        newResultTypes.push_back(*resultType);
      } else {
        newResultTypes.push_back(result.getType());
      }
    }

    auto newGenericOp = PCF::GenericOp::create(
        rewriter, loc, newResultTypes, genericOp.getScope(), newInits,
        genericOp.getDynamicSizes(), genericOp.getIsTied(),
        genericOp.getNumIterators(), genericOp.getSyncOnReturn());
    newGenericOp.getRegion().takeBody(genericOp.getRegion());
    newGenericOp.getInitializer().takeBody(genericOp.getInitializer());
    replaceOpWithBufferizedValues(rewriter, op, newGenericOp.getResults());
    return success();
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto genericOp = cast<PCF::GenericOp>(op);

    // Block arguments are `pcf.sref`, so this must always be an opresult.
    auto result = cast<OpResult>(value);
    assert(result.getOwner() == op && "invalid value");

    // If the result has a tied init, use that as the buffer type.
    OpOperand *tiedInit = genericOp.getTiedInit(result.getResultNumber());
    if (tiedInit) {
      return bufferization::detail::asMemRefType(bufferization::getBufferType(
          tiedInit->get(), options, state, invocationStack));
    }

    auto resultType = cast<RankedTensorType>(result.getType());

    // Else query the scope for the memory space to allocate for.
    FailureOr<Attribute> memSpace =
        genericOp.getScope().getAllocMemSpace(op->getContext());
    if (failed(memSpace)) {
      return failure();
    }
    return cast<BufferLikeType>(
        getMemRefTypeWithStaticIdentityLayout(resultType, *memSpace));
  }
};

struct LoopOpInterface
    : BufferizableOpInterface::ExternalModel<LoopOpInterface, PCF::LoopOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Parallel ops can be treated as though they never read.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Generic ops must always be assumed to write to a tensor (init) operand.
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto loopOp = cast<PCF::LoopOp>(op);
    OpResult tiedResult = loopOp.getTiedResult(opOperand);
    if (!tiedResult) {
      return {};
    }

    return {{tiedResult, BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto loopOp = cast<PCF::LoopOp>(op);
    Location loc = loopOp.getLoc();

    SmallVector<Value> newInits;
    newInits.reserve(loopOp.getInits().size());
    for (Value init : loopOp.getInits()) {
      if (isa<RankedTensorType>(init.getType())) {
        FailureOr<Value> newInit = getBuffer(rewriter, init, options, state);
        if (failed(newInit)) {
          return op->emitOpError("failed to get init buffer");
        }
        newInits.push_back(*newInit);
      } else {
        newInits.push_back(init);
      }
    }

    SmallVector<Type> newResultTypes;
    for (Value result : loopOp.getResults()) {
      if (isa<TensorType>(result.getType())) {
        FailureOr<BufferLikeType> resultType =
            bufferization::getBufferType(result, options, state);
        if (failed(resultType)) {
          return failure();
        }
        newResultTypes.push_back(*resultType);
      } else {
        newResultTypes.push_back(result.getType());
      }
    }

    auto newLoopOp = PCF::LoopOp::create(
        rewriter, loc, newResultTypes, loopOp.getScope(), loopOp.getCount(),
        newInits, loopOp.getDynamicSizes(), loopOp.getIsTied(),
        loopOp.getSyncOnReturn());
    newLoopOp.getRegion().takeBody(loopOp.getRegion());
    replaceOpWithBufferizedValues(rewriter, op, newLoopOp.getResults());
    return success();
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto loopOp = cast<PCF::LoopOp>(op);

    // Block arguments are `pcf.sref`, so this must always be an opresult.
    auto result = cast<OpResult>(value);
    assert(result.getOwner() == op && "invalid value");

    // If the result has a tied init, use that as the buffer type.
    OpOperand *tiedInit = loopOp.getTiedInit(result.getResultNumber());
    if (tiedInit) {
      return bufferization::detail::asMemRefType(bufferization::getBufferType(
          tiedInit->get(), options, state, invocationStack));
    }

    auto resultType = cast<RankedTensorType>(result.getType());

    // Else query the scope for the memory space to allocate for.
    FailureOr<Attribute> memSpace =
        loopOp.getScope().getAllocMemSpace(op->getContext());
    if (failed(memSpace)) {
      return failure();
    }
    return cast<BufferLikeType>(
        getMemRefTypeWithStaticIdentityLayout(resultType, *memSpace));
  }
};

struct WriteSliceOpInterface
    : BufferizableOpInterface::ExternalModel<WriteSliceOpInterface,
                                             PCF::WriteSliceOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The only valid tensor operand is the source which is always read.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // The only valid tensor operand is the source which is only read.
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto writeOp = cast<PCF::WriteSliceOp>(op);

    if (isa<RankedTensorType>(writeOp.getSourceType())) {
      FailureOr<Value> newSrc =
          getBuffer(rewriter, writeOp.getSource(), options, state);
      if (failed(newSrc)) {
        return failure();
      }
      writeOp.getSourceMutable().assign(*newSrc);
    }
    return success();
  }
};

struct ReadSliceOpInterface
    : BufferizableOpInterface::ExternalModel<ReadSliceOpInterface,
                                             PCF::ReadSliceOp> {
  static MemRefType
  getMaximallyDynamicBufferType(MLIRContext *context,
                                PCF::ShapedRefType sourceType) {
    // Create result type with maximally dynamic layout and no memory space.
    // Layout and memory space aren't known until resolving sref types, after
    // which we will propagate both to this operation's users.
    SmallVector<int64_t> strides(sourceType.getRank(), ShapedType::kDynamic);
    auto layout =
        StridedLayoutAttr::get(context, ShapedType::kDynamic, strides);
    return MemRefType::get(sourceType.getShape(), sourceType.getElementType(),
                           layout,
                           /*memorySpace=*/nullptr);
  }
  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto readOp = cast<PCF::ReadSliceOp>(op);
    return getMaximallyDynamicBufferType(op->getContext(),
                                         readOp.getSourceType());
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto readOp = cast<PCF::ReadSliceOp>(op);

    // Skip vector results.
    if (!isa<RankedTensorType>(readOp.getResultType())) {
      return success();
    }

    // Create result type with maximally dynamic layout and no memory space.
    auto resultType =
        getMaximallyDynamicBufferType(op->getContext(), readOp.getSourceType());

    // GetMemrefOp lets us get a memref out of a read_slice. Accesses to srefs
    // are allowed to ignore accesses to this memref.
    auto getMemrefOp = PCF::GetMemrefOp::create(
        rewriter, readOp.getLoc(), resultType, readOp.getSource(),
        readOp.getMixedOffsets(), readOp.getMixedSizes(),
        readOp.getMixedStrides());
    replaceOpWithBufferizedValues(rewriter, op, getMemrefOp.getResult());
    return success();
  }
};

} // namespace

void registerBufferizationExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, PCF::PCFDialect *dialect) {
    GenericOp::attachInterface<GenericOpInterface>(*ctx);
    LoopOp::attachInterface<LoopOpInterface>(*ctx);
    ReadSliceOp::attachInterface<ReadSliceOpInterface>(*ctx);
    WriteSliceOp::attachInterface<WriteSliceOpInterface>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::PCF
