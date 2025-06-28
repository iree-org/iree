// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler::IREE::VectorExt {

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferizationState;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::replaceOpWithNewBufferizedOp;

namespace {

struct TransferGatherOpInterface
    : public BufferizableOpInterface::ExternalModel<
          TransferGatherOpInterface, IREE::VectorExt::TransferGatherOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    assert(isa<RankedTensorType>(opOperand.get().getType()) &&
           "only tensor types expected");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    assert(isa<RankedTensorType>(opOperand.get().getType()) &&
           "only tensor types expected");
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto gatherOp = cast<IREE::VectorExt::TransferGatherOp>(op);
    assert(isa<TensorType>(gatherOp.getShapedType()) &&
           "only tensor types expected");
    FailureOr<Value> buffer =
        getBuffer(rewriter, gatherOp.getBase(), options, state);
    if (failed(buffer))
      return failure();
    replaceOpWithNewBufferizedOp<IREE::VectorExt::TransferGatherOp>(
        rewriter, gatherOp, gatherOp.getVectorType(), *buffer,
        gatherOp.getIndices(), gatherOp.getIndexVecs(), gatherOp.getIndexed(),
        gatherOp.getIndexedMaps(), gatherOp.getPermutationMap(),
        gatherOp.getPadding(), gatherOp.getMask(), gatherOp.getInBoundsAttr());
    return success();
  }
};

} // namespace

void registerIREEVectorExtBufferizationInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, IREEVectorExtDialect *dialect) {
        TransferGatherOp::attachInterface<TransferGatherOpInterface>(*context);
      });
}

} // namespace mlir::iree_compiler::IREE::VectorExt
