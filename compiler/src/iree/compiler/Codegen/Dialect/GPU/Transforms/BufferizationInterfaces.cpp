// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/BufferizationInterfaces.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::replaceOpWithBufferizedValues;

namespace mlir::iree_compiler {

namespace {

/// Bufferization of iree_gpu.tensor_barrier. Always just bufferizes in place
/// and replaces with a barrier.
struct ValueBarrierOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          ValueBarrierOpBufferizationInterface, IREE::GPU::ValueBarrierOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // This op never needs to bufferize to a copy.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const AnalysisState &state) const {
    SmallVector<bufferization::AliasingValue> alist;
    alist.push_back({op->getResult(opOperand.getOperandNumber()),
                     BufferRelation::Equivalent});
    return alist;
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto barrierOp = cast<IREE::GPU::ValueBarrierOp>(op);
    assert(value.getDefiningOp() == barrierOp && "invalid value");
    if (!barrierOp.hasTensorSemantics()) {
      return failure();
    }
    auto srcMemrefType = bufferization::getBufferType(
        barrierOp.getInputs()[cast<OpResult>(value).getResultNumber()], options,
        invocationStack);
    if (failed(srcMemrefType))
      return failure();
    return srcMemrefType;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto barrierOp = cast<IREE::GPU::ValueBarrierOp>(op);
    if (!barrierOp.hasTensorSemantics()) {
      return failure();
    }

    rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());

    SmallVector<Value> buffers;
    buffers.reserve(barrierOp.getNumOperands());
    for (auto input : barrierOp.getInputs()) {
      FailureOr<Value> buffer = getBuffer(rewriter, input, options);
      if (failed(buffer)) {
        return failure();
      }
      buffers.push_back(buffer.value());
    }

    // This operation bufferizes in place
    bufferization::replaceOpWithBufferizedValues(rewriter, op, buffers);
    return success();
  }
};

} // namespace

void registerIREEGPUBufferizationInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, IREE::GPU::IREEGPUDialect *dialect) {
        IREE::GPU::ValueBarrierOp::attachInterface<
            ValueBarrierOpBufferizationInterface>(*context);
      });
}

} // namespace mlir::iree_compiler
