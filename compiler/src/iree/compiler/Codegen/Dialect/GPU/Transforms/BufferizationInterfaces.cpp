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
struct TensorBarrierOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          TensorBarrierOpBufferizationInterface, IREE::GPU::TensorBarrierOp> {
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
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto barrierOp = cast<IREE::GPU::TensorBarrierOp>(op);
    assert(value == barrierOp.getResult() && "invalid value");
    auto srcMemrefType = bufferization::getBufferType(barrierOp.getInput(),
                                                      options, invocationStack);
    if (failed(srcMemrefType))
      return failure();
    return srcMemrefType;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto barrierOp = cast<IREE::GPU::TensorBarrierOp>(op);
    FailureOr<Value> buffer =
        getBuffer(rewriter, barrierOp.getInput(), options);
    if (failed(buffer)) {
      return failure();
    }

    rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());

    // This operation bufferizes in place
    bufferization::replaceOpWithBufferizedValues(rewriter, op, *buffer);
    return success();
  }
};

} // namespace

void registerIREEGPUBufferizationInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, IREE::GPU::IREEGPUDialect *dialect) {
        IREE::GPU::TensorBarrierOp::attachInterface<
            TensorBarrierOpBufferizationInterface>(*context);
      });
}

} // namespace mlir::iree_compiler
