// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/BufferizationInterfaces.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::replaceOpWithBufferizedValues;

namespace mlir::iree_compiler {

namespace {

static FailureOr<SmallVector<Value>>
getBuffers(RewriterBase &rewriter, const MutableOperandRange &operands,
           const BufferizationOptions &options,
           const bufferization::BufferizationState &state) {
  SmallVector<Value> result;
  for (OpOperand &opOperand : operands) {
    if (isa<TensorType>(opOperand.get().getType())) {
      FailureOr<Value> resultBuffer =
          getBuffer(rewriter, opOperand.get(), options, state);
      if (failed(resultBuffer))
        return failure();
      result.push_back(*resultBuffer);
    } else {
      result.push_back(opOperand.get());
    }
  }
  return result;
}

/// Bufferization of iree_gpu.barrier_region. Always just bufferizes in place
/// and gets inlined with barriers.
struct BarrierRegionOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          BarrierRegionOpBufferizationInterface, IREE::GPU::BarrierRegionOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // This op itself never needs to bufferize to a copy. It's possible
    // that operations within its body will need to bufferize to a copy,
    // but those copies should happen between the two barriers.
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    return true;
  }

  // Operands alias with the region operands.
  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const AnalysisState &state) const {
    SmallVector<bufferization::AliasingValue> alist;
    auto barrierOp = cast<IREE::GPU::BarrierRegionOp>(op);
    alist.push_back(
        {barrierOp.getBody()->getArguments()[opOperand.getOperandNumber()],
         BufferRelation::Equivalent, /*isDefinite=*/true});
    return alist;
  }

  bufferization::AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const AnalysisState &state) const {
    auto barrierOp = cast<IREE::GPU::BarrierRegionOp>(op);
    bufferization::AliasingOpOperandList result;
    if (auto opResult = dyn_cast<OpResult>(value)) {
      int64_t resultNum = opResult.getResultNumber();
      auto yieldOp =
          cast<IREE::GPU::YieldOp>(barrierOp.getBody()->getTerminator());
      result.addAlias(bufferization::AliasingOpOperand(
          &yieldOp->getOpOperand(resultNum), BufferRelation::Equivalent,
          /*isDefinite=*/true));
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      result.addAlias(bufferization::AliasingOpOperand(
          &barrierOp->getOpOperand(blockArg.getArgNumber()),
          BufferRelation::Equivalent,
          /*isDefinite=*/true));
    }
    return result;
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const bufferization::BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto barrierOp = cast<IREE::GPU::BarrierRegionOp>(op);

    FailureOr<BaseMemRefType> memrefType = failure();
    if (auto opResult = dyn_cast<OpResult>(value)) {
      int64_t resultNum = opResult.getResultNumber();
      memrefType = bufferization::getBufferType(
          barrierOp.getBody()->getTerminator()->getOperand(resultNum), options,
          state, invocationStack);
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      int64_t argNum = blockArg.getArgNumber();
      memrefType = bufferization::getBufferType(
          barrierOp.getOperand(argNum), options, state, invocationStack);
    }
    if (failed(memrefType))
      return failure();
    return memrefType;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto barrierOp = cast<IREE::GPU::BarrierRegionOp>(op);
    auto terminator =
        cast<IREE::GPU::YieldOp>(barrierOp.getBody()->getTerminator());

    FailureOr<SmallVector<Value>> newOperands =
        getBuffers(rewriter, barrierOp.getInputsMutable(), options, state);
    FailureOr<SmallVector<Value>> newResults =
        getBuffers(rewriter, terminator.getValuesMutable(), options, state);
    if (failed(newOperands) || failed(newResults)) {
      return failure();
    }

    SmallVector<Value> tensorizedOperands;
    for (auto [type, replacement] :
         llvm::zip_equal(barrierOp.getOperandTypes(), *newOperands)) {
      if (!isa<RankedTensorType>(type)) {
        tensorizedOperands.push_back(replacement);
        continue;
      }
      tensorizedOperands.push_back(rewriter
                                       .create<bufferization::ToTensorOp>(
                                           replacement.getLoc(), replacement)
                                       .getResult());
    }

    rewriter.setInsertionPoint(barrierOp);
    rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());
    rewriter.setInsertionPointAfter(barrierOp);
    auto afterBarrier = rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());

    rewriter.inlineBlockBefore(barrierOp.getBody(), afterBarrier,
                               tensorizedOperands);

    bufferization::replaceOpWithBufferizedValues(rewriter, op, *newResults);
    rewriter.eraseOp(terminator);
    return success();
  }
};

/// Bufferization of iree_gpu.value_barrier. Always just bufferizes in place
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
                const bufferization::BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto barrierOp = cast<IREE::GPU::ValueBarrierOp>(op);
    assert(value.getDefiningOp() == barrierOp && "invalid value");
    if (!barrierOp.hasTensorSemantics()) {
      return failure();
    }
    auto srcMemrefType = bufferization::getBufferType(
        barrierOp.getInputs()[cast<OpResult>(value).getResultNumber()], options,
        state, invocationStack);
    if (failed(srcMemrefType))
      return failure();
    return srcMemrefType;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto barrierOp = cast<IREE::GPU::ValueBarrierOp>(op);
    if (!barrierOp.hasTensorSemantics()) {
      return failure();
    }

    rewriter.create<gpu::BarrierOp>(barrierOp.getLoc());

    SmallVector<Value> buffers;
    buffers.reserve(barrierOp.getNumOperands());
    for (auto input : barrierOp.getInputs()) {
      FailureOr<Value> buffer = getBuffer(rewriter, input, options, state);
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

/// Bufferization of iree_gpu.yield. Bufferized as part of their enclosing ops,
/// so this is for analysis only.
struct YieldOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          YieldOpBufferizationInterface, IREE::GPU::YieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const AnalysisState &state) const {
    assert(isa<IREE::GPU::BarrierRegionOp>(op->getParentOp()));
    return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
             BufferRelation::Equivalent}};
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    // Yield operands always bufferize inplace. Otherwise, an alloc + copy
    // may be generated inside the block. We should not return/yield allocations
    // when possible.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto yieldOp = cast<IREE::GPU::YieldOp>(op);

    SmallVector<Value> newResults;
    for (const auto &it : llvm::enumerate(yieldOp.getValues())) {
      Value value = it.value();
      if (isa<TensorType>(value.getType())) {
        FailureOr<Value> maybeBuffer =
            getBuffer(rewriter, value, options, state);
        if (failed(maybeBuffer))
          return failure();
        newResults.push_back(*maybeBuffer);
      } else {
        newResults.push_back(value);
      }
    }

    bufferization::replaceOpWithNewBufferizedOp<IREE::GPU::YieldOp>(
        rewriter, op, newResults);
    return success();
  }
};

/// AMD Specific Ops

static bool hasStorageBufferMemSpace(BaseMemRefType m) {
  Attribute maybeMemorySpace = m.getMemorySpace();
  Builder b(m.getContext());
  Attribute storageBufferMemSpace = b.getAttr<IREE::HAL::DescriptorTypeAttr>(
      IREE::HAL::DescriptorType::StorageBuffer);
  return maybeMemorySpace && maybeMemorySpace == storageBufferMemSpace;
}

/// Bufferization of iree_gpu.buffer_resource_cast. Bufferizes to
/// amdgpu.fat_raw_buffer_cast if the source memref is of memory space
/// `storage_buffer`, else just forwards the input. This op never
/// reads or writes.
struct BufferResourceCastOpBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          BufferResourceCastOpBufferizationInterface,
          IREE::GPU::BufferResourceCastOp> {
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
    auto castOp = cast<IREE::GPU::BufferResourceCastOp>(op);
    SmallVector<bufferization::AliasingValue> alist;
    if (opOperand.get() == castOp.getInput()) {
      alist.push_back({castOp.getResult(), BufferRelation::Equivalent});
    }
    return alist;
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const bufferization::BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto castOp = cast<IREE::GPU::BufferResourceCastOp>(op);
    assert(value.getDefiningOp() == castOp && "invalid value");
    auto srcMemrefType = bufferization::getBufferType(
        castOp.getInput(), options, state, invocationStack);
    if (failed(srcMemrefType))
      return failure();

    if (!hasStorageBufferMemSpace(srcMemrefType.value())) {
      return srcMemrefType;
    }

    auto rankedSrcType = cast<MemRefType>(srcMemrefType.value());

    // Sad AMDGPU dep.
    Attribute bufferMemSpace = amdgpu::AddressSpaceAttr::get(
        op->getContext(), amdgpu::AddressSpace::FatRawBuffer);
    return MemRefType::get(rankedSrcType.getShape(),
                           rankedSrcType.getElementType(),
                           rankedSrcType.getLayout(), bufferMemSpace);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto castOp = cast<IREE::GPU::BufferResourceCastOp>(op);

    FailureOr<Value> buffer =
        getBuffer(rewriter, castOp.getInput(), options, state);
    if (failed(buffer)) {
      return failure();
    }

    // This operation either bufferizes in place or with a cast.
    if (hasStorageBufferMemSpace(cast<MemRefType>(buffer.value().getType()))) {
      Location loc = castOp.getLoc();
      Value cacheSwizzleStride = Value{};
      if (auto maybeIndexCacheSwizzle = castOp.getCacheSwizzleStride()) {
        // Cache swizzle supports only upto 8k stride. Also simply swizzling the
        // largest available stride (8k) doesn't help those unsupported large
        // stride. Especially better to avoid using the stride which is 2^N when
        // N>13, e.g. by add padding to the buffer.
        //
        // stride[13:0] = swizzling stride
        // stride[14] = swizzle enabling bit
        // FatRawBufferCast's lowering handles this for us. Just truncate to 14
        // bits.
        Type i14Type = rewriter.getIntegerType(14);
        cacheSwizzleStride = rewriter.create<arith::IndexCastOp>(
            loc, i14Type, maybeIndexCacheSwizzle);
      }
      buffer = rewriter
                   .create<amdgpu::FatRawBufferCastOp>(
                       loc, buffer.value(), /*validBytes=*/Value{},
                       /*cacheSwizzleStride=*/cacheSwizzleStride,
                       /*boundsCheck=*/true,
                       /*resetOffset=*/true)
                   .getResult();
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op, buffer.value());
    return success();
  }
};

} // namespace

void registerIREEGPUBufferizationInterfaces(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, IREE::GPU::IREEGPUDialect *dialect) {
        IREE::GPU::BarrierRegionOp::attachInterface<
            BarrierRegionOpBufferizationInterface>(*context);
        IREE::GPU::ValueBarrierOp::attachInterface<
            ValueBarrierOpBufferizationInterface>(*context);
        IREE::GPU::YieldOp::attachInterface<YieldOpBufferizationInterface>(
            *context);

        IREE::GPU::BufferResourceCastOp::attachInterface<
            BufferResourceCastOpBufferizationInterface>(*context);
      });
}

} // namespace mlir::iree_compiler
