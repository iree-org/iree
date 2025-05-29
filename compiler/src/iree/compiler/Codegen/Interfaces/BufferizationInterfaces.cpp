// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Support/LLVM.h"

using mlir::SubsetInsertionOpInterface;
using mlir::bufferization::AliasingOpOperand;
using mlir::bufferization::AliasingOpOperandList;
using mlir::bufferization::AliasingValue;
using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferizationState;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::eliminateEmptyTensors;
using mlir::bufferization::OneShotAnalysisState;
using mlir::bufferization::OneShotBufferizationOptions;
using mlir::bufferization::replaceOpWithBufferizedValues;
using mlir::bufferization::replaceOpWithNewBufferizedOp;

namespace mlir::iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// IREE specific External models for BufferizableOpInterface.
//===----------------------------------------------------------------------===//

struct DispatchTensorLoadOpInterface
    : public BufferizableOpInterface::ExternalModel<
          DispatchTensorLoadOpInterface,
          IREE::TensorExt::DispatchTensorLoadOp> {
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto loadOp = cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    auto shapedType = llvm::dyn_cast<IREE::TensorExt::DispatchTensorType>(
        loadOp.getSource().getType());
    assert(shapedType && "unexpected source type");
    return shapedType.getAccess() != IREE::TensorExt::TensorAccess::ReadOnly;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto loadOp = cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    auto tensorSubspanOp =
        loadOp.getSource()
            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    assert(tensorSubspanOp && "expected that source is a SubspanOp");
    Value source = findOrCreateSubspanBuffer(rewriter, tensorSubspanOp);

    if (equalTensorShape(loadOp.getType(), loadOp.sizes(),
                         llvm::cast<IREE::TensorExt::DispatchTensorType>(
                             loadOp.getSource().getType()),
                         loadOp.getSourceDims())) {
      // The entire tensor is loaded.
      replaceOpWithBufferizedValues(rewriter, op, source);
      return success();
    }

    // Bufferize to subview.
    auto subviewMemRefType = memref::SubViewOp::inferRankReducedResultType(
        loadOp.getType().getShape(), llvm::cast<MemRefType>(source.getType()),
        loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
        loadOp.getMixedStrides());
    replaceOpWithNewBufferizedOp<memref::SubViewOp>(
        rewriter, op, llvm::cast<MemRefType>(subviewMemRefType), source,
        loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
        loadOp.getMixedStrides());

    return success();
  }
};

struct DispatchTensorStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<
          DispatchTensorStoreOpInterface,
          IREE::TensorExt::DispatchTensorStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto storeOp = cast<IREE::TensorExt::DispatchTensorStoreOp>(op);
    auto tensorSubspanOp =
        storeOp.getTarget()
            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    assert(tensorSubspanOp && "expected that target is a SubspanOp");
    Value target = findOrCreateSubspanBuffer(rewriter, tensorSubspanOp);

    if (!equalTensorShape(
            llvm::cast<RankedTensorType>(storeOp.getValue().getType()),
            storeOp.getSizes(),
            llvm::cast<IREE::TensorExt::DispatchTensorType>(
                storeOp.getTarget().getType()),
            storeOp.getTargetDims())) {
      // Writing to a part of the tensor.
      auto subviewMemRefType =
          llvm::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
              cast<ShapedType>(storeOp.getValue().getType()).getShape(),
              cast<MemRefType>(target.getType()), storeOp.getMixedOffsets(),
              storeOp.getMixedSizes(), storeOp.getMixedStrides()));

      target = rewriter.create<memref::SubViewOp>(
          storeOp->getLoc(), subviewMemRefType, target,
          storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
          storeOp.getMixedStrides());
    } // else: Writing the entire tensor, no subview required.

    auto maybeBuffer =
        getBuffer(rewriter, storeOp->getOpOperand(0).get(), options, state);
    if (failed(maybeBuffer))
      return failure();
    Value srcMemref = *maybeBuffer;

    // If everything bufferized inplace, no copy is needed. We wrote to the
    // target buffer already. The copy folds away in that case.
    if (failed(options.createMemCpy(rewriter, storeOp->getLoc(), srcMemref,
                                    target)))
      return failure();

    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct LoadFromBufferOpInterface
    : public BufferizableOpInterface::ExternalModel<
          LoadFromBufferOpInterface, IREE::Codegen::LoadFromBufferOp> {
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // Walk memref Value producers until a hal.interface.binding.subspan op is
    // found, and check if the subspan is read only.
    Operation *currentOp = op;
    while (currentOp) {
      if (auto subspanOp =
              dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(currentOp)) {
        std::optional<IREE::HAL::DescriptorFlags> descriptorFlags =
            subspanOp.getDescriptorFlags();
        return !descriptorFlags.has_value() ||
               descriptorFlags.value() != IREE::HAL::DescriptorFlags::ReadOnly;
      }
      // There is expected to be only a single memref source for a given memref
      // OpResult, because producers of memref Values are expected to be
      // view-like or cast-like operations. If multiple operands have a memref
      // type, then conservatively return not writable.
      if (llvm::count_if(currentOp->getOperandTypes(),
                         llvm::IsaPred<MemRefType>) != 1) {
        return false;
      }
      // Otherwise, follow the memref operand to find the source buffer.
      for (Value operand : currentOp->getOperands()) {
        if (isa<MemRefType>(operand.getType())) {
          currentOp = operand.getDefiningOp();
          break;
        }
      }
    }
    // Conservatively default to not writable if the source of the buffer is
    // not found.
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto loadOp = cast<IREE::Codegen::LoadFromBufferOp>(op);
    replaceOpWithBufferizedValues(rewriter, op, loadOp.getBuffer());
    return success();
  }
};

struct StoreToBufferOpInterface
    : public BufferizableOpInterface::ExternalModel<
          StoreToBufferOpInterface, IREE::Codegen::StoreToBufferOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto storeOp = cast<IREE::Codegen::StoreToBufferOp>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, storeOp.getTensor(), options, state);
    if (failed(maybeBuffer))
      return failure();
    Value srcMemref = *maybeBuffer;

    // If everything bufferized inplace, no copy is needed. We wrote to the
    // target buffer already. The copy folds away in that case.
    if (failed(options.createMemCpy(rewriter, storeOp.getLoc(), srcMemref,
                                    storeOp.getBuffer())))
      return failure();

    rewriter.eraseOp(storeOp);
    return success();
  }
};
} // namespace

/// Generic conversion for any LinalgExtOp on tensors.
static LogicalResult bufferizeLinalgExtOp(RewriterBase &rewriter,
                                          IREE::LinalgExt::LinalgExtOp op,
                                          const BufferizationOptions &options,
                                          const BufferizationState &state) {
  auto dspOp = cast<DestinationStyleOpInterface>(op.getOperation());

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (dspOp.hasPureBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!dspOp.hasPureTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newOperands, newOutputBuffers;
  AnalysisState analysisState(options);
  newOperands.reserve(op->getNumOperands());

  for (OpOperand &opOperand : op->getOpOperands()) {
    if (dspOp.isScalar(&opOperand)) {
      newOperands.push_back(opOperand.get());
      continue;
    }
    if (!dspOp.isDpsInit(&opOperand)) {
      auto maybeBuffer = getBuffer(rewriter, opOperand.get(), options, state);
      if (failed(maybeBuffer))
        return failure();
      // Input operands are never written to.
      newOperands.push_back(*maybeBuffer);
      continue;
    }
    // New output operands for the cloned op.
    OpResult opResult = dspOp.getTiedOpResult(&opOperand);
    AliasingOpOperandList aliasingOpOperands =
        analysisState.getAliasingOpOperands(opResult);
    assert(aliasingOpOperands.getNumAliases() == 1 && "expected 1 OpOperand");
    FailureOr<Value> resultBuffer = getBuffer(
        rewriter, aliasingOpOperands.getAliases().front().opOperand->get(),
        options, state);
    if (failed(resultBuffer))
      return failure();
    newOperands.push_back(*resultBuffer);
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  auto newOp = cast<IREE::LinalgExt::LinalgExtOp>(mlir::cloneWithoutRegions(
      rewriter, op, /*resultTypes=*/TypeRange{}, newOperands));
  int64_t numRegions = op->getNumRegions();
  for (int64_t i = 0; i < numRegions; ++i) {
    rewriter.inlineRegionBefore(op->getRegion(i), newOp->getRegion(i),
                                newOp->getRegion(i).begin());
  }

  // Replace the results of the old op with the new output buffers.
  bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

/// Bufferization of ops that implement the LinalgExtOp interface. Replace with
/// a new op that operates entirely on memrefs.
template <typename OpTy>
struct LinalgExtOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          LinalgExtOpInterface<OpTy>, OpTy> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // TODO: Revisit this for ScatterOp. We can then get rid of
    //       `bufferizesToMemoryRead` completely.
    return !isa<IREE::LinalgExt::ScatterOp>(op);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    return bufferizeLinalgExtOp(
        rewriter, cast<IREE::LinalgExt::LinalgExtOp>(op), options, state);
  }
};

/// Returns the buffers of the source and destination for pack and unpack ops.
/// Returns a failure if the buffers can not be found.
template <typename OpTy>
static FailureOr<std::pair<Value, Value>>
getSourceAndDestFromPackUnPackOp(RewriterBase &rewriter, OpTy op,
                                 const BufferizationOptions &options,
                                 const BufferizationState &state) {
  static_assert(llvm::is_one_of<OpTy, linalg::PackOp, linalg::UnPackOp>::value);
  Value source;
  auto maybeBuffer = getBuffer(rewriter, op.getSource(), options, state);
  if (failed(maybeBuffer))
    return failure();
  source = *maybeBuffer;

  Value dest;
  AnalysisState analysisState(options);
  AliasingOpOperandList aliasingOpOperands =
      analysisState.getAliasingOpOperands(op->getOpResult(0));
  assert(aliasingOpOperands.getNumAliases() == 1 && "expected 1 OpOperand");
  FailureOr<Value> resultBuffer = getBuffer(
      rewriter, aliasingOpOperands.getAliases().front().opOperand->get(),
      options, state);
  if (failed(resultBuffer))
    return failure();
  dest = *resultBuffer;
  return std::make_pair(source, dest);
}

static LogicalResult bufferizePackOp(RewriterBase &rewriter, linalg::PackOp op,
                                     const BufferizationOptions &options,
                                     const BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto maybeSrcAndDest =
      getSourceAndDestFromPackUnPackOp(rewriter, op, options, state);
  if (failed(maybeSrcAndDest))
    return failure();
  auto [source, dest] = *maybeSrcAndDest;

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  rewriter.create<IREE::LinalgExt::PackOp>(
      op.getLoc(), source, dest, op.getInnerDimsPos(), op.getMixedTiles(),
      op.getPaddingValue(), op.getOuterDimsPerm());

  // Replace the results of the old op with the new output buffers.
  bufferization::replaceOpWithBufferizedValues(rewriter, op, dest);

  return success();
}

static LogicalResult bufferizeUnPackOp(RewriterBase &rewriter,
                                       linalg::UnPackOp op,
                                       const BufferizationOptions &options,
                                       const BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto maybeSrcAndDest =
      getSourceAndDestFromPackUnPackOp(rewriter, op, options, state);
  if (failed(maybeSrcAndDest))
    return failure();
  auto [source, dest] = *maybeSrcAndDest;

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  rewriter.create<IREE::LinalgExt::UnPackOp>(
      op.getLoc(), source, dest, op.getInnerDimsPos(), op.getMixedTiles(),
      op.getOuterDimsPerm());

  // Replace the results of the old op with the new output buffers.
  bufferization::replaceOpWithBufferizedValues(rewriter, op, dest);

  return success();
}

template <typename OpTy>
struct PackUnPackOpInterface
    : public BufferizableOpInterface::ExternalModel<PackUnPackOpInterface<OpTy>,
                                                    OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it has an aliasing OpResult.
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return dpsOp.isDpsInit(&opOperand);
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const AnalysisState &state) const {
    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    return {dpsOp.getDpsInitOperand(opResult.getResultNumber())};
  }

  SmallVector<OpResult> getAliasingValue(Operation *op, OpOperand &opOperand,
                                         const AnalysisState &state) const {
    auto dspOp = cast<DestinationStyleOpInterface>(op);

    // The i-th "out" tensor may alias with the i-th OpResult.
    if (dspOp.isDpsInit(&opOperand))
      return {dspOp.getTiedOpResult(&opOperand)};
    return {};
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const AnalysisState &state) const {
    auto dspOp = cast<DestinationStyleOpInterface>(op);

    // The i-th "out" tensor may alias with the i-th OpResult.
    if (dspOp.isDpsInit(&opOperand))
      return {AliasingValue(dspOp.getTiedOpResult(&opOperand),
                            BufferRelation::Equivalent,
                            /*isDefinite=*/false)};
    return {};
  }

  bufferization::BufferRelation
  bufferRelation(Operation *op, OpResult opResult,
                 const AnalysisState &state) const {
    return bufferization::BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .template Case<linalg::PackOp>([&](auto pack) {
          return bufferizePackOp(rewriter, pack, options, state);
        })
        .template Case<linalg::UnPackOp>([&](auto unpack) {
          return bufferizeUnPackOp(rewriter, unpack, options, state);
        })
        .Default([](auto) { return failure(); });
  }
};

struct DispatchTensorLoadOpSubsetInterface
    : public SubsetOpInterface::ExternalModel<
          DispatchTensorLoadOpSubsetInterface,
          IREE::TensorExt::DispatchTensorLoadOp> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // Returns true if the value of a `loadOp` bufferizes to an equivalent
    // DispatchTensorStoreOp result that bufferizes inplace.
    auto loadOp = cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    auto storeOp = dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(op);
    if (!storeOp)
      return false;
    return equivalenceFn(loadOp.getSource(), storeOp.getTarget());
  }

  bool operatesOnDisjointSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // TODO: This is a new entry point and not clear it is correct.
    return false;
  }
};

struct DispatchTensorStoreOpSubsetInterface
    : public SubsetOpInterface::ExternalModel<
          DispatchTensorStoreOpSubsetInterface,
          IREE::TensorExt::DispatchTensorStoreOp> {

  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // Returns true if the value of a `storeOp` bufferizes to an equivalent
    // DispatchTensorLoadOp result that bufferizes inplace.
    auto storeOp = cast<IREE::TensorExt::DispatchTensorStoreOp>(op);
    auto loadOp = dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    if (!loadOp)
      return false;
    return equivalenceFn(loadOp.getSource(), storeOp.getTarget());
  }

  bool operatesOnDisjointSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // TODO: This is a new entry point and not clear it is correct.
    return false;
  }
};

struct DispatchTensorStoreOpSubsetInsertionInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          DispatchTensorStoreOpSubsetInsertionInterface,
          IREE::TensorExt::DispatchTensorStoreOp> {

  OpOperand &getSourceOperand(Operation *op) const {
    return op->getOpOperand(0);
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return op->getOpOperand(1);
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    auto storeOp = cast<IREE::TensorExt::DispatchTensorStoreOp>(op);
    auto loadOp = builder.create<IREE::TensorExt::DispatchTensorLoadOp>(
        loc, llvm::cast<RankedTensorType>(storeOp.getValue().getType()),
        storeOp.getTarget(), storeOp.getTargetDims(), storeOp.getMixedOffsets(),
        storeOp.getMixedSizes(), storeOp.getMixedStrides());
    return loadOp.getResult();
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    auto storeOp = cast<IREE::TensorExt::DispatchTensorStoreOp>(op);
    SmallVector<Value> neededValues;
    // Collect all values that are needed to construct the replacement op.
    neededValues.push_back(storeOp.getTarget());
    neededValues.append(storeOp.getTargetDims().begin(),
                        storeOp.getTargetDims().end());
    neededValues.append(storeOp.getOffsets().begin(),
                        storeOp.getOffsets().end());
    neededValues.append(storeOp.getSizes().begin(), storeOp.getSizes().end());
    neededValues.append(storeOp.getStrides().begin(),
                        storeOp.getStrides().end());
    return neededValues;
  }
};

struct LoadFromBufferOpSubsetInterface
    : public SubsetOpInterface::ExternalModel<LoadFromBufferOpSubsetInterface,
                                              IREE::Codegen::LoadFromBufferOp> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    auto loadOp = cast<IREE::Codegen::LoadFromBufferOp>(op);
    Operation *otherOp = candidate.getOperation();
    if (auto storeOp = dyn_cast<IREE::Codegen::StoreToBufferOp>(otherOp)) {
      return equivalenceFn(loadOp.getBuffer(), storeOp.getBuffer());
    }
    if (auto otherLoadOp = dyn_cast<IREE::Codegen::LoadFromBufferOp>(otherOp)) {
      return equivalenceFn(loadOp.getBuffer(), otherLoadOp.getBuffer());
    }
    return false;
  }

  bool operatesOnDisjointSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // TODO: This is a new entry point and not clear it is correct.
    return false;
  }
};

struct StoreToBufferOpSubsetInterface
    : public SubsetOpInterface::ExternalModel<StoreToBufferOpSubsetInterface,
                                              IREE::Codegen::StoreToBufferOp> {

  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // Returns true if the value of a `storeOp` bufferizes to an equivalent
    // DispatchTensorLoadOp result that bufferizes inplace.
    auto storeOp = cast<IREE::Codegen::StoreToBufferOp>(op);
    Operation *otherOp = candidate.getOperation();
    if (auto otherStoreOp = dyn_cast<IREE::Codegen::StoreToBufferOp>(otherOp)) {
      return equivalenceFn(storeOp.getBuffer(), otherStoreOp.getBuffer());
    }
    if (auto loadOp = dyn_cast<IREE::Codegen::LoadFromBufferOp>(otherOp)) {
      return equivalenceFn(storeOp.getBuffer(), loadOp.getBuffer());
    }
    return false;
  }

  bool operatesOnDisjointSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // TODO: This is a new entry point and not clear it is correct.
    return false;
  }
};

struct StoreToBufferOpSubsetInsertionInterface
    : public SubsetInsertionOpInterface::ExternalModel<
          StoreToBufferOpSubsetInsertionInterface,
          IREE::Codegen::StoreToBufferOp> {

  OpOperand &getSourceOperand(Operation *op) const {
    return cast<IREE::Codegen::StoreToBufferOp>(op).getTensorMutable();
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return cast<IREE::Codegen::StoreToBufferOp>(op).getBufferMutable();
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    auto storeOp = cast<IREE::Codegen::StoreToBufferOp>(op);
    auto loadOp = builder.create<IREE::Codegen::LoadFromBufferOp>(
        loc, storeOp.getTensor().getType(), storeOp.getBuffer());
    return loadOp.getResult();
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    auto storeOp = cast<IREE::Codegen::StoreToBufferOp>(op);
    return {storeOp.getBuffer()};
  }
};

//===----------------------------------------------------------------------===//
// IREE specific post analysis transformations.
//===----------------------------------------------------------------------===//

void registerBufferizationInterfaces(DialectRegistry &registry) {
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register IREE operations.
  registerIREEGPUBufferizationInterfaces(registry);
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::TensorExt::IREETensorExtDialect *dialect) {
        // DispatchTensorLoadOp
        IREE::TensorExt::DispatchTensorLoadOp::attachInterface<
            DispatchTensorLoadOpInterface>(*ctx);
        IREE::TensorExt::DispatchTensorLoadOp::attachInterface<
            DispatchTensorLoadOpSubsetInterface>(*ctx);
        // DispatchTensorStoreOp
        IREE::TensorExt::DispatchTensorStoreOp::attachInterface<
            DispatchTensorStoreOpInterface>(*ctx);
        IREE::TensorExt::DispatchTensorStoreOp::attachInterface<
            DispatchTensorStoreOpSubsetInterface>(*ctx);
        IREE::TensorExt::DispatchTensorStoreOp::attachInterface<
            DispatchTensorStoreOpSubsetInsertionInterface>(*ctx);
      });
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        IREE::Codegen::LoadFromBufferOp::attachInterface<
            LoadFromBufferOpInterface, LoadFromBufferOpSubsetInterface>(*ctx);
        IREE::Codegen::StoreToBufferOp::attachInterface<
            StoreToBufferOpInterface, StoreToBufferOpSubsetInterface,
            StoreToBufferOpSubsetInsertionInterface>(*ctx);
      });
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    IREE::LinalgExt::FftOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::FftOp>>(*ctx);
    IREE::LinalgExt::PackOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::PackOp>>(*ctx);
    IREE::LinalgExt::UnPackOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::UnPackOp>>(*ctx);
    IREE::LinalgExt::ScanOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::ScanOp>>(*ctx);
    IREE::LinalgExt::ScatterOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::ScatterOp>>(*ctx);
    IREE::LinalgExt::GatherOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::GatherOp>>(*ctx);
    IREE::LinalgExt::SortOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::SortOp>>(*ctx);
    IREE::LinalgExt::TopkOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::TopkOp>>(*ctx);
    IREE::LinalgExt::WinogradInputTransformOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::WinogradInputTransformOp>>(*ctx);
    IREE::LinalgExt::WinogradFilterTransformOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::WinogradFilterTransformOp>>(*ctx);
    IREE::LinalgExt::WinogradOutputTransformOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::WinogradOutputTransformOp>>(*ctx);
    IREE::LinalgExt::AttentionOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::AttentionOp>>(*ctx);
    IREE::LinalgExt::MapScatterOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::MapScatterOp>>(*ctx);
  });
  registry.insert<linalg::LinalgDialect>();
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::PackOp::attachInterface<PackUnPackOpInterface<linalg::PackOp>>(
        *ctx);
    linalg::UnPackOp::attachInterface<PackUnPackOpInterface<linalg::UnPackOp>>(
        *ctx);
  });
}

} // namespace mlir::iree_compiler
