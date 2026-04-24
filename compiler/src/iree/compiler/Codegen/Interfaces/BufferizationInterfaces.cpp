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
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
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
    : BufferizableOpInterface::ExternalModel<
          DispatchTensorLoadOpInterface,
          IREE::TensorExt::DispatchTensorLoadOp> {
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

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto loadOp = cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    auto shapedType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
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
                         cast<IREE::TensorExt::DispatchTensorType>(
                             loadOp.getSource().getType()),
                         loadOp.getSourceDims())) {
      // The entire tensor is loaded.
      replaceOpWithBufferizedValues(rewriter, op, source);
      return success();
    }

    // Bufferize to subview.
    auto subviewMemRefType = memref::SubViewOp::inferRankReducedResultType(
        loadOp.getType().getShape(), cast<MemRefType>(source.getType()),
        loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
        loadOp.getMixedStrides());
    replaceOpWithNewBufferizedOp<memref::SubViewOp>(
        rewriter, op, cast<MemRefType>(subviewMemRefType), source,
        loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
        loadOp.getMixedStrides());

    return success();
  }
};

struct DispatchTensorStoreOpInterface
    : BufferizableOpInterface::ExternalModel<
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

    if (!equalTensorShape(cast<RankedTensorType>(storeOp.getValue().getType()),
                          storeOp.getSizes(),
                          cast<IREE::TensorExt::DispatchTensorType>(
                              storeOp.getTarget().getType()),
                          storeOp.getTargetDims())) {
      // Writing to a part of the tensor.
      auto subviewMemRefType =
          cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
              cast<ShapedType>(storeOp.getValue().getType()).getShape(),
              cast<MemRefType>(target.getType()), storeOp.getMixedOffsets(),
              storeOp.getMixedSizes(), storeOp.getMixedStrides()));

      target = memref::SubViewOp::create(
          rewriter, storeOp->getLoc(), subviewMemRefType, target,
          storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
          storeOp.getMixedStrides());
    } // else: Writing the entire tensor, no subview required.

    auto maybeBuffer =
        getBuffer(rewriter, storeOp->getOpOperand(0).get(), options, state);
    if (failed(maybeBuffer)) {
      return failure();
    }
    Value srcMemref = *maybeBuffer;

    // If everything bufferized inplace, no copy is needed. We wrote to the
    // target buffer already. The copy folds away in that case.
    if (failed(options.createMemCpy(rewriter, storeOp->getLoc(), srcMemref,
                                    target))) {
      return failure();
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct LoadFromBufferOpInterface
    : BufferizableOpInterface::ExternalModel<LoadFromBufferOpInterface,
                                             IREE::Codegen::LoadFromBufferOp> {
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // Search for a hal.interface.binding.subspan op that is the source of the
    // buffer, and check if the subspan is read only.
    auto loadFromBufferOp = cast<IREE::Codegen::LoadFromBufferOp>(op);
    std::optional<IREE::HAL::InterfaceBindingSubspanOp> subspanOp =
        getSourceSubspanMemref(
            cast<TypedValue<MemRefType>>(loadFromBufferOp.getBuffer()));
    // Conservatively return false if the subspan is not found.
    if (!subspanOp) {
      return false;
    }
    std::optional<IREE::HAL::DescriptorFlags> descriptorFlags =
        subspanOp->getDescriptorFlags();
    return !descriptorFlags.has_value() ||
           !bitEnumContainsAll(*descriptorFlags,
                               IREE::HAL::DescriptorFlags::ReadOnly);
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
    : BufferizableOpInterface::ExternalModel<StoreToBufferOpInterface,
                                             IREE::Codegen::StoreToBufferOp> {
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
    if (failed(maybeBuffer)) {
      return failure();
    }
    Value srcMemref = *maybeBuffer;

    // If everything bufferized inplace, no copy is needed. We wrote to the
    // target buffer already. The copy folds away in that case.
    if (failed(options.createMemCpy(rewriter, storeOp.getLoc(), srcMemref,
                                    storeOp.getBuffer()))) {
      return failure();
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct SwizzleHintOpInterface final
    : BufferizableOpInterface::ExternalModel<SwizzleHintOpInterface,
                                             IREE::Codegen::SwizzleHintOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  bool mustBufferizeInPlace(Operation *, OpOperand &,
                            const AnalysisState &) const {
    return true;
  }

  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &, const AnalysisState &) const {
    return {{op->getResult(0), BufferRelation::Equivalent, /*definite=*/true}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto hintOp = cast<IREE::Codegen::SwizzleHintOp>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, hintOp.getOperand(), options, state);
    if (failed(maybeBuffer)) {
      return failure();
    }
    replaceOpWithNewBufferizedOp<IREE::Codegen::SwizzleHintOp>(
        rewriter, op, *maybeBuffer, hintOp.getSwizzle());
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
  if (dspOp.hasPureBufferSemantics()) {
    return success();
  }

  // New input operands for the cloned op.
  SmallVector<Value> newOperands, newOutputBuffers;
  AnalysisState analysisState(options);
  newOperands.reserve(op->getNumOperands());

  for (OpOperand &opOperand : op->getOpOperands()) {
    if (dspOp.isScalar(&opOperand)) {
      newOperands.push_back(opOperand.get());
      continue;
    }
    // Skip operands that are already memrefs (mixed tensor-buffer semantics).
    if (isa<MemRefType>(opOperand.get().getType())) {
      newOperands.push_back(opOperand.get());
      continue;
    }
    if (!dspOp.isDpsInit(&opOperand)) {
      auto maybeBuffer = getBuffer(rewriter, opOperand.get(), options, state);
      if (failed(maybeBuffer)) {
        return failure();
      }
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
    if (failed(resultBuffer)) {
      return failure();
    }
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
    : bufferization::DstBufferizableOpInterfaceExternalModel<
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

template <typename... Ops>
struct LinalgExtOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (void)std::initializer_list<int>{
        0, (Ops::template attachInterface<LinalgExtOpInterface<Ops>>(*context),
            0)...};
  }
};

struct DispatchTensorLoadOpSubsetInterface
    : SubsetOpInterface::ExternalModel<DispatchTensorLoadOpSubsetInterface,
                                       IREE::TensorExt::DispatchTensorLoadOp> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // Returns true if the value of a `loadOp` bufferizes to an equivalent
    // DispatchTensorStoreOp result that bufferizes inplace.
    auto loadOp = cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    auto storeOp = dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(op);
    if (!storeOp) {
      return false;
    }
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
    : SubsetOpInterface::ExternalModel<DispatchTensorStoreOpSubsetInterface,
                                       IREE::TensorExt::DispatchTensorStoreOp> {

  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // Returns true if the value of a `storeOp` bufferizes to an equivalent
    // DispatchTensorLoadOp result that bufferizes inplace.
    auto storeOp = cast<IREE::TensorExt::DispatchTensorStoreOp>(op);
    auto loadOp = dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(op);
    if (!loadOp) {
      return false;
    }
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
    : SubsetInsertionOpInterface::ExternalModel<
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
    auto loadOp = IREE::TensorExt::DispatchTensorLoadOp::create(
        builder, loc, cast<RankedTensorType>(storeOp.getValue().getType()),
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
    : SubsetOpInterface::ExternalModel<LoadFromBufferOpSubsetInterface,
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
    : SubsetOpInterface::ExternalModel<StoreToBufferOpSubsetInterface,
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
    : SubsetInsertionOpInterface::ExternalModel<
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
    auto loadOp = IREE::Codegen::LoadFromBufferOp::create(
        builder, loc, storeOp.getTensor().getType(), storeOp.getBuffer());
    return loadOp.getResult();
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    auto storeOp = cast<IREE::Codegen::StoreToBufferOp>(op);
    return {storeOp.getBuffer()};
  }
};

// Bufferization interface for CastToRaggedShapeOp. The op is a pure view: it
// reinterprets its source as a rank-(n+1) ragged buffer by attaching a
// `RaggedShapeAttr` layout, without copying data. The bufferized op therefore
// aliases its source buffer and must bufferize in place.
struct CastToRaggedShapeBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          CastToRaggedShapeBufferizationInterface,
          IREE::TensorExt::CastToRaggedShapeOp> {

  // Both operands are read: `source` for its data and `column_lengths` for
  // the per-row sizes of the resulting ragged view.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return true;
  }

  // The op produces a view and never writes to any operand.
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    return false;
  }

  // Materializing the result view does not introduce new writes either.
  bool resultBufferizesToMemoryWrite(
      Operation *op, OpResult opResult,
      const bufferization::AnalysisState &state) const {
    return false;
  }

  // A view must share storage with its source; bufferization cannot introduce
  // a copy for this op.
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const bufferization::AnalysisState &state) const {
    return true;
  }

  // The result aliases the `source` operand (same underlying buffer, new
  // layout). `column_lengths` is consumed as shape metadata and does not
  // alias the result.
  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    auto castToRaggedShapeOp = cast<IREE::TensorExt::CastToRaggedShapeOp>(op);
    if (opOperand.get() == castToRaggedShapeOp.getSource()) {
      return {{castToRaggedShapeOp.getResult(), BufferRelation::Equivalent}};
    };
    return {};
  }

  // Inverse of `getAliasingValues`: the result's only aliasing operand is
  // `source`.
  bufferization::AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const bufferization::AnalysisState &state) const {
    auto castToRaggedShapeOp = cast<IREE::TensorExt::CastToRaggedShapeOp>(op);
    if (value == castToRaggedShapeOp.getResult()) {
      return {{&castToRaggedShapeOp.getSourceMutable(),
               BufferRelation::Equivalent}};
    }
    return {};
  }

  // For the result, the memref type preserves the tensor's ragged encoding by
  // promoting it to the memref layout attribute. For `source` and
  // `column_lengths` operands, use the default identity layout so the op
  // consumes plain buffers.
  FailureOr<bufferization::BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> & /*invocationStack*/) const {
    auto castToRaggedShapeOp = cast<IREE::TensorExt::CastToRaggedShapeOp>(op);

    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return failure();
    }

    if (value == castToRaggedShapeOp.getResult()) {
      return cast<bufferization::BufferLikeType>(MemRefType::get(
          tensorType.getShape(), tensorType.getElementType(),
          /*layout=*/
          cast<MemRefLayoutAttrInterface>(tensorType.getEncoding())));
    }

    return cast<bufferization::BufferLikeType>(
        MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
  }

  // Replace the tensor-typed op with the identical op on memrefs: the source
  // and column_lengths operands become buffers and the result takes the
  // ragged-layout memref type computed by `getBufferType`. No data is copied.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto castToRaggedShapeOp = cast<IREE::TensorExt::CastToRaggedShapeOp>(op);

    FailureOr<Value> sourceBuffer =
        getBuffer(rewriter, castToRaggedShapeOp.getSource(), options, state);
    if (failed(sourceBuffer)) {
      return failure();
    }

    FailureOr<Value> columnLengths = getBuffer(
        rewriter, castToRaggedShapeOp.getColumnLengths(), options, state);
    if (failed(columnLengths)) {
      return failure();
    }

    auto resultMemRefType = bufferization::getBufferType(
        castToRaggedShapeOp.getResult(), options, state);
    if (failed(resultMemRefType)) {
      return failure();
    }
    replaceOpWithNewBufferizedOp<IREE::TensorExt::CastToRaggedShapeOp>(
        rewriter, op, resultMemRefType.value(), sourceBuffer.value(),
        castToRaggedShapeOp.getRaggedDimAttr(), columnLengths.value(),
        castToRaggedShapeOp.getNumRaggedRows(),
        castToRaggedShapeOp.getSourceDynamicDims());
    return success();
  }
};

// Bufferization interface for LinearizeRaggedDimsOp. The op is the dual view
// of `CastToRaggedShape`: it collapses the ragged dimensions of its source
// back into a single dense dimension and drops the `RaggedShapeAttr` layout.
// No data is moved, so the bufferized op aliases its source and must
// bufferize in place.
struct LinearizeRaggedDimsBufferizationInterface
    : public BufferizableOpInterface::ExternalModel<
          LinearizeRaggedDimsBufferizationInterface,
          IREE::TensorExt::LinearizeRaggedDimsOp> {

  // The source buffer is read to produce the linearized view.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const bufferization::AnalysisState &state) const {
    return true;
  }

  // The op is a view and does not write to any operand.
  bool
  bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                          const bufferization::AnalysisState &state) const {
    return false;
  }

  // Materializing the result view does not introduce new writes.
  bool resultBufferizesToMemoryWrite(
      Operation *op, OpResult opResult,
      const bufferization::AnalysisState &state) const {
    return false;
  }

  // The result must share storage with the source; no copy is permitted.
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const bufferization::AnalysisState &state) const {
    return true;
  }

  // The result aliases the `source` operand (same underlying buffer, layout
  // without ragged encoding).
  bufferization::AliasingValueList
  getAliasingValues(Operation *op, OpOperand &opOperand,
                    const bufferization::AnalysisState &state) const {
    auto linearizeOp = cast<IREE::TensorExt::LinearizeRaggedDimsOp>(op);
    if (opOperand.get() == linearizeOp.getSource()) {
      return {{linearizeOp.getResult(), BufferRelation::Equivalent}};
    };
    return {};
  }

  // Inverse of `getAliasingValues`: the result's only aliasing operand is
  // `source`.
  bufferization::AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const bufferization::AnalysisState &state) const {
    auto linearizeOp = cast<IREE::TensorExt::LinearizeRaggedDimsOp>(op);
    if (value == linearizeOp.getResult()) {
      return {{&linearizeOp.getSourceMutable(), BufferRelation::Equivalent}};
    }
    return {};
  }

  // Both source and result use the default identity layout. The result type
  // drops the ragged encoding the source carries as part of collapsing the
  // ragged dimensions into one dense dimension.
  FailureOr<bufferization::BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> & /*invocationStack*/) const {
    auto tensorType = dyn_cast<RankedTensorType>(value.getType());
    if (!tensorType) {
      return failure();
    }

    return cast<bufferization::BufferLikeType>(
        MemRefType::get(tensorType.getShape(), tensorType.getElementType()));
  }

  // Replace the tensor-typed op with the identical op on memrefs: the source
  // memref is reinterpreted as a lower-rank non-ragged memref with the
  // provided dynamic result dims. No data is copied.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto linearizeOp = cast<IREE::TensorExt::LinearizeRaggedDimsOp>(op);

    FailureOr<Value> sourceBuffer =
        getBuffer(rewriter, linearizeOp.getSource(), options, state);
    if (failed(sourceBuffer)) {
      return failure();
    }

    auto resultMemRefType =
        bufferization::getBufferType(linearizeOp.getResult(), options, state);
    if (failed(resultMemRefType)) {
      return failure();
    }

    replaceOpWithNewBufferizedOp<IREE::TensorExt::LinearizeRaggedDimsOp>(
        rewriter, op, resultMemRefType.value(), sourceBuffer.value(),
        linearizeOp.getResultDynamicDims());
    return success();
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
  IREE::VectorExt::registerIREEVectorExtBufferizationInterfaces(registry);
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::TensorExt::IREETensorExtDialect *dialect) {
        // CastToRaggedShapeOp
        IREE::TensorExt::CastToRaggedShapeOp::attachInterface<
            CastToRaggedShapeBufferizationInterface>(*ctx);
        // LinearizeRaggedDimsOp
        IREE::TensorExt::LinearizeRaggedDimsOp::attachInterface<
            LinearizeRaggedDimsBufferizationInterface>(*ctx);
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
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::Codegen::IREECodegenDialect *dialect) {
    IREE::Codegen::LoadFromBufferOp::attachInterface<
        LoadFromBufferOpInterface, LoadFromBufferOpSubsetInterface>(*ctx);
    IREE::Codegen::StoreToBufferOp::attachInterface<
        StoreToBufferOpInterface, StoreToBufferOpSubsetInterface,
        StoreToBufferOpSubsetInsertionInterface>(*ctx);
    IREE::Codegen::SwizzleHintOp::attachInterface<SwizzleHintOpInterface>(*ctx);
  });
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::LinalgExt::IREELinalgExtDialect *dialect) {
        LinalgExtOpInterfaceHelper<
#define GET_OP_LIST
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
            >::registerOpInterface(ctx);
      });
}

} // namespace mlir::iree_compiler
