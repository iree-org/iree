// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
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
using mlir::bufferization::BufferRelation;
using mlir::bufferization::eliminateEmptyTensors;
using mlir::bufferization::OneShotAnalysisState;
using mlir::bufferization::OneShotBufferizationOptions;
using mlir::bufferization::replaceOpWithBufferizedValues;
using mlir::bufferization::replaceOpWithNewBufferizedOp;

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

/// Get strides for row-major oredering of a tensor with the given `shape`.
static SmallVector<int64_t> getStridesFromShape(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return {};
  }
  SmallVector<int64_t> strides(shape.size(), ShapedType::kDynamic);
  strides.back() = 1;
  for (int i = strides.size() - 1; i > 0; --i) {
    if (ShapedType::isDynamic(shape[i])) {
      break;
    }
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

static MemRefType
getMemrefTypeForTensor(IREE::Flow::DispatchTensorType tensorType,
                       MemRefLayoutAttrInterface layout = {},
                       Attribute memorySpace = {}) {
  return MemRefType::get(tensorType.getShape(),
                         tensorType.getBoundElementType(), layout, memorySpace);
}

/// Find the memref version of the given InterfaceBindingSubspanOp. If no such
/// op exists in the same block (before the given op), create a new op.
// TODO(#12933): Because of regressions in CUDA backend, there is an
// option to keep a legacy mode of not representing the offset in the
// type. Remove once the bug is fixed.
static Value
findOrCreateSubspanBuffer(RewriterBase &rewriter,
                          IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  // Ensure that this a tensor subspan op.
  auto shapedType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
      subspanOp.getResult().getType());
  assert(shapedType && shapedType.hasRank());

  Value byteOffset = subspanOp.getByteOffset();
  MemRefLayoutAttrInterface layoutAttr = {};
  if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
    OpFoldResult elementOffset = convertByteOffsetToElementOffset(
        rewriter, subspanOp->getLoc(), subspanOp.getByteOffset(),
        shapedType.getBoundElementType());
    std::optional<int64_t> elementOffsetInt =
        getConstantIntValue(elementOffset);
    if (!elementOffsetInt) {
      elementOffsetInt = ShapedType::kDynamic;
    }
    auto tensorType = llvm::cast<RankedTensorType>(shapedType.getBoundType());
    SmallVector<int64_t> strides = getStridesFromShape(tensorType.getShape());
    layoutAttr = StridedLayoutAttr::get(rewriter.getContext(),
                                        elementOffsetInt.value(), strides);
  }
  auto memRefType = getMemrefTypeForTensor(shapedType, layoutAttr,
                                           subspanOp.getDescriptorTypeAttr());

  // Look for an existing op.
  Block *block = subspanOp->getBlock();
  for (Operation &op : *block) {
    if (&op == subspanOp.getOperation())
      break;
    auto bufferSubspanOp = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(&op);
    if (!bufferSubspanOp)
      continue;

    auto bufferMemrefType =
        llvm::dyn_cast<MemRefType>(bufferSubspanOp.getResult().getType());
    if (!bufferMemrefType)
      continue;

    if (bufferSubspanOp.getSet() != subspanOp.getSet() ||
        bufferSubspanOp.getBinding() != subspanOp.getBinding() ||
        bufferSubspanOp.getDescriptorType() != subspanOp.getDescriptorType() ||
        bufferSubspanOp.getByteOffset() != subspanOp.getByteOffset() ||
        !llvm::equal(bufferSubspanOp.getDynamicDims(),
                     subspanOp.getDynamicDims()) ||
        bufferSubspanOp.getAlignment() != subspanOp.getAlignment() ||
        memRefType != bufferMemrefType)
      continue;
    return bufferSubspanOp.getResult();
  }

  // None found, create a new op.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(subspanOp);
  // Just change the result type of the InterfaceBindingSubspanOp.
  Value buffer = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
      subspanOp->getLoc(), memRefType, subspanOp.getSet(),
      subspanOp.getBinding(), subspanOp.getDescriptorType(),
      subspanOp.getByteOffset(), subspanOp.getDynamicDims(),
      subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());
  rewriter.create<memref::AssumeAlignmentOp>(
      subspanOp->getLoc(), buffer, subspanOp.calculateAlignment().value());
  return buffer;
}

namespace {

//===----------------------------------------------------------------------===//
// IREE specific External models for BufferizableOpInterface.
//===----------------------------------------------------------------------===//

/// Check if the two tensor types (with their respective dynamic dimension
/// values) have the same shape.
static bool equalTensorShape(RankedTensorType tensorType,
                             ValueRange tensorDynSizes,
                             IREE::Flow::DispatchTensorType dispatchTensorType,
                             ValueRange dispatchTensorDynSizes) {
  return llvm::equal(tensorType.getShape(), dispatchTensorType.getShape()) &&
         tensorDynSizes.size() == dispatchTensorDynSizes.size() &&
         llvm::equal(tensorDynSizes, dispatchTensorDynSizes);
}

struct DispatchTensorLoadOpInterface
    : public BufferizableOpInterface::ExternalModel<
          DispatchTensorLoadOpInterface, IREE::Flow::DispatchTensorLoadOp> {
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    auto shapedType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
        loadOp.getSource().getType());
    assert(shapedType && "unexpected source type");
    return shapedType.getAccess() != IREE::Flow::TensorAccess::ReadOnly;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    auto tensorSubspanOp =
        loadOp.getSource()
            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    assert(tensorSubspanOp && "expected that source is a SubspanOp");
    Value source = findOrCreateSubspanBuffer(rewriter, tensorSubspanOp);

    if (equalTensorShape(loadOp.getType(), loadOp.sizes(),
                         llvm::cast<IREE::Flow::DispatchTensorType>(
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
          DispatchTensorStoreOpInterface, IREE::Flow::DispatchTensorStoreOp> {
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
                          const BufferizationOptions &options) const {
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);
    auto tensorSubspanOp =
        storeOp.getTarget()
            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    assert(tensorSubspanOp && "expected that target is a SubspanOp");
    Value target = findOrCreateSubspanBuffer(rewriter, tensorSubspanOp);

    if (!equalTensorShape(
            llvm::cast<RankedTensorType>(storeOp.getValue().getType()),
            storeOp.getSizes(),
            llvm::cast<IREE::Flow::DispatchTensorType>(
                storeOp.getTarget().getType()),
            storeOp.getTargetDims())) {
      // Writing to a part of the tensor.
      auto subviewMemRefType =
          llvm::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
              storeOp.getValue().getType().cast<ShapedType>().getShape(),
              target.getType().cast<MemRefType>(), storeOp.getMixedOffsets(),
              storeOp.getMixedSizes(), storeOp.getMixedStrides()));

      target = rewriter.create<memref::SubViewOp>(
          storeOp->getLoc(), subviewMemRefType, target,
          storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
          storeOp.getMixedStrides());
    } // else: Writing the entire tensor, no subview required.

    auto maybeBuffer =
        getBuffer(rewriter, storeOp->getOpOperand(0).get(), options);
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
} // namespace

/// Generic conversion for any LinalgExtOp on tensors.
static LogicalResult bufferizeLinalgExtOp(RewriterBase &rewriter,
                                          IREE::LinalgExt::LinalgExtOp op,
                                          const BufferizationOptions &options) {
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
      auto maybeBuffer = getBuffer(rewriter, opOperand.get(), options);
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
        options);
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
    // TODO: Revisit this for Scatter/ReverseOp. We can then get rid of
    //       `bufferizesToMemoryRead` completely.
    return !isa<IREE::LinalgExt::ScatterOp, IREE::LinalgExt::ReverseOp>(op);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeLinalgExtOp(
        rewriter, cast<IREE::LinalgExt::LinalgExtOp>(op), options);
  }
};

/// Returns the buffers of the source and destination for pack and unpack ops.
/// Returns a failure if the buffers can not be found.
template <typename OpTy>
static FailureOr<std::pair<Value, Value>>
getSourceAndDestFromPackUnPackOp(RewriterBase &rewriter, OpTy op,
                                 const BufferizationOptions &options) {
  static_assert(llvm::is_one_of<OpTy, tensor::PackOp, tensor::UnPackOp>::value);
  Value source;
  auto maybeBuffer = getBuffer(rewriter, op.getSource(), options);
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
      options);
  if (failed(resultBuffer))
    return failure();
  dest = *resultBuffer;
  return std::make_pair(source, dest);
}

static LogicalResult bufferizePackOp(RewriterBase &rewriter, tensor::PackOp op,
                                     const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto maybeSrcAndDest =
      getSourceAndDestFromPackUnPackOp(rewriter, op, options);
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
                                       tensor::UnPackOp op,
                                       const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  auto maybeSrcAndDest =
      getSourceAndDestFromPackUnPackOp(rewriter, op, options);
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
                          const BufferizationOptions &options) const {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .template Case<tensor::PackOp>(
            [&](auto pack) { return bufferizePackOp(rewriter, pack, options); })
        .template Case<tensor::UnPackOp>([&](auto unpack) {
          return bufferizeUnPackOp(rewriter, unpack, options);
        })
        .Default([](auto) { return failure(); });
  }
};

struct DispatchTensorLoadOpSubsetInterface
    : public SubsetOpInterface::ExternalModel<
          DispatchTensorLoadOpSubsetInterface,
          IREE::Flow::DispatchTensorLoadOp> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // Returns true if the value of a `loadOp` bufferizes to an equivalent
    // DispatchTensorStoreOp result that bufferizes inplace.
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(op);
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
          IREE::Flow::DispatchTensorStoreOp> {

  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // Returns true if the value of a `storeOp` bufferizes to an equivalent
    // DispatchTensorLoadOp result that bufferizes inplace.
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);
    auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(op);
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
          IREE::Flow::DispatchTensorStoreOp> {

  OpOperand &getSourceOperand(Operation *op) const {
    return op->getOpOperand(0);
  }

  OpOperand &getDestinationOperand(Operation *op) const {
    return op->getOpOperand(1);
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);
    auto loadOp = builder.create<IREE::Flow::DispatchTensorLoadOp>(
        loc, llvm::cast<RankedTensorType>(storeOp.getValue().getType()),
        storeOp.getTarget(), storeOp.getTargetDims(), storeOp.getMixedOffsets(),
        storeOp.getMixedSizes(), storeOp.getMixedStrides());
    return loadOp.getResult();
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);
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
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Flow::FlowDialect *dialect) {
        // DispatchTensorLoadOp
        IREE::Flow::DispatchTensorLoadOp::attachInterface<
            DispatchTensorLoadOpInterface>(*ctx);
        IREE::Flow::DispatchTensorLoadOp::attachInterface<
            DispatchTensorLoadOpSubsetInterface>(*ctx);

        // DispatchTensorStoreOp
        IREE::Flow::DispatchTensorStoreOp::attachInterface<
            DispatchTensorStoreOpInterface>(*ctx);
        IREE::Flow::DispatchTensorStoreOp::attachInterface<
            DispatchTensorStoreOpSubsetInterface>(*ctx);
        IREE::Flow::DispatchTensorStoreOp::attachInterface<
            DispatchTensorStoreOpSubsetInsertionInterface>(*ctx);
      });
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    IREE::LinalgExt::FftOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::FftOp>>(*ctx);
    IREE::LinalgExt::PackOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::PackOp>>(*ctx);
    IREE::LinalgExt::UnPackOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::UnPackOp>>(*ctx);
    IREE::LinalgExt::ReverseOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::ReverseOp>>(*ctx);
    IREE::LinalgExt::ScanOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::ScanOp>>(*ctx);
    IREE::LinalgExt::ScatterOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::ScatterOp>>(*ctx);
    IREE::LinalgExt::SortOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::SortOp>>(*ctx);
    IREE::LinalgExt::TopkOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::TopkOp>>(*ctx);
    IREE::LinalgExt::WinogradInputTransformOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::WinogradInputTransformOp>>(*ctx);
    IREE::LinalgExt::WinogradOutputTransformOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::WinogradOutputTransformOp>>(*ctx);
    IREE::LinalgExt::AttentionOp::attachInterface<
        LinalgExtOpInterface<IREE::LinalgExt::AttentionOp>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::PackOp::attachInterface<PackUnPackOpInterface<tensor::PackOp>>(
        *ctx);
    tensor::UnPackOp::attachInterface<PackUnPackOpInterface<tensor::UnPackOp>>(
        *ctx);
  });
}

} // namespace mlir::iree_compiler
