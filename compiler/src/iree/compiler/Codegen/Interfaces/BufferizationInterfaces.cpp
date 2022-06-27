// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/AllocTensorElimination.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Support/LLVM.h"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationAliasInfo;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::DialectAnalysisState;
using mlir::bufferization::eliminateAllocTensors;
using mlir::bufferization::OneShotBufferizationOptions;
using mlir::bufferization::replaceOpWithBufferizedValues;
using mlir::bufferization::replaceOpWithNewBufferizedOp;

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

template <typename TensorType>
static MemRefType getMemrefTypeForTensor(TensorType tensorType,
                                         MemRefLayoutAttrInterface layout = {},
                                         Attribute memorySpace = {}) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, memorySpace);
}

/// Find the memref version of the given InterfaceBindingSubspanOp. If no such
/// op exists in the same block (before the given op), create a new op.
static Value findOrCreateSubspanBuffer(
    OpBuilder &b, IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  // Ensure that this a tensor subspan op.
  auto shapedType = subspanOp.getResult()
                        .getType()
                        .dyn_cast<IREE::Flow::DispatchTensorType>();
  assert(shapedType && shapedType.hasRank());

  // Look for an existing op.
  Block *block = subspanOp->getBlock();
  for (Operation &op : *block) {
    if (&op == subspanOp.getOperation()) break;
    auto bufferSubspanOp = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(&op);
    if (!bufferSubspanOp) continue;
    if (bufferSubspanOp.set() != subspanOp.set() ||
        bufferSubspanOp.binding() != subspanOp.binding() ||
        bufferSubspanOp.type() != subspanOp.type() ||
        bufferSubspanOp.byte_offset() != subspanOp.byte_offset() ||
        !llvm::equal(bufferSubspanOp.dynamic_dims(),
                     subspanOp.dynamic_dims()) ||
        bufferSubspanOp.alignment() != subspanOp.alignment())
      continue;
    return bufferSubspanOp.getResult();
  }

  // None found, create a new op.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(subspanOp);
  // Just change the result type of the InterfaceBindingSubspanOp.
  auto memRefType = getMemrefTypeForTensor(shapedType);
  Value buffer = b.create<IREE::HAL::InterfaceBindingSubspanOp>(
      subspanOp->getLoc(), memRefType, subspanOp.set(), subspanOp.binding(),
      subspanOp.type(), subspanOp.byte_offset(), subspanOp.dynamic_dims(),
      subspanOp.alignmentAttr());
  if (subspanOp.alignment()) {
    b.create<memref::AssumeAlignmentOp>(subspanOp->getLoc(), buffer,
                                        subspanOp.alignment()->getZExtValue());
  }
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
    auto shapedType =
        loadOp.source().getType().dyn_cast<IREE::Flow::DispatchTensorType>();
    assert(shapedType && "unexpected source type");
    return shapedType.getAccess() != IREE::Flow::TensorAccess::ReadOnly;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    auto tensorSubspanOp =
        loadOp.source().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    assert(tensorSubspanOp && "expected that source is a SubspanOp");
    Value source = findOrCreateSubspanBuffer(rewriter, tensorSubspanOp);

    if (equalTensorShape(
            loadOp.getType(), loadOp.sizes(),
            loadOp.source().getType().cast<IREE::Flow::DispatchTensorType>(),
            loadOp.source_dims())) {
      // The entire tensor is loaded.
      replaceOpWithBufferizedValues(rewriter, op, source);
      return success();
    }

    // Bufferize to subview.
    auto subviewMemRefType = memref::SubViewOp::inferRankReducedResultType(
        loadOp.getType().getRank(), source.getType().cast<MemRefType>(),
        loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
        loadOp.getMixedStrides());
    replaceOpWithNewBufferizedOp<memref::SubViewOp>(
        rewriter, op, subviewMemRefType.cast<MemRefType>(), source,
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

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);
    auto tensorSubspanOp =
        storeOp.target().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    assert(tensorSubspanOp && "expected that target is a SubspanOp");
    Value target = findOrCreateSubspanBuffer(rewriter, tensorSubspanOp);

    if (!equalTensorShape(
            storeOp.value().getType().cast<RankedTensorType>(), storeOp.sizes(),
            storeOp.target().getType().cast<IREE::Flow::DispatchTensorType>(),
            storeOp.target_dims())) {
      // Writing to a part of the tensor.
      auto subviewMemRefType =
          memref::SubViewOp::inferRankReducedResultType(
              storeOp.value().getType().cast<ShapedType>().getRank(),
              target.getType().cast<MemRefType>(), storeOp.getMixedOffsets(),
              storeOp.getMixedSizes(), storeOp.getMixedStrides())
              .cast<MemRefType>();

      target = rewriter.create<memref::SubViewOp>(
          storeOp->getLoc(), subviewMemRefType, target,
          storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
          storeOp.getMixedStrides());
    }  // else: Writing the entire tensor, no subview required.

    Value srcMemref =
        getBuffer(rewriter, storeOp->getOpOperand(0).get(), options);

    // If everything bufferized inplace, no copy is needed. We wrote to the
    // target buffer already. The copy folds away in that case.
    if (failed(options.createMemCpy(rewriter, storeOp->getLoc(), srcMemref,
                                    target)))
      return failure();

    rewriter.eraseOp(storeOp);
    return success();
  }
};
}  // namespace

/// Generic conversion for any LinalgExtOp on tensors.
static LogicalResult bufferizeLinalgExtOp(RewriterBase &rewriter,
                                          IREE::LinalgExt::LinalgExtOp op,
                                          const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasBufferSemantics()) return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    // Input operands are never written to.
    newInputBuffers.push_back(getBuffer(rewriter, opOperand->get(), options));
  }

  // New output operands for the cloned op.
  AnalysisState analysisState(options);
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    SmallVector<OpOperand *> aliasingOpOperands =
        analysisState.getAliasingOpOperand(opResult);
    assert(aliasingOpOperands.size() == 1 && "expected 1 OpOperand");
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, aliasingOpOperands.front()->get(), options);
    if (failed(resultBuffer)) return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  auto newOp = cast<IREE::LinalgExt::LinalgExtOp>(op.cloneWithoutRegions(
      rewriter, op.getLoc(), /*resultTypes=*/TypeRange{}, newOperands));
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
    : public BufferizableOpInterface::ExternalModel<LinalgExtOpInterface<OpTy>,
                                                    OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // TODO: Implement payloadUsesValueFromOperand for individual ops. There
    // are a limited number of LinalgExt ops, so we hardcode them here. We don't
    // expect to add more LinalgExt ops.
    auto linalgExtOp = cast<IREE::LinalgExt::LinalgExtOp>(op);
    if (linalgExtOp.isInputTensor(&opOperand)) return true;
    return !isa<IREE::LinalgExt::ScatterOp, IREE::LinalgExt::ReverseOp>(op);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it has an aliasing OpResult.
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    return !bufferizableOp.getAliasingOpResult(opOperand, state).empty();
  }

  SmallVector<OpOperand *> getAliasingOpOperand(
      Operation *op, OpResult opResult, const AnalysisState &state) const {
    auto linalgExtOp = cast<IREE::LinalgExt::LinalgExtOp>(op);

    // The i-th OpResult may alias with the i-th "out" tensor.
    return {linalgExtOp.getOutputOperand(opResult.getResultNumber())};
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto linalgExtOp = cast<IREE::LinalgExt::LinalgExtOp>(op);

    // The i-th "out" tensor may alias with the i-th OpResult.
    if (linalgExtOp.isOutputTensor(&opOperand))
      return {linalgExtOp.getTiedOpResult(&opOperand)};
    return {};
  }

  bufferization::BufferRelation bufferRelation(
      Operation *op, OpResult opResult, const AnalysisState &state) const {
    return bufferization::BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeLinalgExtOp(
        rewriter, cast<IREE::LinalgExt::LinalgExtOp>(op), options);
  }
};

//===----------------------------------------------------------------------===//
// IREE specific post analysis transformations.
//===----------------------------------------------------------------------===//

/// Returns true if the value of a `storeOp` bufferizes to an equivalent
/// DispatchTensorLoadOp result that bufferizes inplace.
static bool isValueEquivalentToAnInplaceTensorLoadOp(
    const BufferizationAliasInfo &aliasInfo,
    IREE::Flow::DispatchTensorStoreOp storeOp) {
  bool foundOp = false;
  aliasInfo.applyOnEquivalenceClass(storeOp.value(), [&](Value value) {
    auto loadOp = value.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    // TODO: Assert that offsets, sizes and strides are the same.
    if (loadOp &&
        aliasInfo.areEquivalentBufferizedValues(loadOp.result(),
                                                storeOp.value()) &&
        loadOp.source() == storeOp.target()) {
      foundOp = true;
    }
  });

  return foundOp;
}

/// Try to eliminate InitTensorOps that are eventually fed into a
/// DispatchTensorStoreOp. Such InitTensorOps are replaced with matching
/// DispatchTensorLoadOps. Two conditions must be met:
///
/// * The target must be a "readwrite" tensor.
/// * All ops along the reverse SSA use-def chain from the
///   DispatchTensorStoreOp to the InitTensorOp must have bufferized in-place.
LogicalResult storeTensorOpAnchoredInitTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, AnalysisState &state) {
  return eliminateAllocTensors(
      rewriter, op, state,
      /*anchorMatchFunc=*/
      [&](OpOperand &operand, SmallVector<Value> &) {
        return isa<IREE::Flow::DispatchTensorStoreOp>(operand.getOwner());
      },
      /*rewriteFunc=*/
      [](OpBuilder &b, Location loc, OpOperand &operand) {
        auto storeOp =
            cast<IREE::Flow::DispatchTensorStoreOp>(operand.getOwner());
        auto loadOp = b.create<IREE::Flow::DispatchTensorLoadOp>(
            loc, storeOp.value().getType().cast<RankedTensorType>(),
            storeOp.target(), storeOp.target_dims(), storeOp.getMixedOffsets(),
            storeOp.getMixedSizes(), storeOp.getMixedStrides());
        return loadOp.result();
      });
}

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
        IREE::Flow::DispatchTensorLoadOp::attachInterface<
            DispatchTensorLoadOpInterface>(*ctx);
        IREE::Flow::DispatchTensorStoreOp::attachInterface<
            DispatchTensorStoreOpInterface>(*ctx);
      });
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::LinalgExt::IREELinalgExtDialect *dialect) {
        IREE::LinalgExt::ReverseOp::attachInterface<
            LinalgExtOpInterface<IREE::LinalgExt::ReverseOp>>(*ctx);
        IREE::LinalgExt::FftOp::attachInterface<
            LinalgExtOpInterface<IREE::LinalgExt::FftOp>>(*ctx);
        IREE::LinalgExt::SortOp::attachInterface<
            LinalgExtOpInterface<IREE::LinalgExt::SortOp>>(*ctx);
        IREE::LinalgExt::ScatterOp::attachInterface<
            LinalgExtOpInterface<IREE::LinalgExt::ScatterOp>>(*ctx);
        IREE::LinalgExt::ScanOp::attachInterface<
            LinalgExtOpInterface<IREE::LinalgExt::ScanOp>>(*ctx);
        IREE::LinalgExt::TopkOp::attachInterface<
            LinalgExtOpInterface<IREE::LinalgExt::TopkOp>>(*ctx);
      });
}

}  // namespace iree_compiler
}  // namespace mlir
