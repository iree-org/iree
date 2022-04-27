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
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Support/LLVM.h"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationAliasInfo;
using mlir::bufferization::BufferizationState;
using mlir::bufferization::createMemCpy;
using mlir::bufferization::DialectAnalysisState;
using mlir::bufferization::OneShotBufferizationOptions;
using mlir::bufferization::PostAnalysisStepFn;
using mlir::bufferization::replaceOpWithNewBufferizedOp;
using mlir::linalg::eliminateInitTensors;

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Flow dialect-specific bufferization state.
struct FlowBufferizationState : public DialectAnalysisState {
  DenseMap<Value, Value> subspan_to_buffer;

  /// DispatchTensorStoreOps that do not require a copy.
  DenseSet<Operation *> store_ops_without_copy;
};
}  // namespace

/// Get FlowBufferizationState.
static const FlowBufferizationState &getFlowBufferizationState(
    const AnalysisState &state) {
  Optional<const FlowBufferizationState *> maybeState =
      state.getDialectState<FlowBufferizationState>(
          IREE::Flow::FlowDialect::getDialectNamespace());
  assert(maybeState.hasValue() && "FlowBufferizationState does not exist");
  return **maybeState;
}
static FlowBufferizationState &getFlowBufferizationState(AnalysisState &state) {
  return state.getOrCreateDialectState<FlowBufferizationState>(
      IREE::Flow::FlowDialect::getDialectNamespace());
}

template <typename TensorType>
static MemRefType getMemrefTypeForTensor(TensorType tensorType,
                                         MemRefLayoutAttrInterface layout = {},
                                         Attribute memorySpace = {}) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, memorySpace);
}

static Value getSubspanBuffer(Value tensor, RewriterBase &rewriter,
                              const AnalysisState &state) {
  const FlowBufferizationState &flowState = getFlowBufferizationState(state);
  auto it = flowState.subspan_to_buffer.find(tensor);
  assert(it != flowState.subspan_to_buffer.end() && "subspan buffer not found");
  return it->getSecond();
}

namespace {

//===----------------------------------------------------------------------===//
// IREE specific External models for BufferizableOpInterface.
//===----------------------------------------------------------------------===//

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
                          BufferizationState &state) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    Value source =
        getSubspanBuffer(loadOp.source(), rewriter, state.getAnalysisState());

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
                          BufferizationState &state) const {
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);

    const AnalysisState &analysisState = state.getAnalysisState();
    Value target = getSubspanBuffer(storeOp.target(), rewriter, analysisState);
    Value subView = rewriter.create<memref::SubViewOp>(
        storeOp->getLoc(), target, storeOp.getMixedOffsets(),
        storeOp.getMixedSizes(), storeOp.getMixedStrides());
    Value srcMemref =
        *state.getBuffer(rewriter, storeOp->getOpOperand(0) /*tensor*/);

    // If everything bufferized inplace, no copy is needed. We wrote to the
    // target buffer already. The copy folds away in that case.
    if (failed(createMemCpy(rewriter, storeOp->getLoc(), srcMemref, subView,
                            state.getOptions())))
      return failure();

    rewriter.eraseOp(storeOp);
    return success();
  }
};
}  // namespace

/// Generic conversion for any LinalgExtOp on tensors.
static LogicalResult bufferizeLinalgExtOp(RewriterBase &rewriter,
                                          IREE::LinalgExt::LinalgExtOp op,
                                          BufferizationState &state) {
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
    newInputBuffers.push_back(*state.getBuffer(
        rewriter, *opOperand,
        BufferizationState::ForceInPlacability::FORCE_INPLACE));
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    SmallVector<OpOperand *> aliasingOpOperands =
        state.getAnalysisState().getAliasingOpOperand(opResult);
    assert(aliasingOpOperands.size() == 1 && "expected 1 OpOperand");
    FailureOr<Value> resultBuffer =
        state.getBuffer(rewriter, *aliasingOpOperands.front());
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
    // All operands are read.
    // TODO: Is this correct?
    return true;
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
                          BufferizationState &state) const {
    return bufferizeLinalgExtOp(rewriter,
                                cast<IREE::LinalgExt::LinalgExtOp>(op), state);
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

static LogicalResult inplaceTensorStoreOpAnalysis(
    Operation *op, AnalysisState &state, BufferizationAliasInfo &aliasInfo,
    SmallVector<Operation *> &newOps) {
  FlowBufferizationState &flowState = getFlowBufferizationState(state);
  op->walk([&](IREE::Flow::DispatchTensorStoreOp storeOp) {
    // If a store op's dest is eqivalent to a load op's source, no copy is
    // needed for the store op. All writes already happened inplace.
    if (isValueEquivalentToAnInplaceTensorLoadOp(aliasInfo, storeOp))
      flowState.store_ops_without_copy.insert(storeOp);
  });
  return success();
}

/// Try to eliminate InitTensorOps that are eventually fed into a
/// DispatchTensorStoreOp. Such InitTensorOps are replaced with matching
/// DispatchTensorLoadOps. Two conditions must be met:
///
/// * The target must be a "readwrite" tensor.
/// * All ops along the reverse SSA use-def chain from the
///   DispatchTensorStoreOp to the InitTensorOp must have bufferized in-place.
static LogicalResult storeTensorOpAnchoredInitTensorEliminationStep(
    Operation *op, AnalysisState &state, BufferizationAliasInfo &aliasInfo,
    SmallVector<Operation *> &newOps) {
  return eliminateInitTensors(
      op, state, aliasInfo,
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
      },
      newOps);
}

static LogicalResult createSubSpanBuffers(Operation *op, AnalysisState &state,
                                          BufferizationAliasInfo &aliasInfo,
                                          SmallVector<Operation *> &newOps) {
  FlowBufferizationState &flowState = getFlowBufferizationState(state);

  op->walk([&](Operation *op) {
    Value tensor;
    if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(op)) {
      tensor = storeOp.target();
    } else if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(op)) {
      tensor = loadOp.source();
    } else {
      return WalkResult::skip();
    }

    if (!flowState.subspan_to_buffer.count(tensor)) {
      IRRewriter rewriter(op->getContext());
      OpBuilder::InsertionGuard g(rewriter);
      auto subspanOp =
          tensor.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
      assert(subspanOp && "expected LoadOp/StoreOp source/target is SubspanOp");

      auto shapedType = subspanOp.getResult()
                            .getType()
                            .dyn_cast<IREE::Flow::DispatchTensorType>();
      assert(shapedType && shapedType.hasRank());

      rewriter.setInsertionPoint(subspanOp);
      // Just change the result type of the InterfaceBindingSubspanOp.
      auto memRefType = getMemrefTypeForTensor(shapedType);
      Value baseBuffer = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
          subspanOp->getLoc(), memRefType, subspanOp.set(), subspanOp.binding(),
          subspanOp.type(), subspanOp.byte_offset(), subspanOp.dynamic_dims(),
          subspanOp.alignmentAttr());
      if (subspanOp.alignment()) {
        rewriter.create<memref::AssumeAlignmentOp>(
            subspanOp->getLoc(), baseBuffer,
            subspanOp.alignment()->getZExtValue());
      }
      flowState.subspan_to_buffer[tensor] = baseBuffer;
    }

    return WalkResult::advance();
  });

  return success();
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
      });
}

void addPostAnalysisTransformations(OneShotBufferizationOptions &options) {
  options.addPostAnalysisStep(createSubSpanBuffers);
  options.addPostAnalysisStep(storeTensorOpAnchoredInitTensorEliminationStep);
  options.addPostAnalysisStep(inplaceTensorStoreOpAnalysis);
}

}  // namespace iree_compiler
}  // namespace mlir
