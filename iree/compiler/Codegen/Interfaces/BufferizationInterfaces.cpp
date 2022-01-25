// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ArithInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/SCFInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/StdInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Support/LLVM.h"

using mlir::bufferization::AnalysisBufferizationOptions;
using mlir::bufferization::AnalysisBufferizationState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationAliasInfo;
using mlir::bufferization::BufferizationState;
using mlir::bufferization::createMemCpy;
using mlir::bufferization::DialectBufferizationState;
using mlir::bufferization::PostAnalysisStep;
using mlir::bufferization::replaceOpWithNewBufferizedOp;
using mlir::linalg::comprehensive_bufferize::linalg_ext::
    InitTensorEliminationStep;

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

namespace {
/// Flow dialect-specific bufferization state.
struct FlowBufferizationState : public DialectBufferizationState {
  DenseMap<Value, Value> subspan_to_buffer;

  /// DispatchTensorStoreOps that do not require a copy.
  DenseSet<Operation *> store_ops_without_copy;
};
}  // namespace

/// Get FlowBufferizationState.
static const FlowBufferizationState &getFlowBufferizationState(
    const BufferizationState &state) {
  Optional<const FlowBufferizationState *> maybeState =
      state.getDialectState<FlowBufferizationState>(
          IREE::Flow::FlowDialect::getDialectNamespace());
  assert(maybeState.hasValue() && "FlowBufferizationState does not exist");
  return **maybeState;
}

/// Get or create FlowBufferizationState.
static FlowBufferizationState &getFlowBufferizationState(
    BufferizationState &state) {
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
                              const BufferizationState &state) {
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
                  const BufferizationState &state) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    auto shapedType =
        loadOp.source().getType().dyn_cast<IREE::Flow::DispatchTensorType>();
    assert(shapedType && "unexpected source type");
    return shapedType.getAccess() != IREE::Flow::TensorAccess::ReadOnly;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    Value source = getSubspanBuffer(loadOp.source(), rewriter, state);

    // Bufferize to subview.
    replaceOpWithNewBufferizedOp<memref::SubViewOp>(
        rewriter, op, source, loadOp.getMixedOffsets(), loadOp.getMixedSizes(),
        loadOp.getMixedStrides());

    return success();
  }
};

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

struct InplaceTensorStoreOpAnalysis : public PostAnalysisStep {
  LogicalResult run(Operation *op, BufferizationState &state,
                    BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override {
    auto &flowState = getFlowBufferizationState(state);
    op->walk([&](IREE::Flow::DispatchTensorStoreOp storeOp) {
      // If a store op's dest is eqivalent to a load op's source, no copy is
      // needed for the store op. All writes already happened inplace.
      if (isValueEquivalentToAnInplaceTensorLoadOp(aliasInfo, storeOp))
        flowState.store_ops_without_copy.insert(storeOp);
    });
    return success();
  }
};

struct DispatchTensorStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<
          DispatchTensorStoreOpInterface, IREE::Flow::DispatchTensorStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);

    Value target = getSubspanBuffer(storeOp.target(), rewriter, state);
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

//===----------------------------------------------------------------------===//
// IREE specific post analysis transformations.
//===----------------------------------------------------------------------===//

/// Try to eliminate InitTensorOps that are eventually fed into a
/// DispatchTensorStoreOp. Such InitTensorOps are replaced with matching
/// DispatchTensorLoadOps. Two conditions must be met:
///
/// * The target must be a "readwrite" tensor.
/// * All ops along the reverse SSA use-def chain from the
///   DispatchTensorStoreOp to the InitTensorOp must have bufferized in-place.
struct StoreTensorOpAnchoredInitTensorEliminationStep
    : public InitTensorEliminationStep {
  LogicalResult run(Operation *op, BufferizationState &state,
                    BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override {
    return eliminateInitTensors(
        op, state, aliasInfo,
        /*anchorMatchFunc=*/
        [&](OpOperand &operand) {
          return isa<IREE::Flow::DispatchTensorStoreOp>(operand.getOwner());
        },
        /*rewriteFunc=*/
        [](OpBuilder &b, Location loc, OpOperand &operand) {
          auto storeOp =
              cast<IREE::Flow::DispatchTensorStoreOp>(operand.getOwner());
          auto loadOp = b.create<IREE::Flow::DispatchTensorLoadOp>(
              loc, storeOp.value().getType().cast<RankedTensorType>(),
              storeOp.target(), storeOp.target_dims(),
              storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
              storeOp.getMixedStrides());
          return loadOp.result();
        },
        newOps);
  }
};

struct CreateSubSpanBuffers : public PostAnalysisStep {
  LogicalResult run(Operation *op, BufferizationState &state,
                    BufferizationAliasInfo &aliasInfo,
                    SmallVector<Operation *> &newOps) override {
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
        assert(subspanOp &&
               "expected LoadOp/StoreOp source/target is SubspanOp");

        auto shapedType = subspanOp.getResult()
                              .getType()
                              .dyn_cast<IREE::Flow::DispatchTensorType>();
        assert(shapedType && shapedType.hasRank());

        rewriter.setInsertionPoint(subspanOp);
        // Just change the result type of the InterfaceBindingSubspanOp.
        auto memRefType = getMemrefTypeForTensor(shapedType);
        Value baseBuffer =
            rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
                subspanOp->getLoc(), memRefType, subspanOp.set(),
                subspanOp.binding(), subspanOp.type(), subspanOp.byte_offset(),
                subspanOp.dynamic_dims(), subspanOp.alignmentAttr());
        flowState.subspan_to_buffer[tensor] = baseBuffer;
      }

      return WalkResult::advance();
    });

    return success();
  }
};
}  // namespace

void registerBufferizationInterfaces(DialectRegistry &registry) {
  linalg::comprehensive_bufferize::affine_ext::
      registerBufferizableOpInterfaceExternalModels(registry);
  linalg::comprehensive_bufferize::arith_ext::
      registerBufferizableOpInterfaceExternalModels(registry);
  linalg::comprehensive_bufferize::linalg_ext::
      registerBufferizableOpInterfaceExternalModels(registry);
  linalg::comprehensive_bufferize::scf_ext::
      registerBufferizableOpInterfaceExternalModels(registry);
  linalg::comprehensive_bufferize::std_ext::
      registerModuleBufferizationExternalModels(registry);
  linalg::comprehensive_bufferize::std_ext::
      registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::comprehensive_bufferize::vector_ext::
      registerBufferizableOpInterfaceExternalModels(registry);

  // Register IREE operations.
  registry.addOpInterface<IREE::Flow::DispatchTensorLoadOp,
                          DispatchTensorLoadOpInterface>();
  registry.addOpInterface<IREE::Flow::DispatchTensorStoreOp,
                          DispatchTensorStoreOpInterface>();
}

void addPostAnalysisTransformations(AnalysisBufferizationOptions &options) {
  options.addPostAnalysisStep<CreateSubSpanBuffers>();
  options.addPostAnalysisStep<StoreTensorOpAnchoredInitTensorEliminationStep>();
  options.addPostAnalysisStep<InplaceTensorStoreOpAnalysis>();
}

}  // namespace iree_compiler
}  // namespace mlir
