// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREEComprehensiveBufferizePass.cpp.cpp - -------------------------===//
//
// Wrapper pass to use MLIRs ComprehensiveBufferization pass.
//
//===----------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/BufferizationAnalysis.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/TensorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/VectorInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Pass that interfaces with ComprehensiveBufferization in core.
//===----------------------------------------------------------------------===//

template <typename TensorType>
static MemRefType getMemrefTypeForTensor(TensorType tensorType,
                                         MemRefLayoutAttrInterface layout = {},
                                         Attribute memorySpace = {}) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, memorySpace);
}

using linalg::comprehensive_bufferize::BufferizableOpInterface;
using linalg::comprehensive_bufferize::BufferizationAliasInfo;
using linalg::comprehensive_bufferize::BufferizationState;

Value getSubspanBuffer(Value tensor, OpBuilder &b, BufferizationState &state) {
  if (!state.isMapped(tensor)) {
    OpBuilder::InsertionGuard g(b);
    auto subspanOp =
        tensor.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    assert(subspanOp && "expected LoadOp/StoreOp source/target is SubspanOp");

    auto shapedType = subspanOp.getResult()
                          .getType()
                          .dyn_cast<IREE::Flow::DispatchTensorType>();
    assert(shapedType && shapedType.hasRank());

    b.setInsertionPoint(subspanOp);
    // Just change the result type of the InterfaceBindingSubspanOp.
    auto memRefType = getMemrefTypeForTensor(shapedType);
    auto baseBuffer = b.create<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp->getLoc(), memRefType, subspanOp.binding(),
        subspanOp.byte_offset(), subspanOp.byte_length(),
        subspanOp.dynamic_dims(), subspanOp.alignmentAttr());
    state.mapValue(subspanOp, baseBuffer);
    state.aliasInfo.createAliasInfoEntry(subspanOp.result());
  }

  return state.lookupValue(tensor);
}

namespace {

struct DispatchTensorLoadOpInterface
    : public BufferizableOpInterface::ExternalModel<
          DispatchTensorLoadOpInterface, IREE::Flow::DispatchTensorLoadOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {};
  }

  bool isWritable(Operation *op, Value value) const {
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    auto shapedType =
        loadOp.source().getType().dyn_cast<IREE::Flow::DispatchTensorType>();
    assert(shapedType && "unexpected source type");
    return shapedType.getAccess() != IREE::Flow::TensorAccess::ReadOnly;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    auto loadOp = cast<IREE::Flow::DispatchTensorLoadOp>(op);
    Value source = getSubspanBuffer(loadOp.source(), b, state);

    // Bufferize to subview.
    Value subView = b.create<memref::SubViewOp>(
        loadOp->getLoc(), source, loadOp.getMixedOffsets(),
        loadOp.getMixedSizes(), loadOp.getMixedStrides());
    state.mapBuffer(loadOp.result(), subView);

    return success();
  }
};

/// Return true if the value of a `storeOp` bufferizes to an equivalent
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

struct DispatchTensorStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<
          DispatchTensorStoreOpInterface, IREE::Flow::DispatchTensorStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    auto storeOp = cast<IREE::Flow::DispatchTensorStoreOp>(op);

    // If everything bufferized inplace, no copy is needed. We wrote to the
    // target buffer already.
    if (!isValueEquivalentToAnInplaceTensorLoadOp(state.aliasInfo, storeOp)) {
      Value target = getSubspanBuffer(storeOp.target(), b, state);
      Value subView = b.create<memref::SubViewOp>(
          storeOp->getLoc(), target, storeOp.getMixedOffsets(),
          storeOp.getMixedSizes(), storeOp.getMixedStrides());
      Value srcMemref = state.lookupBuffer(storeOp.value());
      state.allocationFns.memCpyFn(b, storeOp->getLoc(), srcMemref, subView);
    }

    state.markOpObsolete(storeOp);
    return success();
  }
};

using mlir::linalg::comprehensive_bufferize::linalg_ext::
    InitTensorEliminationStep;

/// Try to eliminate InitTensorOps that are eventually fed into a
/// DispatchTensorStoreOp. Such InitTensorOps are replaced with matching
/// DispatchTensorLoadOps. Two conditions must be met:
///
/// * The target must be a "readwrite" tensor.
/// * All ops along the reverse SSA use-def chain from the
///   DispatchTensorStoreOp to the InitTensorOp must have bufferized in-place.
struct StoreTensorOpAnchoredInitTensorEliminationStep
    : public InitTensorEliminationStep {
  LogicalResult run(FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
                    DominanceInfo &domInfo,
                    SmallVector<Operation *> &newOps) override {
    return eliminateInitTensors(
        funcOp, aliasInfo, domInfo,
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

/// Pass to convert from tensor based ops to memref based ops.
class IREEComprehensiveBufferizePass
    : public IREEComprehensiveBufferizeBase<IREEComprehensiveBufferizePass> {
 public:
  explicit IREEComprehensiveBufferizePass(
      std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
          allocationFn)
      : allocationFn(std::move(allocationFn)) {}

  IREEComprehensiveBufferizePass(const IREEComprehensiveBufferizePass &other) {
    llvm_unreachable("pass cannot be copied");
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, IREE::Util::UtilDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, StandardOpsDialect, tensor::TensorDialect,
                    vector::VectorDialect, AffineDialect,
                    IREE::Flow::FlowDialect>();

    // TODO: Find a better place to register external models.
    // Registers operations of other dialects.
    linalg::comprehensive_bufferize::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::linalg_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::tensor_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    linalg::comprehensive_bufferize::vector_ext::
        registerBufferizableOpInterfaceExternalModels(registry);

    // Register IREE operations.
    registry.addOpInterface<IREE::Flow::DispatchTensorLoadOp,
                            DispatchTensorLoadOpInterface>();
    registry.addOpInterface<IREE::Flow::DispatchTensorStoreOp,
                            DispatchTensorStoreOpInterface>();
  }

  void runOnOperation() override;

 private:
  std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
      allocationFn;
};
}  // namespace

static bool isaTensor(Type t) { return t.isa<TensorType>(); };

/// Run comprehensive bufferize.
void IREEComprehensiveBufferizePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  linalg::comprehensive_bufferize::BufferizationOptions options;
  options.testAnalysisOnly = false;

  // Enable InitTensorOp elimination.
  options.addPostAnalysisStep<StoreTensorOpAnchoredInitTensorEliminationStep>();

  // TODO: Use allocationFn.

  if (failed(runComprehensiveBufferize(moduleOp, options))) signalPassFailure();
}

// TODO: pass this to comprehensive bufferize.
static Value defaultAllocationFn(OpBuilder &builder, Location loc,
                                 ArrayRef<int64_t> staticShape,
                                 Type elementType,
                                 ArrayRef<Value> dynamicSizes) {
  auto allocationType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocOp>(loc, allocationType, dynamicSizes);
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
        allocationFns) {
  return std::make_unique<IREEComprehensiveBufferizePass>(
      std::move(allocationFns));
}

void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    std::unique_ptr<linalg::comprehensive_bufferize::AllocationCallbacks>
        allocationFns) {
  passManager.addPass(
      createIREEComprehensiveBufferizePass(std::move(allocationFns)));
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());
}

}  // namespace iree_compiler
}  // namespace mlir
