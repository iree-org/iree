// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- LinalgBufferizePass.cpp.cpp - Pass to bufferize Linalg on tensors --===//
//
// Pass to convert from Linalg ops on tensors to Linalg ops on buffers.
// This just inserts AllocOp to address space 0 that can be later hoisted,
// promoted and generally rewritten to the desired backend.
//
// TODO(nicolasvasilache): the implementation of this pass is unnecessarily
// convoluted due to asymmetries arising from tie_shape weirdness. Revisit once
// this abstraction is replaced.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

namespace mlir {
namespace iree_compiler {

// Transfer all `dim` ops on `tensor` to `memref`.
static void transferShapeOpsToMemref(OpBuilder &b, Value tensor, Value memref,
                                     BlockAndValueMapping &bvm) {
  for (OpOperand &opOperand : llvm::make_early_inc_range(tensor.getUses())) {
    if (isa<DimOp>(opOperand.getOwner())) {
      opOperand.set(memref);
      continue;
    }
    if (auto flowTieShapeOp =
            dyn_cast<IREE::Flow::DispatchTieShapeOp>(opOperand.getOwner())) {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPoint(flowTieShapeOp);
      auto tieShapeOp =
          b.create<Shape::TieShapeOp>(flowTieShapeOp.getLoc(), memref.getType(),
                                      memref, flowTieShapeOp.shape());
      bvm.map(flowTieShapeOp.getResult(), tieShapeOp.getResult());
      continue;
    }
  }
}

static Value maybeConvertToIndex(Location loc, Value val, OpBuilder &b) {
  if (val.getType().isIndex()) {
    return val;
  }
  return b.create<IndexCastOp>(loc, val, b.getIndexType());
}

// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
// Allocate the output buffers for the bufferized Linalg op to write into.
// If the tensor is an init tensor, we additionally copy the original value into
// the newly allocated buffer.
static LogicalResult allocateBuffersForResults(
    OpBuilder &b, Location loc, WorkgroupMemoryAllocationFn allocationFn,
    linalg::LinalgOp op, SmallVectorImpl<Value> &resultBuffers,
    BlockAndValueMapping &bvm) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  assert(op.getNumOutputs() == op->getNumResults());
  for (auto en : llvm::enumerate(op->getResultTypes())) {
    size_t resultIndex = en.index();
    Value outTensor = op.getOutput(resultIndex);

    // If output tensor was produced by a LinalgOp, just reuse the buffer.
    // TODO(nicolasvasilache): this may be too brutal and we may prefer to leave
    // this decision to a copy + alloc removal pass.
    if (outTensor.getDefiningOp<linalg::LinalgOp>()) {
      resultBuffers.push_back(bvm.lookup(outTensor));
      continue;
    }

    // If resultTensor already has a buffer, just use that.
    Value resultTensor = op->getResult(en.index());
    Value alloc = bvm.lookupOrNull(resultTensor);
    if (!alloc) {
      Type resultType = en.value();
      auto tensorType = resultType.dyn_cast<RankedTensorType>();
      auto tensorShape = tensorType.getShape();
      auto memrefType =
          MemRefType::get(tensorShape, tensorType.getElementType());
      SmallVector<Value, 4> dynOperands;
      for (auto dim : llvm::enumerate(tensorShape)) {
        Value dimTensor = bvm.lookupOrNull(outTensor);
        if (!dimTensor) dimTensor = outTensor;
        if (dim.value() == TensorType::kDynamicSize) {
          dynOperands.push_back(b.create<DimOp>(loc, dimTensor, dim.index()));
        }
      }
      alloc = allocationFn(b, loc, dynOperands, memrefType);
      bvm.map(resultTensor, alloc);
    }
    resultBuffers.push_back(alloc);

    // Additionally, if the output buffer is used, clone its value for now.
    if (op.payloadUsesValueFromOutputOperandIndex(resultIndex))
      b.create<linalg::CopyOp>(loc, bvm.lookup(outTensor), alloc);
  }
  for (auto it : llvm::zip(op->getResults(), resultBuffers)) {
    transferShapeOpsToMemref(b, std::get<0>(it), std::get<1>(it), bvm);
  }
  return success();
}

// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
static void finalizeBufferAllocation(OpBuilder &b, linalg::LinalgOp op,
                                     ValueRange inputs, ValueRange outputs,
                                     BlockAndValueMapping &bvm) {
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto otherOperands = op.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  Location loc = op.getLoc();
  op.clone(b, loc, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  for (auto result : enumerate(op.getOperation()->getResults())) {
    Value resultValue = result.value();
    Value resultBuffer = bvm.lookup(resultValue);
    if (resultBuffer != outputs[result.index()]) {
      b.create<linalg::CopyOp>(loc, outputs[result.index()], resultBuffer);
    }
  }
}

//===----------------------------------------------------------------------===//
// Bufferization helper functions using BlockAndValueMapping.
//===----------------------------------------------------------------------===//

/// Generic conversion pattern that matches any linalg::LinalgOp. This avoids
/// template instantiating one pattern for each linalg::LinalgOp.
LogicalResult convertAnyLinalgOp(OpBuilder &b,
                                 WorkgroupMemoryAllocationFn allocationFn,
                                 linalg::LinalgOp op,
                                 BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  SmallVector<Value, 2> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (Value v : op.getInputs()) {
    newInputBuffers.push_back(bvm.lookup(v));
  }
  SmallVector<Value, 2> newOutputBuffers;
  if (failed(allocateBuffersForResults(b, loc, allocationFn, op,
                                       newOutputBuffers, bvm))) {
    assert(false);
  }

  // Delegate to the linalg generic pattern.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation())) {
    finalizeBufferAllocation(b, genericOp, newInputBuffers, newOutputBuffers,
                             bvm);
    return success();
  }

  finalizeBufferAllocation(b, op, newInputBuffers, newOutputBuffers, bvm);
  return success();
}

static LogicalResult convertTransferOp(OpBuilder &b,
                                       WorkgroupMemoryAllocationFn allocationFn,
                                       VectorTransferOpInterface op,
                                       BlockAndValueMapping &bvm) {
  if (op.getShapedType().isa<MemRefType>()) return failure();
  assert(op->getNumResults() == 1);
  Value outputTensor = op->getResult(0);
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value newInputBuffer = bvm.lookup(op.source());
  if (auto tensorType =
          op->getResult(0).getType().dyn_cast<RankedTensorType>()) {
    // If the op return a Tensor allocate a buffer for the returned value.
    auto tensorShape = tensorType.getShape();
    auto memrefType = MemRefType::get(tensorShape, tensorType.getElementType());
    SmallVector<Value, 4> dynOperands;
    for (size_t idx : llvm::seq(size_t(0), tensorShape.size())) {
      if (tensorType.isDynamicDim(idx)) {
        Value tensor = bvm.lookupOrNull(outputTensor);
        if (!tensor) tensor = outputTensor;
        dynOperands.push_back(b.create<DimOp>(loc, tensor, idx));
      }
    }
    auto alloc = allocationFn(b, loc, dynOperands, memrefType);
    bvm.map(op->getResult(0), alloc);
    transferShapeOpsToMemref(b, op->getResult(0), alloc, bvm);
  }

  // Replace the tensor operand.
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op.getOperation())) {
    readOp.sourceMutable().assign(newInputBuffer);
  } else {
    auto writeOp = cast<vector::TransferWriteOp>(op.getOperation());
    // Create a new transfer_write on buffer that doesn't have a return value.
    // Leave the previous transfer_write to dead code as it still has uses at
    // this point.
    b.create<vector::TransferWriteOp>(
        loc, writeOp.vector(), newInputBuffer, writeOp.indices(),
        writeOp.permutation_map(),
        writeOp.masked() ? *writeOp.masked() : ArrayAttr());
  }
  return success();
}

// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(llvm::map_range(
      attr.cast<ArrayAttr>(),
      [](Attribute a) -> int64_t { return a.cast<IntegerAttr>().getInt(); }));
}

static MemRefType getMemrefTypeForTensor(
    RankedTensorType tensorType, ArrayRef<AffineMap> affineMapComposition = {},
    unsigned memorySpace = 0) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         affineMapComposition, memorySpace);
}

// TODO(nicolasvasilache): this will go away once integrated at the flow level.
SymbolRefAttr getBinding(Operation *op) {
  return llvm::TypeSwitch<Operation *, SymbolRefAttr>(op)
      .Case<IREE::HAL::InterfaceLoadTensorOp, IREE::HAL::InterfaceStoreTensorOp,
            IREE::HAL::InterfaceLoadTensorTileOp,
            IREE::HAL::InterfaceStoreTensorTileOp>(
          [](auto op) { return op.binding(); })
      .Default([](auto op) {
        llvm_unreachable("Expected op with binding");
        return SymbolRefAttr();
      });
}

// Create the placeholder op for the backing buffer.
// TODO(nicolasvasilache): evolve to target flow ops directly with additional
// shape annotation.
IREE::PlaceholderOp createPlaceholderOp(OpBuilder &b, Location loc,
                                        Operation *op, Value rankedTensor,
                                        IREE::HAL::InterfaceBindingOp bindingOp,
                                        bool typeErase = false) {
  assert(bindingOp);
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(
      &op->getParentOfType<FuncOp>().getBlocks().front());

  // Get the corresponding memref type from the tensor type.
  auto tensorType = rankedTensor.getType().cast<RankedTensorType>();
  auto bufferType = getMemrefTypeForTensor(tensorType);

  if (typeErase) {
    // HAL tensor tile ops does not have enough info to reconstruct the original
    // buffer type. So we just type erase it for now.
    // TODO(nicolasvasilache): evolve to target flow ops directly with
    // additional shape annotation so we do not need to type erase.
    bufferType = MemRefType::get(
        SmallVector<int64_t>(bufferType.getRank(), ShapedType::kDynamicSize),
        bufferType.getElementType(), bufferType.getAffineMaps(),
        bufferType.getMemorySpace());
  }

  // Create the placeholder op for the backing buffer. Make sure shape
  // annotation is carried over if exists.
  auto phOp =
      b.create<IREE::PlaceholderOp>(loc, bufferType, "interface buffer");
  phOp->setAttr(getBindingAttrName(), getBinding(op));
  StringRef attrName = getOperandResultNumAttrName();
  if (auto operandResultNumAttr = op->getAttr(attrName)) {
    phOp->setAttr(attrName, operandResultNumAttr);
  }

  return phOp;
}

// TODO(nicolasvasilache): evolve into flow.interface.load.tensor when the
// abstraction to get a raw void* / memref<?xi8> + offset + shape exists. Then
// we can immediately emit a view op.
// TODO(nicolasvasilache): canonicalizations of the view op.
LogicalResult convertInterfaceLoadTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceLoadTensorOp loadOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loadOp);

  // TODO(nicolasvasilache): view + subview
  if (!matchPattern(loadOp.offset(), m_Zero())) {
    return loadOp.emitError("unhandled non-zero offset");
  }

  auto phOp = createPlaceholderOp(b, loadOp->getLoc(), loadOp, loadOp.result(),
                                  loadOp.queryBindingOp());
  Value buffer = phOp.getResult();
  bvm.map(loadOp.result(), buffer);
  transferShapeOpsToMemref(b, loadOp.result(), buffer, bvm);
  return success();
}

// TODO(nicolasvasilache): evolve into flow.interface.load.tensor.tile when the
// abstraction exists. This will turn into some subview chain from the base
// view.
LogicalResult convertInterfaceLoadTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceLoadTensorTileOp loadOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loadOp);

  // TODO(nicolasvasilache): view + subview
  if (!matchPattern(loadOp.base_offset(), m_Zero())) {
    return loadOp.emitError("unhandled non-zero offset");
  }

  auto phOp = createPlaceholderOp(b, loadOp->getLoc(), loadOp, loadOp.result(),
                                  loadOp.queryBindingOp(), /*typeErase=*/true);
  Value buffer = phOp.getResult();
  Value subview =
      b.create<SubViewOp>(loadOp->getLoc(), buffer, loadOp.getMixedOffsets(),
                          loadOp.getMixedSizes(), loadOp.getMixedStrides());
  bvm.map(loadOp.result(), subview);
  transferShapeOpsToMemref(b, loadOp.result(), subview, bvm);
  return success();
}

LogicalResult convertInterfaceLoadTensorOp(
    OpBuilder &b, IREE::Flow::DispatchInputLoadOp loadOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value memref = bvm.lookup(loadOp.source());
  Value res = !loadOp.offsets().empty()
                  ? b.create<SubViewOp>(loc, memref, loadOp.offsets(),
                                        loadOp.sizes(), loadOp.strides())
                  :
                  // If the loadOp has no offsets/sizes and strides, it is the
                  // original op that "converts" a !flow.dispatch.input to a
                  // tensor. Just forward the subview.
                  bvm.lookup(loadOp.source());
  bvm.map(loadOp.result(), res);
  transferShapeOpsToMemref(b, loadOp.result(), res, bvm);
  return success();
}

// TODO(nicolasvasilache): evolve into flow.interface.store.tensor when the
// abstraction to get a raw void* / memref<?xi8> + offset + shape exists. Then
// we can immediately emit a view op.
// TODO(nicolasvasilache): canonicalizations of the view op.
LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceStoreTensorOp storeOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);

  // TODO(nicolasvasilache): view + subview
  if (!matchPattern(storeOp.offset(), m_Zero())) {
    return storeOp.emitError("unhandled non-zero offset");
  }

  auto phOp = createPlaceholderOp(b, storeOp.getLoc(), storeOp,
                                  storeOp.operand(), storeOp.queryBindingOp());
  Value buffer = phOp.getResult();
  b.create<linalg::CopyOp>(storeOp->getLoc(), bvm.lookup(storeOp.operand()),
                           buffer);
  storeOp->erase();
  return success();
}

// TODO(nicolasvasilache): evolve into flow.interface.store.tensor.tile when the
// abstraction exists. This will turn into some subview chain from the base
// view.
LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceStoreTensorTileOp storeOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);

  // TODO(nicolasvasilache): view + subview
  if (!matchPattern(storeOp.base_offset(), m_Zero())) {
    return storeOp.emitError("unhandled non-zero offset");
  }

  auto phOp =
      createPlaceholderOp(b, storeOp.getLoc(), storeOp, storeOp.operand(),
                          storeOp.queryBindingOp(), /*typeErase=*/true);
  Value buffer = phOp.getResult();
  Value subview =
      b.create<SubViewOp>(storeOp->getLoc(), buffer, storeOp.getMixedOffsets(),
                          storeOp.getMixedSizes(), storeOp.getMixedStrides());
  b.create<linalg::CopyOp>(storeOp->getLoc(), bvm.lookup(storeOp.operand()),
                           subview);
  storeOp->erase();
  return success();
}

/// For a given store-like `op` that is to be replaced, find the insertion point
/// in the same block earliest possible when
/// - the replacement op uses values in `usedValues`, so has to be inserted
///   after the ops that define these.
/// - The op needs to be inserted before `insertBefore` (which is in the same
/// block). Return nullptr all other times.
static Operation *getInsertionPointForReplacementStoreOp(
    Operation *op, Operation *insertBefore, ArrayRef<Value> usedValues) {
  if (op->getBlock() != insertBefore->getBlock()) return nullptr;
  Operation *insertAfter = nullptr;
  for (auto value : usedValues) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || definingOp->getBlock() != insertBefore->getBlock())
      continue;
    if (!insertAfter || insertAfter->isBeforeInBlock(definingOp))
      insertAfter = definingOp;
  }
  // All defining ops are outside of the block, so just insertBefore
  if (!insertAfter) return insertBefore;
  if (insertAfter->isBeforeInBlock(insertBefore))
    return insertAfter->getNextNode();
  return nullptr;
}

/// For cases where the value operand of the `storeOp` is produced by a
/// LinalgOp, create the subview operation that can be used by the op itself to
/// store the result into directly. This avoids an extra allocation + copies.
LogicalResult preProcessConvertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::Flow::DispatchOutputStoreOp storeOp,
    BlockAndValueMapping &bvm) {
  if (!storeOp.value().getDefiningOp<linalg::LinalgOp>()) return success();
  // Find the insertion point for the subview.
  SmallVector<Value, 4> operandsOfSubviewOp;
  operandsOfSubviewOp.push_back(bvm.lookup(storeOp.target()));
  operandsOfSubviewOp.append(storeOp.offsets().begin(),
                             storeOp.offsets().end());
  operandsOfSubviewOp.append(storeOp.sizes().begin(), storeOp.sizes().end());
  operandsOfSubviewOp.append(storeOp.strides().begin(),
                             storeOp.strides().end());
  Operation *insertionPoint = getInsertionPointForReplacementStoreOp(
      storeOp.getOperation(), storeOp.value().getDefiningOp(),
      operandsOfSubviewOp);
  if (!insertionPoint) return success();
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(insertionPoint);
  Value subview = b.create<SubViewOp>(
      storeOp.getLoc(), bvm.lookup(storeOp.target()), storeOp.offsets(),
      storeOp.sizes(), storeOp.strides());
  bvm.map(storeOp.value(), subview);
  return success();
}

LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::Flow::DispatchOutputStoreOp storeOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);
  Value storeTo = bvm.lookup(storeOp.target());
  // If the value already has a mapping, it should already have been updated in
  // place by the converted producer.
  if (storeTo) {
    storeOp->erase();
    return success();
  }
  Value subview = b.create<SubViewOp>(
      storeOp.getLoc(), bvm.lookup(storeOp.target()), storeOp.offsets(),
      storeOp.sizes(), storeOp.strides());
  b.create<linalg::CopyOp>(storeOp->getLoc(), bvm.lookup(storeOp.value()),
                           subview);
  storeOp->erase();
  return success();
}

namespace {
class LinalgBufferizePass
    : public PassWrapper<LinalgBufferizePass, FunctionPass> {
 public:
  LinalgBufferizePass(WorkgroupMemoryAllocationFn fn) : allocationFn(fn) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect, linalg::LinalgDialect, scf::SCFDialect,
                    StandardOpsDialect>();
  }
  void runOnFunction() override;

 private:
  WorkgroupMemoryAllocationFn allocationFn;
};
}  // namespace

// Special handling of dynamic sizes that must tie to InterfaceBindingSubspanOp.
// This is necessary to propagate the InterfaceLoadConstantOp to memrefs.
// In tensor world, the information is carried by TieShape ops.
// TODO(ravishankarm): This needs to be moved to MaterializeInterface pass so
// that here we dont need to deal with tie-shape ops.
static Shape::MakeRankedShapeOp getMakeRankedShapeFromInterface(
    IREE::HAL::InterfaceBindingSubspanOp op) {
  for (Operation *user : op->getUsers()) {
    auto tieOp = dyn_cast<IREE::Flow::DispatchTieShapeOp>(user);
    if (!tieOp) continue;
    auto makeRankedShapeOp =
        tieOp.shape().getDefiningOp<Shape::MakeRankedShapeOp>();
    assert(makeRankedShapeOp);
    return makeRankedShapeOp;
  }
  llvm_unreachable("Expected IREE::Flow::DispatchTieShapeOp of op");
}

void LinalgBufferizePass::runOnFunction() {
  FuncOp funcOp = getFunction();
  MLIRContext *context = &getContext();
  OpBuilder b(context);

  BlockAndValueMapping bvm;
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp op) {
    auto shapedType =
        op.getResult().getType().dyn_cast<IREE::Flow::DispatchTensorType>();
    if (!shapedType || !shapedType.hasRank()) return;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    // Just change the resulttype of InterfaceBindingSubspanOp to form the base
    // buffer.
    auto tensorType =
        op.result().getType().cast<IREE::Flow::DispatchTensorType>();
    auto memRefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    auto baseBuffer = b.create<IREE::HAL::InterfaceBindingSubspanOp>(
        op->getLoc(), memRefType, op.binding(), op.byte_offset(),
        op.byte_length());
    bvm.map(op, baseBuffer);
    transferShapeOpsToMemref(b, op.getResult(), baseBuffer.getResult(), bvm);
  });
  funcOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<IREE::HAL::InterfaceLoadTensorOp>(op)) {
      convertInterfaceLoadTensorOp(b, loadOp, bvm);
    } else if (auto loadOp =
                   dyn_cast<IREE::HAL::InterfaceLoadTensorTileOp>(op)) {
      convertInterfaceLoadTensorOp(b, loadOp, bvm);
    } else if (auto loadOp = dyn_cast<IREE::Flow::DispatchInputLoadOp>(op)) {
      convertInterfaceLoadTensorOp(b, loadOp, bvm);
    }
  });
  funcOp.walk([&](IREE::Flow::DispatchOutputStoreOp op) {
    preProcessConvertInterfaceStoreTensorOp(b, op, bvm);
  });
  funcOp.walk([&](linalg::LinalgOp op) {
    convertAnyLinalgOp(b, allocationFn, op, bvm);
  });
  funcOp.walk([&](VectorTransferOpInterface op) {
    convertTransferOp(b, allocationFn, op, bvm);
  });
  funcOp.walk([&](Operation *op) {
    if (auto storeOp = dyn_cast<IREE::HAL::InterfaceStoreTensorOp>(op)) {
      convertInterfaceStoreTensorOp(b, storeOp, bvm);
    } else if (auto storeOp =
                   dyn_cast<IREE::HAL::InterfaceStoreTensorTileOp>(op)) {
      convertInterfaceStoreTensorOp(b, storeOp, bvm);
    } else if (auto storeOp = dyn_cast<IREE::Flow::DispatchOutputStoreOp>(op)) {
      convertInterfaceStoreTensorOp(b, storeOp, bvm);
    }
  });
}

static Value defaultAllocationFn(OpBuilder &builder, Location loc,
                                 ArrayRef<Value> dynamicSizes,
                                 MemRefType allocationType) {
  return builder.create<AllocOp>(loc, allocationType, dynamicSizes);
}

std::unique_ptr<OperationPass<FuncOp>> createLinalgBufferizePass(
    WorkgroupMemoryAllocationFn allocationFn) {
  return std::make_unique<LinalgBufferizePass>(
      allocationFn ? allocationFn : defaultAllocationFn);
}

static PassRegistration<LinalgBufferizePass> pass(
    "iree-codegen-linalg-bufferize-llvm",
    "Convert from to Linalg ops on tensors to buffers",
    [] { return std::make_unique<LinalgBufferizePass>(defaultAllocationFn); });
}  // namespace iree_compiler
}  // namespace mlir
