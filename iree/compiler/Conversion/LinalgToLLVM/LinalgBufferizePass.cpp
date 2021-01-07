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
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

// Traverse all transitive "TieShapeOp" uses of `tensor`, map them to `memref`.
// Replace all uses of any such transitive tensor in a "DimOp" by `memref`.
// TODO(nicolasvasilache): kill "TieShapeOp" with fire.
static void mapAllTieShapeUsesAndReplaceDimUsesOf(Value tensor, Value memref,
                                                  BlockAndValueMapping &bvm) {
  SmallVector<Value, 4> workList{tensor};
  for (unsigned idx = 0; idx < workList.size() /*recompute: it grows*/; ++idx) {
    // Update of DimOp uses must occur in 2 steps or use list gets messed up.
    SmallVector<OpOperand *, 4> dimOps;
    for (auto &use : workList[idx].getUses()) {
      auto tieShapeOp = dyn_cast<Shape::TieShapeOp>(use.getOwner());
      auto dimOp = dyn_cast<DimOp>(use.getOwner());
      if (tieShapeOp) {
        workList.push_back(tieShapeOp.getResult());
        bvm.map(tieShapeOp.getResult(), memref);
      } else if (dimOp) {
        dimOps.push_back(&dimOp->getOpOperand(use.getOperandNumber()));
      }
    }
    for (OpOperand *opOperand : dimOps) {
      opOperand->set(memref);
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
    OpBuilder &b, Location loc, linalg::LinalgOp op,
    SmallVectorImpl<Value> &resultBuffers, BlockAndValueMapping &bvm) {
  // Lazily compute loopRanges.
  SmallVector<Range, 4> loopRanges;

  assert(op.getNumOutputs() == op->getNumResults());
  for (auto en : llvm::enumerate(op->getResultTypes())) {
    size_t resultIndex = en.index();
    Value resultTensor = op.getOutput(resultIndex);

    // If output tensor was produced by a LinalgOp, just reuse the buffer.
    // TODO(nicolasvasilache): this may be too brutal and we may prefer to leave
    // this decision to a copy + alloc removal pass.
    if (resultTensor.getDefiningOp<linalg::LinalgOp>()) {
      resultBuffers.push_back(bvm.lookup(resultTensor));
      continue;
    }

    Type resultType = en.value();
    auto tensorType = resultType.dyn_cast<RankedTensorType>();
    auto tensorShape = tensorType.getShape();
    auto memrefType = MemRefType::get(tensorShape, tensorType.getElementType());
    SmallVector<Value, 4> dynOperands;
    for (auto dim : llvm::enumerate(tensorShape)) {
      if (dim.value() == TensorType::kDynamicSize) {
        dynOperands.push_back(
            b.create<DimOp>(loc, bvm.lookup(resultTensor), dim.index()));
      }
    }
    auto alloc = b.create<AllocOp>(loc, memrefType, dynOperands);
    resultBuffers.push_back(alloc);

    // Additionally, if the output buffer is used, clone its value for now.
    if (op.payloadUsesValueFromOutputOperandIndex(resultIndex))
      b.create<linalg::CopyOp>(loc, bvm.lookup(resultTensor), alloc);
  }
  bvm.map(op->getResults(), resultBuffers);
  // TODO(nicolasvasilache): kill tie_shape with fire.
  for (auto it : llvm::zip(op->getResults(), resultBuffers)) {
    mapAllTieShapeUsesAndReplaceDimUsesOf(std::get<0>(it), std::get<1>(it),
                                          bvm);
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
  bvm.map(op.getOperation()->getResults(), outputs);
  // TODO(nicolasvasilache): kill tie_shape with fire.
  for (auto it : llvm::zip(op.getOperation()->getResults(), outputs)) {
    mapAllTieShapeUsesAndReplaceDimUsesOf(std::get<0>(it), std::get<1>(it),
                                          bvm);
  }
}

//===----------------------------------------------------------------------===//
// Bufferization helper functions using BlockAndValueMapping.
//===----------------------------------------------------------------------===//

/// Generic conversion pattern that matches any linalg::LinalgOp. This avoids
/// template instantiating one pattern for each linalg::LinalgOp.
LogicalResult convertAnyLinalgOp(OpBuilder &b, linalg::LinalgOp op,
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
  if (failed(allocateBuffersForResults(b, loc, op, newOutputBuffers, bvm))) {
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
  // TODO(nicolasvasilache): kill tie_shape with fire.
  mapAllTieShapeUsesAndReplaceDimUsesOf(loadOp.result(), buffer, bvm);
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
      b.create<SubViewOp>(loadOp->getLoc(), buffer,
                          extractFromI64ArrayAttr(loadOp.static_offsets()),
                          extractFromI64ArrayAttr(loadOp.static_sizes()),
                          extractFromI64ArrayAttr(loadOp.static_strides()),
                          loadOp.offsets(), loadOp.sizes(), loadOp.strides());
  bvm.map(loadOp.result(), subview);
  // TODO(nicolasvasilache): kill tie_shape with fire.
  mapAllTieShapeUsesAndReplaceDimUsesOf(loadOp.result(), subview, bvm);
  return success();
}

LogicalResult convertInterfaceLoadTensorOp(
    OpBuilder &b, IREE::Flow::DispatchInputLoadOp loadOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loadOp);
  Value subview =
      b.create<SubViewOp>(loadOp.getLoc(), bvm.lookup(loadOp.source()),
                          loadOp.offsets(), loadOp.sizes(), loadOp.strides());
  bvm.map(loadOp.result(), subview);
  // TODO(nicolasvasilache): kill tie_shape with fire.
  mapAllTieShapeUsesAndReplaceDimUsesOf(loadOp.result(), subview, bvm);
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
  ;
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
  Value subview = b.create<SubViewOp>(
      storeOp->getLoc(), buffer,
      extractFromI64ArrayAttr(storeOp.static_offsets()),
      extractFromI64ArrayAttr(storeOp.static_sizes()),
      extractFromI64ArrayAttr(storeOp.static_strides()), storeOp.offsets(),
      storeOp.sizes(), storeOp.strides());
  b.create<linalg::CopyOp>(storeOp->getLoc(), bvm.lookup(storeOp.operand()),
                           subview);
  storeOp->erase();
  return success();
}

LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::Flow::DispatchOutputStoreOp storeOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);
  Value subview = b.create<SubViewOp>(
      storeOp.getLoc(), bvm.lookup(storeOp.target()), storeOp.offsets(),
      storeOp.sizes(), storeOp.strides());
  b.create<linalg::CopyOp>(storeOp->getLoc(), bvm.lookup(storeOp.value()),
                           subview);
  storeOp->erase();
  return success();
}

namespace {
struct LinalgLLVMBufferizePass
    : public PassWrapper<LinalgLLVMBufferizePass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect, linalg::LinalgDialect, scf::SCFDialect,
                    StandardOpsDialect>();
  }
  void runOnFunction() override;
};
}  // namespace

void LinalgLLVMBufferizePass::runOnFunction() {
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
    Value memref = b.create<IREE::HAL::InterfaceBindingSubspanOp>(
        op->getLoc(),
        MemRefType::get(shapedType.getShape(), shapedType.getElementType()),
        op.binding(), op.byte_offset(), op.byte_length());
    bvm.map(op, memref);
    // TODO(nicolasvasilache): kill tie_shape with fire.
    mapAllTieShapeUsesAndReplaceDimUsesOf(op.result(), memref, bvm);
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
  funcOp.walk([&](linalg::LinalgOp op) { convertAnyLinalgOp(b, op, bvm); });
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

std::unique_ptr<OperationPass<FuncOp>> createLinalgLLVMBufferizePass() {
  return std::make_unique<LinalgLLVMBufferizePass>();
}

static PassRegistration<LinalgLLVMBufferizePass> pass(
    "iree-codegen-linalg-bufferize-llvm",
    "Convert from to Linalg ops on tensors to buffers",
    [] { return std::make_unique<LinalgLLVMBufferizePass>(); });
}  // namespace iree_compiler
}  // namespace mlir
