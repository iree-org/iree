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
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

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

  for (auto en : llvm::enumerate(op.getOperation()->getResultTypes())) {
    size_t resultIndex = en.index();
    Type resultType = en.value();

    auto tensorType = resultType.dyn_cast<RankedTensorType>();
    if (tensorType == nullptr) {
      op.emitOpError()
          << "tensor to buffer conversion expects ranked tensor results";
      return failure();
    }
    auto tensorShape = tensorType.getShape();
    auto memrefType = MemRefType::get(tensorShape, tensorType.getElementType());

    // Allocate buffers for init tensors that are assumed to fold onto the first
    // results.
    // TODO: update this assumption because the reality is more complex
    // under linalg on tensor based transformations.
    bool foldedInitTensor = resultIndex < op.getNumInitTensors();
    if (foldedInitTensor) {
      Value initTensor = op.getInitTensor(resultIndex);
      if (initTensor.use_empty()) {
        continue;
      }
      Value initBuffer = bvm.lookup(initTensor);
      SmallVector<Value, 4> dynOperands;
      for (auto dim : llvm::enumerate(tensorShape)) {
        if (dim.value() == TensorType::kDynamicSize) {
          dynOperands.push_back(b.create<DimOp>(loc, initTensor, dim.index()));
        }
      }
      auto alloc = b.create<AllocOp>(loc, memrefType, dynOperands);
      b.create<linalg::CopyOp>(loc, initBuffer, alloc);
      resultBuffers.push_back(alloc);
      bvm.map(op->getResult(resultBuffers.size() - 1), alloc);
      continue;
    }

    // Allocate buffers for statically-shaped results.
    if (memrefType.hasStaticShape()) {
      resultBuffers.push_back(b.create<AllocOp>(loc, memrefType));
      bvm.map(op->getResult(resultBuffers.size() - 1), resultBuffers.back());
      continue;
    }

    // Perform a naive shape inference for the dynamically-shaped results.
    // Extract the required element out of the vector.
    SmallVector<Value, 4> dynOperands;
    auto resultIndexingMap = op.getOutputIndexingMap(resultIndex);
    for (auto shapeElement : llvm::enumerate(tensorType.getShape())) {
      if (loopRanges.empty()) {
        loopRanges = op.createLoopRanges(b, loc);
      }

      if (shapeElement.value() != ShapedType::kDynamicSize) {
        continue;
      }

      AffineExpr expr = resultIndexingMap.getResult(shapeElement.index());
      switch (expr.getKind()) {
        case AffineExprKind::DimId: {
          int64_t loopIndex = expr.cast<AffineDimExpr>().getPosition();
          Value size = maybeConvertToIndex(loc, loopRanges[loopIndex].size, b);
          dynOperands.push_back(size);
          break;
        }
        default:
          return failure();
      }
    }
    resultBuffers.push_back(b.create<AllocOp>(loc, memrefType, dynOperands));
    bvm.map(op->getResult(resultBuffers.size() - 1), resultBuffers.back());
  }
  return success();
}

// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
// Specialization for `linalg::GenericOp`.
/// A pattern to convert Generic Linalg operations which work on tensors to
/// use buffers. BufferPlacement pass should be later used to move
/// Alloc operations to the correct positions and insert the missing Dealloc
/// operations in the correct places.
static void finalizeBufferAllocation(OpBuilder &b, linalg::GenericOp genericOp,
                                     ValueRange inputs, ValueRange outputs,
                                     BlockAndValueMapping &bvm) {
  // Generate a new linalg operation that works on buffers.
  auto newGenericOp = b.create<linalg::GenericOp>(
      genericOp.getLoc(),
      /*resultTensorTypes=*/llvm::None,
      /*inputs=*/inputs,
      /*outputBuffers=*/outputs,
      /*initTensors=*/llvm::None, genericOp.indexing_maps(),
      genericOp.iterator_types(), genericOp.docAttr(),
      genericOp.library_callAttr(), genericOp.sparseAttr());

  // Create a new block in the region of the new Generic Op.
  Block *oldBlock = genericOp.getBody();
  Region &newRegion = newGenericOp.region();
  Block *newBlock = b.createBlock(&newRegion, newRegion.begin(),
                                  oldBlock->getArgumentTypes());

  // Add the result arguments to the new block.
  for (Value v : ValueRange(outputs).drop_front(genericOp.getNumInitTensors()))
    newBlock->addArgument(v.getType().cast<MemRefType>().getElementType());

  // Clone the body of the old block to the new block.
  BlockAndValueMapping mapping;
  mapping.map(oldBlock->getArguments(), newBlock->getArguments());

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(newBlock);
  for (auto &op : oldBlock->getOperations()) {
    Operation *clonedOp = b.clone(op, mapping);
    mapping.map(op.getResults(), clonedOp->getResults());
  }

  // Replace the results of the old op with the new output buffers.
  for (auto it : llvm::zip(genericOp.getOperation()->getResults(), outputs))
    bvm.map(std::get<0>(it), std::get<1>(it));
}

// TODO: Specialization for `linalg::IndexedGenericOp`.

// Non-conversion equivalent of the core MLIR Linalg bufferization patterns.
// Specialization for all other `linalg::LinalgOp`.
static void finalizeBufferAllocation(OpBuilder &b, linalg::LinalgOp op,
                                     ValueRange inputs, ValueRange outputs,
                                     BlockAndValueMapping &bvm) {
  assert(!isa<linalg::GenericOp>(op.getOperation()));
  assert(!isa<linalg::IndexedGenericOp>(op.getOperation()));
  SmallVector<Value, 8> newOperands = inputs;
  newOperands.append(outputs.begin(), outputs.end());
  auto otherOperands = op.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  linalg::LinalgOp res = cast<linalg::LinalgOp>(
      op.clone(b, op.getLoc(),
               /*resultTypes=*/ArrayRef<Type>{}, newOperands));

  // Need to mutate the operands_segment_sizes in the resulting op.
  res.setNumOutputBuffers(outputs.size());
  res.setNumInitTensors(0);

  // Replace the results of the old op with the new output buffers.
  for (auto it : llvm::zip(op.getOperation()->getResults(), outputs))
    bvm.map(std::get<0>(it), std::get<1>(it));
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
  for (Value v : op.getInputs()) newInputBuffers.push_back(bvm.lookup(v));
  SmallVector<Value, 2> newOutputBuffers(op.getOutputBuffers().begin(),
                                         op.getOutputBuffers().end());

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

// TODO: this will go away once integrated at the flow level.
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
// TODO: evolve to target flow ops directly with additional shape annotation.
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
    // TODO: evolve to target flow ops directly with additional shape annotation
    // so we do not need to type erase.
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

// TODO: evolve into flow.interface.load.tensor when the abstraction to get
// a raw void* / memref<?xi8> + offset + shape exists.
// Then we can immediately emit a view op.
// TODO: canonicalizations of the view op.
LogicalResult convertInterfaceLoadTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceLoadTensorOp loadOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loadOp);

  // TODO: view + subview
  if (!matchPattern(loadOp.offset(), m_Zero())) {
    return loadOp.emitError("unhandled non-zero offset");
  }

  auto phOp = createPlaceholderOp(b, loadOp->getLoc(), loadOp, loadOp.result(),
                                  loadOp.queryBindingOp());
  Value buffer = phOp.getResult();
  bvm.map(loadOp.result(), buffer);
  return success();
}

// TODO: evolve into flow.interface.load.tensor.tile when the abstraction
// exists. This will turn into some subview chain from the base view.
LogicalResult convertInterfaceLoadTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceLoadTensorTileOp loadOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(loadOp);

  // TODO: view + subview
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
  return success();
}

// TODO: evolve into flow.interface.store.tensor when the abstraction to get
// a raw void* / memref<?xi8> + offset + shape exists.
// Then we can immediately emit a view op.
// TODO: canonicalizations of the view op.
LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceStoreTensorOp storeOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);

  // TODO: view + subview
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

// TODO: evolve into flow.interface.store.tensor.tile when the abstraction
// exists. This will turn into some subview chain from the base view.
LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::HAL::InterfaceStoreTensorTileOp storeOp,
    BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);

  // TODO: view + subview
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
  funcOp.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<IREE::HAL::InterfaceLoadTensorOp>(op)) {
      convertInterfaceLoadTensorOp(b, loadOp, bvm);
    } else if (auto loadOp =
                   dyn_cast<IREE::HAL::InterfaceLoadTensorTileOp>(op)) {
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
