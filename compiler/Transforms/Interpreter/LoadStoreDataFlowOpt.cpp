// Copyright 2019 Google LLC
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

#include <algorithm>

#include "third_party/llvm/llvm/include/llvm/ADT/ArrayRef.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SetVector.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/llvm/llvm/include/llvm/ADT/iterator_range.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Dialect/StandardOps/Ops.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/BlockAndValueMapping.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/StandardTypes.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/PassRegistry.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Transforms/Utils.h"
#include "third_party/mlir_edge/iree/compiler/IR/Interpreter/HLOps.h"
#include "third_party/mlir_edge/iree/compiler/IR/Ops.h"
#include "third_party/mlir_edge/iree/compiler/Utils/MemRefUtils.h"
#include "third_party/mlir_edge/iree/compiler/Utils/OpUtils.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Returns a value containing the indices in the form of a memref with shape
// {|indices|.size()}.
Value *makeIndicesMemRef(const MemRefType &type,
                         Operation::operand_range indices, OpBuilder &builder) {
  auto &useOp = *builder.getInsertionPoint();
  size_t indicesCount = std::distance(indices.begin(), indices.end());
  if (indicesCount == 0) {
    return builder
        .create<IREE::ConstantOp>(
            useOp.getLoc(), builder.getMemRefType({1}, builder.getIndexType()),
            builder.getIntegerAttr(builder.getIndexType(), 0))
        .getResult();
  } else if (indicesCount == 1) {
    auto allocOp = builder.create<AllocOp>(
        useOp.getLoc(), builder.getMemRefType({1}, builder.getIndexType()));
    auto storeIndex = builder.create<ConstantOp>(
        useOp.getLoc(), builder.getIndexType(),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    builder.create<StoreOp>(useOp.getLoc(), *indices.begin(),
                            allocOp.getResult(), ArrayRef<Value *>{storeIndex});
    return allocOp;
  }

  // TODO(benvanik): support arbitrary indices.
  useOp.emitError() << "Multiple indices are not yet implemented";
  return nullptr;
}

// Returns a value containing the lengths in the form of a memref with shape
// {|dims|.size()}.
Value *makeLengthsMemRef(Value *storedValue, OpBuilder &builder) {
  Type valueType = storedValue->getType();
  if (auto shapedType = valueType.dyn_cast<ShapedType>()) {
    auto shapeType =
        builder.getMemRefType({shapedType.getRank()}, builder.getIndexType());
    return builder.create<IREEInterp::HL::ShapeOp>(storedValue->getLoc(),
                                                   shapeType, storedValue);
  } else {
    return builder.create<IREE::ConstantOp>(
        storedValue->getLoc(),
        builder.getMemRefType({1}, builder.getIndexType()),
        builder.getIntegerAttr(builder.getIndexType(), 1));
  }
}

// Returns the origin operation of a value if it is a load.
LoadOp findOriginLoadOperation(Value *value) {
  // TODO(benvanik): follow through identity ops or something?
  if (auto loadOp = dyn_cast<LoadOp>(value->getDefiningOp())) {
    return loadOp;
  }
  return nullptr;
}

// Inserts a copy operation performing the same work as a store.
//
// Example:
//   %0 = ... : memref<4xf32>
//   %1 = load %0[%offset] : memref<4xf32>
//   %2 = ... : memref<f32>
//   store %1, %2[] : memref<f32>
//  ->
//   %0 = ... : memref<4xf32>
//   %2 = ... : memref<f32>
//   iree_hl_interp.copy %0[%offset], %2[], [%length]
void insertCopyForStore(LoadOp &loadOp, StoreOp &storeOp) {
  OpBuilder builder(storeOp);
  auto *srcIndices =
      makeIndicesMemRef(loadOp.getMemRefType(), loadOp.getIndices(), builder);
  auto *dstIndices =
      makeIndicesMemRef(storeOp.getMemRefType(), storeOp.getIndices(), builder);
  auto *lengths = makeLengthsMemRef(storeOp.getValueToStore(), builder);
  builder.create<IREEInterp::HL::CopyOp>(storeOp.getLoc(), loadOp.getMemRef(),
                                         srcIndices, storeOp.getMemRef(),
                                         dstIndices, lengths);
}

}  // namespace

class InterpreterLoadStoreDataFlowOptPass
    : public FunctionPass<InterpreterLoadStoreDataFlowOptPass> {
 public:
  void runOnFunction() override {
    auto func = getFunction();

    // Find stores and attempt to optimize load+store pairs.
    llvm::SetVector<Operation *> deadOperations;
    func.walk([&](StoreOp storeOp) {
      if (auto loadOp = findOriginLoadOperation(storeOp.getValueToStore())) {
        insertCopyForStore(loadOp, storeOp);
        deadOperations.insert(storeOp);
      }
    });

    // Remove all the now-unused ops.
    removeDeadOperations(deadOperations);
  }
};

std::unique_ptr<OpPassBase<FuncOp>>
createInterpreterLoadStoreDataFlowOptPass() {
  return std::make_unique<InterpreterLoadStoreDataFlowOptPass>();
}

static PassRegistration<InterpreterLoadStoreDataFlowOptPass> pass(
    "iree-interpreter-load-store-data-flow-opt",
    "Optimize local load and store data flow by removing redundant accesses");

}  // namespace iree_compiler
}  // namespace mlir
