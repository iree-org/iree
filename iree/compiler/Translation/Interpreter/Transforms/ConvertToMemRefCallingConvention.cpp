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

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/Interpreter/IR/CommonOps.h"
#include "iree/compiler/Translation/Interpreter/Utils/MemRefUtils.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Attempts to resolve the use of a value back to the MemRef it was loaded from.
// Returns either a MemRef view containing the value or nullptr if the value was
// not loaded from a MemRef (or is possibly unknown).
Value resolveValueToSourceMemRef(Value value, Operation *useOp) {
  // TODO(benvanik): implement this for real; this is naive but enough for our
  // simple load patterns.
  auto *defInstr = value.getDefiningOp();
  if (auto loadOp = dyn_cast_or_null<LoadOp>(defInstr)) {
    // TODO(benvanik): support views.
    return loadOp.getMemRef();
  }
  return nullptr;
}

void copyOperationAttrs(Operation *oldOp, Operation *newOp) {
  for (const auto &oldAttr : oldOp->getAttrs()) {
    newOp->setAttr(oldAttr.first, oldAttr.second);
  }
}

FunctionType getMemRefFunctionType(FunctionType type) {
  Builder builder(type.getContext());
  llvm::SmallVector<Type, 8> replacementInputs;
  for (auto type : type.getInputs()) {
    auto memRefType = convertLegacyTypeToMemRef(type);
    if (!memRefType) {
      return nullptr;
    }
    replacementInputs.push_back(memRefType);
  }
  llvm::SmallVector<Type, 8> replacementResults;
  for (auto type : type.getResults()) {
    auto memRefType = convertLegacyTypeToMemRef(type);
    if (!memRefType) {
      return nullptr;
    }
    replacementResults.push_back(memRefType);
  }
  return builder.getFunctionType(replacementInputs, replacementResults);
}

bool insertLoad(BlockArgument oldArg, BlockArgument newArg, OpBuilder &builder,
                BlockAndValueMapping *mapping) {
  auto loc = oldArg.getOwner()->getParent()->getLoc();

  // If old arg was a memref we don't need to change anything. We still need
  // to remap so that the use lists match through conversion, though.
  if (oldArg.getType().isa<MemRefType>()) {
    mapping->map(oldArg, newArg);
    return false;
  } else if (oldArg.getType().isa<TensorType>()) {
    auto castOp = builder.create<IREEInterp::MemRefToTensorOp>(loc, newArg);
    mapping->map(oldArg, castOp.getResult());
    return false;
  }

  // Insert the load we'll use to unbox the value.
  auto loadedValue = builder.create<LoadOp>(loc, newArg, ArrayRef<Value>{});
  mapping->map(oldArg, loadedValue);

  return false;
}

bool insertLoad(Operation *oldOp, Value oldValue, Value newValue,
                OpBuilder &builder, BlockAndValueMapping *mapping) {
  // If old value was a memref we don't need to change anything.
  if (oldValue.getType().isa<MemRefType>()) {
    mapping->map(oldValue, newValue);
    return false;
  } else if (oldValue.getType().isa<TensorType>()) {
    auto castOp =
        builder.create<IREEInterp::MemRefToTensorOp>(oldOp->getLoc(), newValue);
    mapping->map(oldValue, castOp.getResult());
    return false;
  }

  assert(newValue.getType().isa<MemRefType>());

  // Insert the load we'll use to unbox the value.
  auto loadedValue =
      builder.create<LoadOp>(oldOp->getLoc(), newValue, ArrayRef<Value>{});
  mapping->map(oldValue, loadedValue);

  return false;
}

Value insertStore(Operation *oldOp, Value oldValue, OpBuilder &builder,
                  BlockAndValueMapping *mapping) {
  auto newValue = mapping->lookupOrNull(oldValue);
  if (!newValue) {
    return nullptr;
  }

  // If the previous value was already a memref we don't need to change
  // anything.
  // TODO(benvanik): ensure indices make sense.
  if (oldValue.getType().isa<MemRefType>()) {
    return newValue;
  } else if (oldValue.getType().isa<TensorType>()) {
    auto castOp =
        builder.create<IREEInterp::TensorToMemRefOp>(oldOp->getLoc(), newValue);
    return castOp.getResult();
  }

  // Look back up and see if we can find the memref the value was loaded from.
  if (auto sourceMemRef = resolveValueToSourceMemRef(oldValue, oldOp)) {
    return mapping->lookupOrNull(sourceMemRef);
  }

  // Allocate the memref to store the value.
  auto newStorage = builder.create<AllocOp>(
      oldOp->getLoc(), convertLegacyTypeToMemRef(oldValue.getType()));

  // Insert the store we'll use to box the value.
  builder.create<StoreOp>(oldOp->getLoc(), newValue, newStorage,
                          ArrayRef<Value>{});

  return newStorage;
}

bool convertCallOp(CallOp *oldOp, OpBuilder &builder,
                   BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  for (auto oldArg : oldOp->getOperands()) {
    auto newArg = insertStore(oldOp->getOperation(), oldArg, builder, mapping);
    if (!newArg) {
      return true;
    }
    newArgs.push_back(newArg);
  }

  SmallVector<Type, 4> resultTypes;
  for (auto oldType : oldOp->getOperation()->getResultTypes()) {
    resultTypes.push_back(convertLegacyTypeToMemRef(oldType));
  }
  auto newOp = builder.create<CallOp>(oldOp->getLoc(), oldOp->getCallee(),
                                      resultTypes, newArgs);
  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  for (int i = 0; i < newOp.getNumResults(); ++i) {
    auto oldResult = oldOp->getResult(i);
    auto newResult = newOp.getResult(i);
    if (insertLoad(oldOp->getOperation(), oldResult, newResult, builder,
                   mapping)) {
      return true;
    }
  }

  return false;
}

bool convertCallIndirectOp(CallIndirectOp *oldOp, OpBuilder &builder,
                           BlockAndValueMapping *mapping) {
  // TODO(benvanik): support wrapping callee values.
  oldOp->emitError("CallIndirectOp not yet supported");
  return true;
#if 0
  llvm::SmallVector<Value, 4> newArgs;
  for (auto *oldArg : oldOp->getArgOperands()) {
    auto *newArg = insertStore(oldOp->getOperation(), oldArg, builder, mapping);
    if (!newArg) {
      return true;
    }
    newArgs.push_back(newArg);
  }

  auto newOp = builder.create<CallIndirectOp>(oldOp->getLoc(),
                                               oldOp->getCallee(), newArgs);
  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  for (int i = 0; i < newOp.getNumResults(); ++i) {
    auto *oldResult = oldOp->getResult(i);
    auto *newResult = newOp.getResult(i);
    if (insertLoad(oldOp->getOperation(), oldResult, newResult, builder,
                   mapping)) {
      return true;
    }
  }

  return false;
#endif  // 0
}

bool convertReturnOp(Operation *oldOp, OpBuilder &builder,
                     BlockAndValueMapping *mapping) {
  BlockAndValueMapping returnMapping;
  for (auto oldArg : oldOp->getOperands()) {
    auto newArg = insertStore(oldOp, oldArg, builder, mapping);
    if (!newArg) {
      return true;
    }
    returnMapping.map(oldArg, newArg);
  }

  builder.clone(*oldOp, returnMapping);
  return false;
}

bool convertBranchOp(BranchOp *oldOp, OpBuilder &builder,
                     BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 4> newArgs;
  for (auto oldArg : oldOp->getOperands()) {
    auto newArg = insertStore(oldOp->getOperation(), oldArg, builder, mapping);
    if (!newArg) {
      return true;
    }
    newArgs.push_back(newArg);
  }

  auto *dest = mapping->lookupOrNull(oldOp->getDest());
  if (!dest) {
    oldOp->emitError("Destination block mapping not found");
    return true;
  }

  auto newOp = builder.create<BranchOp>(oldOp->getLoc(), dest, newArgs);
  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  return false;
}

bool convertCondBranchOp(CondBranchOp *oldOp, OpBuilder &builder,
                         BlockAndValueMapping *mapping) {
  llvm::SmallVector<Value, 4> trueArgs;
  for (auto oldArg : oldOp->getTrueOperands()) {
    auto newArg = insertStore(oldOp->getOperation(), oldArg, builder, mapping);
    if (!newArg) {
      return true;
    }
    trueArgs.push_back(newArg);
  }
  llvm::SmallVector<Value, 4> falseArgs;
  for (auto oldArg : oldOp->getFalseOperands()) {
    auto newArg = insertStore(oldOp->getOperation(), oldArg, builder, mapping);
    if (!newArg) {
      return true;
    }
    falseArgs.push_back(newArg);
  }

  auto *trueDest = mapping->lookupOrNull(oldOp->getTrueDest());
  if (!trueDest) {
    oldOp->emitError("True destination block mapping not found");
    return true;
  }
  auto *falseDest = mapping->lookupOrNull(oldOp->getFalseDest());
  if (!falseDest) {
    oldOp->emitError("False destination block mapping not found");
    return true;
  }

  // Lowering will take care of the condition store.
  auto newCondition = mapping->lookupOrNull(oldOp->getCondition());
  if (!newCondition) {
    oldOp->emitError("Condition value mapping not found");
    return false;
  }

  auto newOp = builder.create<CondBranchOp>(
      oldOp->getLoc(), newCondition, trueDest, trueArgs, falseDest, falseArgs);
  copyOperationAttrs(oldOp->getOperation(), newOp.getOperation());

  return false;
}

bool convertOperation(Operation *oldOp, OpBuilder &builder,
                      BlockAndValueMapping *mapping) {
  if (isa<ConstantOp>(oldOp)) {
    builder.clone(*oldOp, *mapping);
    return false;
  } else if (auto callOp = dyn_cast<CallOp>(oldOp)) {
    return convertCallOp(&callOp, builder, mapping);
  } else if (auto callIndirectOp = dyn_cast<CallIndirectOp>(oldOp)) {
    return convertCallIndirectOp(&callIndirectOp, builder, mapping);
  } else if (isa<ReturnOp>(oldOp)) {
    return convertReturnOp(oldOp, builder, mapping);
  } else if (auto branchOp = dyn_cast<BranchOp>(oldOp)) {
    return convertBranchOp(&branchOp, builder, mapping);
  } else if (auto condBranchOp = dyn_cast<CondBranchOp>(oldOp)) {
    return convertCondBranchOp(&condBranchOp, builder, mapping);
  } else {
    builder.clone(*oldOp, *mapping);
    return false;
  }
}

bool convertFunction(FuncOp oldFunc, FuncOp newFunc) {
  OpBuilder builder(newFunc.getBody());
  BlockAndValueMapping mapping;

  // Create new blocks matching the expected arguments of the old ones.
  // This sets up the block mappings to enable us to reference blocks forward
  // during conversion.
  newFunc.getBlocks().clear();
  for (auto &oldBlock : oldFunc.getBlocks()) {
    auto *newBlock = builder.createBlock(&newFunc.getBody());
    for (auto oldArg : oldBlock.getArguments()) {
      // Replace the block args with memrefs.
      auto memRefType = convertLegacyTypeToMemRef(oldArg.getType());
      if (!memRefType) return true;
      auto newArg = newBlock->addArgument(memRefType);

      // Insert loads to preserve type, if needed.
      // This will replace all uses of the oldArg with the loaded value from
      // newArg so that the block contents are still using unwrapped values.
      if (insertLoad(oldArg, newArg, builder, &mapping)) {
        return true;
      }
    }
    mapping.map(&oldBlock, newBlock);
  }

  // Convert all ops in the blocks.
  for (auto &oldBlock : oldFunc.getBlocks()) {
    builder.setInsertionPointToEnd(mapping.lookupOrNull(&oldBlock));
    for (auto &oldOp : oldBlock.getOperations()) {
      if (convertOperation(&oldOp, builder, &mapping)) {
        return true;
      }
    }
  }

  return false;
}

}  // namespace

class ConvertToMemRefCallingConventionPass
    : public ModulePass<ConvertToMemRefCallingConventionPass> {
 public:
  void runOnModule() override {
    auto module = getModule();

    // Build a list of (oldFunc, newFunc) for all functions we need to
    // replace. This will ensure that when we go to convert function bodies we
    // have only new functions defined.
    std::vector<std::pair<FuncOp, FuncOp>> convertedFunctions;

    for (auto oldFunc : module.getOps<FuncOp>()) {
      // Create the replacement function, ensuring that we copy attributes.
      auto functionType = getMemRefFunctionType(oldFunc.getType());
      if (!functionType) {
        return signalPassFailure();
      }

      auto newFunc = FuncOp::create(oldFunc.getLoc(), oldFunc.getName(),
                                    functionType, oldFunc.getDialectAttrs());
      convertedFunctions.push_back({oldFunc, newFunc});

      // Perform the actual body conversion now.
      if (convertFunction(oldFunc, newFunc)) {
        return signalPassFailure();
      }
    }

    // Replace functions in the module.
    for (auto &pair : convertedFunctions) {
      pair.first.erase();
      module.push_back(pair.second);
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>>
createConvertToMemRefCallingConventionPass() {
  return std::make_unique<ConvertToMemRefCallingConventionPass>();
}

static PassRegistration<ConvertToMemRefCallingConventionPass> pass(
    "convert-to-memref-calling-convention",
    "Convert functions to use a memref-based calling convention.");

}  // namespace iree_compiler
}  // namespace mlir
