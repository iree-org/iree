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

#include <utility>

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/IR/Sequencer/HLOps.h"
#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/IR/Types.h"
#include "iree/compiler/Utils/DispatchUtils.h"
#include "iree/compiler/Utils/MemRefUtils.h"
#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Inserts a load from a wrapped memref (as inserted via insertDispatcherStore).
// Returns the value in the original type.
Value *insertDispatcheeLoad(Operation *op, Type originalType, Value *value,
                            OpBuilder &builder) {
  // If old value was a memref we don't need to change anything.
  if (originalType.isa<MemRefType>()) {
    return value;
  }

  auto loadInputOp =
      builder.create<IREE::LoadInputOp>(op->getLoc(), originalType, value);
  value->replaceAllUsesWith(loadInputOp.getResult());
  loadInputOp.setOperand(value);
  return loadInputOp.getResult();
}

// Marshals args and results as buffers for the given region.
// Beyond inserting the appropriate tensor-to-memref ops we avoid mutating the
// interior of the dispatch region as much as possible.
LogicalResult marshalDispatchSite(IREE::DispatchRegionOp regionOp) {
  auto &entryBlock = regionOp.getBody().getBlocks().front();
  OpBuilder dispatcherBuilder(regionOp);
  OpBuilder dispatcheeBuilder(&entryBlock, entryBlock.begin());

  // Wrap input operands and unwrap in the entry block.
  SmallVector<Value *, 8> newArgs;
  for (int i = 0; i < regionOp.getNumArgOperands(); ++i) {
    // Wrap the input outside of the region.
    auto *blockArg = entryBlock.getArgument(i);
    Type originalType = blockArg->getType();
    auto *originalArg = regionOp.getArgOperand(i);
    auto *wrappedArg =
        insertDispatcherStore(regionOp, originalArg, dispatcherBuilder);
    newArgs.push_back(wrappedArg);
    blockArg->setType(wrappedArg->getType());

    // Unwrap the block arg value and replace all of the uses with the newly
    // unwrapped value.
    insertDispatcheeLoad(regionOp, originalType, blockArg, dispatcheeBuilder);
  }

  // Allocate output arguments and replace the return values with those.
  SmallVector<Type, 8> newResults;
  SmallVector<std::pair<int, Value *>, 8> resultIndicesToOutputArgs;
  SmallVector<int, 8> deadResultIndices;
  SmallVector<std::pair<Value *, Value *>, 8> replacedResults;
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    auto *result = regionOp.getResult(i);
    auto convertedType = convertTypeToMemRef(result->getType());

    // Allocate output buffer in the dispatcher to pass in to the region.
    Value *allocatedValue = allocateDispatchOutputBuffer(
        regionOp.getLoc(), convertedType, dispatcherBuilder);
    if (!allocatedValue) {
      regionOp.emitError("unable to allocate result value");
      return failure();
    }
    newArgs.push_back(allocatedValue);

    auto *newBlockArg = entryBlock.addArgument(allocatedValue->getType());
    resultIndicesToOutputArgs.push_back({i, newBlockArg});

    // NOTE: right now we always replace results. If we want to allow return
    // values we can avoid killing them here.
    deadResultIndices.push_back(i);
    replacedResults.push_back({result, allocatedValue});
  }

  // Remove dead results from return statements.
  regionOp.walk([&](IREE::ReturnOp returnOp) {
    // Replace the results we were returning with stores to output arguments.
    OpBuilder builder(returnOp);
    for (auto resultToArg : resultIndicesToOutputArgs) {
      auto *value = returnOp.getOperand(resultToArg.first);
      auto *outputArg = resultToArg.second;
      builder.create<IREE::StoreOutputOp>(returnOp.getLoc(), value, outputArg);
    }

    // Filter out the results that are now dead.
    SmallVector<Value *, 8> newOperands(returnOp.getOperands());
    for (int i = deadResultIndices.size() - 1; i >= 0; --i) {
      newOperands.erase(newOperands.begin() + deadResultIndices[i]);
    }
    returnOp.getOperation()->setOperands(newOperands);
  });

  // Clone the region op with the new args/results.
  auto newRegionOp = dispatcherBuilder.create<IREE::DispatchRegionOp>(
      regionOp.getLoc(), newResults, regionOp.getWorkload(), newArgs);
  newRegionOp.getBody().takeBody(regionOp.getBody());

  // Marshal back the results by replacing uses of the original with loads from
  // the new output arg.
  for (auto &it : replacedResults) {
    insertDispatcherLoad(regionOp, it.first, it.second, dispatcherBuilder);
  }

  // Remove original region.
  regionOp.erase();

  return success();
}

// Converts a dispatch_region into a dispatch to the outlined region function.
LogicalResult convertToDispatchOp(IREE::DispatchRegionOp regionOp,
                                  IREE::MultiArchExecutableOp executable,
                                  FuncOp entryPoint) {
  // Insert at the same place as the original region.
  OpBuilder dispatcherBuilder(regionOp);

  // Ensure workload is a memref.
  auto *workload =
      wrapAsMemRef(regionOp.getWorkload(), regionOp, dispatcherBuilder);

  // Create the dispatch op to the executable function.
  auto dispatchOp = dispatcherBuilder.create<IREESeq::HL::DispatchOp>(
      regionOp.getLoc(), executable.getName(), entryPoint.getName(), workload,
      entryPoint.getType().getResults(), regionOp.getArgOperands());

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i)->replaceAllUsesWith(dispatchOp.getResult(i));
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

// Outlines a dispatch region into an iree.multi_arch_executable.
LogicalResult outlineDispatchRegion(IREE::DispatchRegionOp regionOp,
                                    int outlinedRegionOrdinal) {
  // Build function type matching 1:1 with the region signature.
  SmallVector<Type, 8> operandTypes;
  for (auto *arg : regionOp.getArgOperands()) {
    operandTypes.push_back(arg->getType());
  }
  SmallVector<Type, 8> resultTypes(regionOp.getResultTypes());
  auto functionType =
      FunctionType::get(operandTypes, resultTypes, regionOp.getContext());

  // Create the executable with the region cloned into it.
  IREE::MultiArchExecutableOp multiArchExecutable;
  FuncOp outlinedFunc;
  std::tie(multiArchExecutable, outlinedFunc) = createRegionExecutable(
      regionOp, functionType,
      "_dispatch_" + std::to_string(outlinedRegionOrdinal));
  outlinedFunc.setAttr("iree.executable.export",
                       UnitAttr::get(regionOp.getContext()));

  // Finally convert the dispatch region into a dispatch to the outlined func.
  return convertToDispatchOp(regionOp, multiArchExecutable, outlinedFunc);
}

}  // namespace

class OutlineDispatchRegionsPass
    : public ModulePass<OutlineDispatchRegionsPass> {
 public:
  void runOnModule() override {
    auto module = getModule();

    SymbolTable symbolTable(module);
    auto funcs = module.getOps<FuncOp>();
    SmallVector<FuncOp, 4> funcOps(funcs.begin(), funcs.end());
    for (auto func : funcOps) {
      // Perform marshaling of the dispatcher and dispatchee I/O.
      // This inserts the required stores and loads to make everything memrefs
      // and adds the iree.load_input/iree.store_output ops to the dispatchee.
      if (func.walk([&](IREE::DispatchRegionOp op) {
                if (failed(marshalDispatchSite(op))) {
                  return WalkResult::interrupt();
                }
                return WalkResult::advance();
              })
              .wasInterrupted()) {
        return signalPassFailure();
      }

      // Outline all of the iree.dispatch_region ops in this function.
      SmallVector<IREE::DispatchRegionOp, 8> dispatchRegionOps;
      func.walk(
          [&](IREE::DispatchRegionOp op) { dispatchRegionOps.push_back(op); });
      for (int i = 0; i < dispatchRegionOps.size(); ++i) {
        if (failed(outlineDispatchRegion(dispatchRegionOps[i], i))) {
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createOutlineDispatchRegionsPass() {
  return std::make_unique<OutlineDispatchRegionsPass>();
}

static PassRegistration<OutlineDispatchRegionsPass> pass(
    "iree-outline-dispatch-regions",
    "Outlines dispatch regions into standalone functions");

}  // namespace iree_compiler
}  // namespace mlir
