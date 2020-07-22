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

#include "iree/compiler/Dialect/Flow/Analysis/Dispatchability.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Utils/TypeConversion.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir {
namespace iree_compiler {

using Shape::getShapeToPrimitiveTypeExpander;

namespace IREE {
namespace Flow {

namespace {

static llvm::cl::opt<bool> traceDispatchTensors(
    "iree-flow-trace-dispatch-tensors",
    llvm::cl::desc("Trace input/output values for each dispatch function"),
    llvm::cl::init(false));

// Converts a dispatch_region into a dispatch to the outlined region function.
LogicalResult convertToDispatchOp(DispatchRegionOp regionOp,
                                  ExecutableOp executableOp,
                                  DispatchEntryOp entryPointOp,
                                  FuncOp outlinedFuncOp) {
  // Insert at the same place as the original region.
  OpBuilder builder(regionOp);

  // Perform shape to primitive type expansion.
  auto &typeExpander = getShapeToPrimitiveTypeExpander();
  SmallVector<Value, 4> origArgs(regionOp.args());
  SmallVector<Value, 4> newArgs;
  if (failed(typeExpander.expandSourceValuesToTarget(
          regionOp.getLoc(), origArgs, newArgs, builder))) {
    return failure();
  }

  auto getTensorTypeArgs = [](auto args) {
    SmallVector<Value, 4> res;
    for (auto arg : args) {
      if (arg.getType().template isa<TensorType>()) res.push_back(arg);
    }
    return res;
  };
  if (traceDispatchTensors) {
    builder.create<TensorTraceOp>(regionOp.getLoc(),
                                  getTensorTypeArgs(newArgs));
  }

  // Create the dispatch op to the executable function.
  auto dispatchOp = builder.create<DispatchOp>(
      regionOp.getLoc(), executableOp.getName(), entryPointOp.getName(),
      regionOp.workload(), outlinedFuncOp.getType().getResults(), newArgs);

  if (traceDispatchTensors) {
    builder.create<TensorTraceOp>(regionOp.getLoc(),
                                  getTensorTypeArgs(dispatchOp.getResults()));
  }

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i).replaceAllUsesWith(dispatchOp.getResult(i));
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

// Converts a region body to a function.
// The region entry block args and return terminators are used to derive the
// function type.
FuncOp createRegionFunction(Location loc, StringRef functionName,
                            Region &region) {
  // Build function type matching 1:1 with the region signature.
  SmallVector<Type, 4> operandTypes;
  SmallVector<Type, 4> resultTypes;
  auto &entryBlock = region.front();
  for (auto &operand : entryBlock.getArguments()) {
    operandTypes.push_back(operand.getType());
  }
  for (auto &block : region.getBlocks()) {
    if (auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(block.back())) {
      resultTypes = llvm::to_vector<4>(returnOp.getOperandTypes());
      break;
    }
  }

  // Clone region into the function body.
  auto functionType =
      FunctionType::get(operandTypes, resultTypes, region.getContext());
  auto funcOp = FuncOp::create(loc, functionName, functionType);
  BlockAndValueMapping mapping;
  region.cloneInto(&funcOp.getBody(), mapping);

  // Replace flow.return with std.return.
  for (auto &block : funcOp.getBlocks()) {
    if (auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(block.back())) {
      OpBuilder builder(returnOp);
      builder.create<mlir::ReturnOp>(
          returnOp.getLoc(), llvm::to_vector<4>(returnOp.getOperands()));
      returnOp.erase();
    }
  }

  // Remove any tie_shapes not from entry block args.
  // TODO(laurenzo): Remove this once we are not materializing ties in
  // dispatch regions at all. For now, this at least provides a better
  // contract to the backends without leaking the fact that dispatch
  // formation fully materializes ties.
  auto *newEntryBlock = &funcOp.getBlocks().front();
  funcOp.walk([&](Shape::TieShapeOp tieOp) {
    if (auto blockArg = tieOp.operand().dyn_cast<BlockArgument>()) {
      if (blockArg.getOwner() == newEntryBlock) return;
    }
    // Elide.
    tieOp.result().replaceAllUsesWith(tieOp.operand());
    tieOp.erase();
  });

  // Expand shape types to primitives.
  auto &typeExpander = getShapeToPrimitiveTypeExpander();
  OpBuilder expandBuilder(funcOp.getContext());
  if (failed(typeExpander.expandFunctionSignature(funcOp, expandBuilder)) ||
      failed(typeExpander.expandAllReturnLikeTerminators<mlir::ReturnOp>(
          funcOp, expandBuilder))) {
    return nullptr;
  }

  return funcOp;
}

// Outlines a dispatch region into a flow.executable.
LogicalResult outlineDispatchRegion(
    DispatchRegionOp regionOp, int outlinedRegionOrdinal,
    llvm::StringMap<FuncOp> &dispatchableFuncOps) {
  // Create the dispatch function.
  auto parentFuncOp = regionOp.getParentOfType<FuncOp>();
  std::string namePrefix = parentFuncOp.getName().str() + "_ex_dispatch_" +
                           std::to_string(outlinedRegionOrdinal);

  // Convert the region to a function.
  auto dispatchFuncOp =
      createRegionFunction(regionOp.getLoc(), namePrefix, regionOp.body());
  if (!dispatchFuncOp) {
    return failure();
  }

  // Create the executable with the region cloned into it.
  auto executableOp = createExecutable(
      regionOp.getLoc(), namePrefix, {dispatchFuncOp},
      parentFuncOp.getParentOfType<ModuleOp>(), dispatchableFuncOps);
  executableOp.getOperation()->moveBefore(parentFuncOp);
  SymbolTable::setSymbolVisibility(executableOp,
                                   SymbolTable::Visibility::Private);

  // Add dispatch export pointing at the function.
  OpBuilder builder(executableOp.body());
  auto entryPointOp = builder.create<DispatchEntryOp>(
      regionOp.getLoc(), builder.getStringAttr(dispatchFuncOp.getName()),
      builder.getSymbolRefAttr(dispatchFuncOp), IntegerAttr{});

  // Finally convert the dispatch region into a dispatch to the outlined func.
  return convertToDispatchOp(regionOp, executableOp, entryPointOp,
                             dispatchFuncOp);
}

}  // namespace

class OutlineDispatchRegionsPass
    : public PassWrapper<OutlineDispatchRegionsPass, OperationPass<ModuleOp>> {
 public:
  OutlineDispatchRegionsPass() = default;

  void runOnOperation() override {
    auto dispatchability = getCachedAnalysis<Dispatchability>();
    llvm::StringMap<FuncOp> dispatchableFuncOps;
    if (dispatchability.hasValue()) {
      // if we do not get dispatchability from cache,
      // we should keep dispatchableFuncOps empty to be comptaible as before
      dispatchability.getValue().get().walkDispatchableOps([&](FuncOp funcOp) {
        dispatchableFuncOps[funcOp.getName()] = funcOp;
      });
    }

    // TODO(benvanik): replace with a pattern rewriter?
    auto funcOps = llvm::to_vector<32>(getOperation().getOps<FuncOp>());
    for (auto funcOp : funcOps) {
      // Outline all of the dispatch regions ops in this function.
      SmallVector<DispatchRegionOp, 8> dispatchRegionOps;
      funcOp.walk(
          [&](DispatchRegionOp op) { dispatchRegionOps.push_back(op); });
      for (int i = 0; i < dispatchRegionOps.size(); ++i) {
        if (failed(outlineDispatchRegion(dispatchRegionOps[i], i,
                                         dispatchableFuncOps))) {
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createOutlineDispatchRegionsPass() {
  return std::make_unique<OutlineDispatchRegionsPass>();
}

static PassRegistration<OutlineDispatchRegionsPass> pass(
    "iree-flow-outline-dispatch-regions",
    "Outlines dispatch regions into standalone functions");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
