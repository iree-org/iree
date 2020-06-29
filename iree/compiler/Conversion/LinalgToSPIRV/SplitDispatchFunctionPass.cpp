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

//===- SplitDispathFunctionPass.cpp ---------------------------------------===//
//
// This file implements a pass to split computation workload to multiple
// sequential dispatch functions. This pass operates on Linalg ops and
// scf.parallel op and prepares for lowering to GPU, where we need to tile the
// workload to workgroups and workitems. If the workload involves computation A
// and B, where B is dependent on A and A needs all workgroups to complete, then
// we need to split A and B into different kernels because there is no mechanism
// to perform cross-workgroup synchronization within a single kernel.
//
//===----------------------------------------------------------------------===//

#include <iterator>

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

namespace {

/// Returns true if the Linalg ops can be separated to multiple kernels.
bool canSeparateOps(ArrayRef<Operation *> ops) {
  if (llvm::any_of(ops, [](Operation *op) {
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
          return !linalgOp.hasBufferSemantics();
        return false;
      }))
    return false;

  // Require no other ops interleave with Linalg structured ops for now. This is
  // the common case and it simplifies further analysis.
  for (auto currOp = ops.begin(), nextOp = std::next(ops.begin());
       nextOp != ops.end(); ++currOp, ++nextOp) {
    if ((*currOp)->getNextNode() != *nextOp) return false;
  }

  return true;
}

/// Recursively collects all the operations that are referenced by given
/// `rootOp` into `closure`.
void collectAllReferencedOps(Operation *rootOp,
                             llvm::SmallPtrSetImpl<Operation *> &closure) {
  llvm::SmallVector<Operation *, 8> workList;
  workList.push_back(rootOp);

  while (!workList.empty()) {
    Operation *curOp = workList.pop_back_val();
    if (!curOp) continue;
    if (!closure.insert(curOp).second) continue;  // Seen before
    // Collect all defining ops for operands.
    for (Value operand : curOp->getOperands()) {
      if (Operation *owner = operand.getDefiningOp()) workList.push_back(owner);
    }
    // Collect all defining ops for the values used in regions.
    for (Region &region : curOp->getRegions()) {
      visitUsedValuesDefinedAbove(region, [&workList](OpOperand *operand) {
        workList.push_back(operand->get().getDefiningOp());
      });
    }
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass and patterns
//===----------------------------------------------------------------------===//

namespace {

struct SplitDispatchFunctionPass
    : public PassWrapper<SplitDispatchFunctionPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
  LogicalResult splitDispatchFunction(FuncOp oldFn, OpBuilder &builder);
};

}  // namespace

void SplitDispatchFunctionPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Collect all dispatch entry functions.
  SmallVector<FuncOp, 1> functions;
  for (FuncOp fn : moduleOp.getOps<FuncOp>()) {
    if (isEntryPoint(fn)) functions.push_back(fn);
  }
  if (functions.empty()) return;
  if (functions.size() > 1) {
    moduleOp.emitError("expected only one entry function");
    return signalPassFailure();
  }

  auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  if (failed(splitDispatchFunction(functions.front(), builder))) {
    return signalPassFailure();
  }
}

LogicalResult SplitDispatchFunctionPass::splitDispatchFunction(
    FuncOp oldFn, OpBuilder &builder) {
  // Entry functions are supported to be of `void(void)` type.
  assert(oldFn.getType().getNumInputs() == 0 &&
         oldFn.getType().getNumResults() == 0);

  if (!llvm::hasSingleElement(oldFn.getBlocks())) {
    return oldFn.emitError("expected only one block");
  }

  // The dispatch function should have more than one separable ops. Otherwise
  // there is nothing to do.
  Block &fnBody = oldFn.getBlocks().front();

  // Collect all Linalg and scf.parallel ops for distributing.
  SmallVector<Operation *, 4> separableOps;
  for (Operation &op : fnBody)
    if (isa<linalg::LinalgOp>(op) || isa<scf::ParallelOp>(op))
      separableOps.push_back(&op);

  if (separableOps.size() <= 1) return success();
  if (!canSeparateOps(separableOps)) {
    return oldFn.emitError(
        "cannot separate Linalg/Parallel ops into multiple kernels");
  }

  ModuleOp moduleOp = cast<ModuleOp>(oldFn.getParentOp());
  Block &oldFnBlock = oldFn.getBlocks().front();
  Location loc = oldFn.getLoc();

  SmallVector<std::string, 4> splitKernels;
  splitKernels.reserve(separableOps.size());
  llvm::SmallPtrSet<Operation *, 16> closure;

  for (const auto &separableOp : llvm::enumerate(separableOps)) {
    // Create a new function for hosting this op.
    splitKernels.emplace_back(llvm::formatv("{0}_dispatch_{1}", oldFn.getName(),
                                            separableOp.index()));
    StringRef newFnName = splitKernels.back();
    builder.setInsertionPointToStart(moduleOp.getBody());
    auto newFn = builder.create<FuncOp>(loc, newFnName, oldFn.getType(),
                                        /*attrs=*/ArrayRef<NamedAttribute>());

    // Copy over all attributes except type and name.
    for (const auto &namedAttr : oldFn.getAttrs()) {
      if (namedAttr.first != impl::getTypeAttrName() &&
          namedAttr.first != SymbolTable::getSymbolAttrName())
        newFn.setAttr(namedAttr.first, namedAttr.second);
    }

    // Collect the closure for the current Linalg op.
    closure.clear();
    collectAllReferencedOps(separableOp.value(), closure);

    // Clone all ops in the closure to the new function.
    Block *newFnBlock = newFn.addEntryBlock();
    builder.setInsertionPointToStart(newFnBlock);
    BlockAndValueMapping remapper;
    for (Operation &op : oldFnBlock) {
      if (closure.count(&op) == 0) continue;
      builder.insert(op.clone(remapper));
      if (&op == separableOp.value()) break;
    }
    builder.insert(oldFnBlock.getTerminator()->clone(remapper));
  }

  // Add the entry point schedule to the module op.
  SmallVector<Attribute, 4> entryPoints;
  entryPoints.reserve(separableOps.size());
  for (const std::string &kernel : splitKernels) {
    entryPoints.emplace_back(builder.getStringAttr(kernel));
  }
  moduleOp.setAttr(getEntryPointScheduleAttrName(),
                   builder.getArrayAttr(entryPoints));

  oldFn.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSplitDispatchFunctionPass() {
  return std::make_unique<SplitDispatchFunctionPass>();
}

static PassRegistration<SplitDispatchFunctionPass> pass(
    "iree-codegen-split-dispatch-function",
    "Split workload to multiple dispatch functions to satisfy computation "
    "dependency for GPU lowering");

}  // namespace iree_compiler
}  // namespace mlir
