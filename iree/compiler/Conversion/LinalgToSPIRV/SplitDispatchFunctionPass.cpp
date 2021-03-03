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
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "split-dispatch-function"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns true if an op can be fused with the list of ops that are to be put
/// in the same entry point function. This should be consistent with whatthe
/// downstream passes can handle.
static bool isFusableWithCurrentOpsList(
    Operation *nextOp, ArrayRef<Operation *> currOpsList,
    const linalg::LinalgDependenceGraph &dependenceGraph) {
  if (currOpsList.empty()) return true;

  linalg::LinalgOp dstOp = dyn_cast<linalg::LinalgOp>(nextOp);
  linalg::LinalgOp srcOp = dyn_cast<linalg::LinalgOp>(currOpsList.back());
  if (dstOp && srcOp) {
    // TODO(#2963): This splits independent linalg opreations into its own
    // dispatch, but in reality if the iteration domain of the ops are the same,
    // and they have all iterator types parallel, they could be put in the same
    // dispatch region.
    if (!dependenceGraph.hasDependenceFrom(srcOp, dstOp)) return false;

#define ADD_FUSABLE_PAIR(SrcOpTy, DstOpTy, DependenceTy)             \
  if (isa<SrcOpTy>(srcOp.getOperation()) &&                          \
      isa<DstOpTy>(dstOp.getOperation()) &&                          \
      dependenceGraph.hasDependenceFrom(srcOp, dstOp, DependenceTy)) \
    return true;

    ADD_FUSABLE_PAIR(linalg::BatchMatmulOp, linalg::GenericOp,
                     linalg::LinalgDependenceGraph::DependenceType::RAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::BatchMatmulOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::ConvInputNWCFilterWCFOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::ConvInputNHWCFilterHWCFOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::ConvInputNDHWCFilterDHWCFOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::DepthwiseConvInputNHWCFilterHWCOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::DepthwiseConvInputNHWCFilterHWCFOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::MatmulOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::BatchMatmulOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::PoolingMaxOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::PoolingMinOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::FillOp, linalg::PoolingSumOp,
                     linalg::LinalgDependenceGraph::DependenceType::WAW)
    ADD_FUSABLE_PAIR(linalg::MatmulOp, linalg::GenericOp,
                     linalg::LinalgDependenceGraph::DependenceType::RAW)

#undef ADD_FUSABLE_PAIR
  }
  return false;
}

/// For the list of operations in `ops` returns a list of lists where each list
/// contains the operations that need to be put in a separate dispatch function.
static LogicalResult separateOps(
    ArrayRef<Operation *> ops,
    const linalg::LinalgDependenceGraph &dependenceGraph,
    SmallVectorImpl<SmallVector<Operation *, 1>> &fusedOpList) {
  assert(!ops.empty() &&
         "expected at least one separable op for splitting dispatch function");
  SmallVector<Operation *, 1> currList;
  for (auto currOp = ops.begin(), nextOp = std::next(ops.begin());
       nextOp != ops.end(); ++currOp, ++nextOp) {
    // Check that the operation has buffer semantics.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(*currOp)) {
      if (!linalgOp.hasBufferSemantics()) return failure();
    }

    // Require no other non-metadata ops interleave with Linalg structured ops
    // for now. This is the common case and it simplifies further analysis.
    Operation *iter = (*currOp)->getNextNode();
    while (iter != *nextOp && (MemoryEffectOpInterface::hasNoEffect(iter) ||
                               isa<IREE::PlaceholderOp>(iter)))
      iter = iter->getNextNode();
    if (iter != *nextOp) return failure();

    currList.push_back(*currOp);

    // If the nextOp is not fusible with the currOp, then record the list of ops
    // so far, and start a new list.
    if (isFusableWithCurrentOpsList(*nextOp, currList, dependenceGraph)) {
      continue;
    }

    // Push the current list of ops into the list of lists `currList` and
    // start a new list.
    fusedOpList.emplace_back();
    std::swap(fusedOpList.back(), currList);
  }
  currList.push_back(ops.back());
  fusedOpList.emplace_back(std::move(currList));
  return success();
}

/// Recursively collects all the operations that are referenced by given
/// `rootOp` into `closure`.
static void collectAllReferencedOps(
    ArrayRef<Operation *> rootOps,
    llvm::SmallPtrSetImpl<Operation *> &closure) {
  llvm::SmallVector<Operation *, 8> workList;
  workList.assign(rootOps.begin(), rootOps.end());

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

//===----------------------------------------------------------------------===//
// Pass and patterns
//===----------------------------------------------------------------------===//

namespace {

struct SplitDispatchFunctionPass
    : public PassWrapper<SplitDispatchFunctionPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
  void runOnOperation() override;
  LogicalResult splitDispatchFunction(FuncOp oldFn, OpBuilder &builder);
};

}  // namespace

void SplitDispatchFunctionPass::runOnOperation() {
  IREE::HAL::ExecutableTargetOp targetOp = getOperation();
  ModuleOp moduleOp = targetOp.getInnerModule();

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
  IREE::HAL::ExecutableEntryPointOp oldEntryPointOp = getEntryPoint(oldFn);
  if (!oldEntryPointOp) {
    return oldFn.emitError("unable to find iree.executable.entry_point for ")
           << oldFn.getName();
  }
  // The dispatch function should have more than one separable ops. Otherwise
  // there is nothing to do.
  Block &fnBody = oldFn.getBlocks().front();

  // Collect all Linalg and scf.parallel ops for splitting.
  SmallVector<Operation *, 4> separableOps;
  for (Operation &op : fnBody)
    if (isa<linalg::LinalgOp, scf::ParallelOp, scf::ForOp>(op))
      separableOps.push_back(&op);

  if (separableOps.size() <= 1) return success();

  linalg::Aliases aliases;
  linalg::LinalgDependenceGraph dependenceGraph =
      linalg::LinalgDependenceGraph::buildDependenceGraph(aliases, oldFn);
  SmallVector<SmallVector<Operation *, 1>, 1> fusedOpsList;
  if (failed(separateOps(separableOps, dependenceGraph, fusedOpsList))) {
    return oldFn.emitError(
        "cannot separate Linalg/Parallel ops into multiple kernels");
  }
  if (fusedOpsList.size() <= 1) return success();

  ModuleOp moduleOp = cast<ModuleOp>(oldFn->getParentOp());
  Block &oldFnBlock = oldFn.getBlocks().front();
  Location loc = oldFn.getLoc();
  SmallVector<Attribute, 4> entryPoints;

  for (const auto &fusedOps : llvm::enumerate(fusedOpsList)) {
    if (fusedOps.value().empty()) continue;
    // Create a new function for hosting this op.
    std::string newFnName =
        llvm::formatv("{0}_dispatch_{1}", oldFn.getName(), fusedOps.index());
    builder.setInsertionPointToStart(moduleOp.getBody());
    auto newFn = builder.create<FuncOp>(loc, newFnName, oldFn.getType());
    LLVM_DEBUG({
      llvm::dbgs() << "Created new function : func @" << newFn.getName()
                   << "\n";
    });

    // Copy over all attributes except type and name.
    for (const auto &namedAttr : oldFn->getAttrs()) {
      if (namedAttr.first != impl::getTypeAttrName() &&
          namedAttr.first != SymbolTable::getSymbolAttrName())
        newFn->setAttr(namedAttr.first, namedAttr.second);
    }

    // Add the entry point operations for the new fn.
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(oldEntryPointOp);
      auto clonedEntryPointOp = cast<IREE::HAL::ExecutableEntryPointOp>(
          builder.clone(*oldEntryPointOp.getOperation()));
      clonedEntryPointOp.sym_nameAttr(builder.getStringAttr(newFnName));
      clonedEntryPointOp.ordinalAttr(
          builder.getI32IntegerAttr(static_cast<int32_t>(entryPoints.size())));
      entryPoints.push_back(builder.getSymbolRefAttr(clonedEntryPointOp));
    }

    // Collect the closure for the current Linalg op.
    llvm::SmallPtrSet<Operation *, 16> closure;
    collectAllReferencedOps(fusedOps.value(), closure);

    // Clone all ops in the closure to the new function.
    Block *newFnBlock = newFn.addEntryBlock();
    builder.setInsertionPointToStart(newFnBlock);
    BlockAndValueMapping remapper;
    for (Operation &op : oldFnBlock) {
      if (closure.count(&op) == 0) continue;
      builder.insert(op.clone(remapper));
      if (&op == fusedOps.value().back()) break;
    }
    builder.insert(oldFnBlock.getTerminator()->clone(remapper));
  }
  moduleOp->setAttr(getEntryPointScheduleAttrName(),
                    builder.getArrayAttr(entryPoints));

  LLVM_DEBUG({ llvm::dbgs() << "Erased func @" << oldFn.getName() << "\n"; });
  oldFn.erase();
  oldEntryPointOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createSplitDispatchFunctionPass() {
  return std::make_unique<SplitDispatchFunctionPass>();
}

static PassRegistration<SplitDispatchFunctionPass> pass(
    "iree-codegen-split-dispatch-function",
    "Split workload to multiple dispatch functions to satisfy computation "
    "dependency for GPU lowering");

}  // namespace iree_compiler
}  // namespace mlir
