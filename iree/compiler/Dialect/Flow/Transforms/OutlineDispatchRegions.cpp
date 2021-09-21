// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {
namespace {

// Creates a flow.executable out of a set of functions, pulling in all other
// functions reachable by the provided functions.
static ExecutableOp createExecutable(Location loc, StringRef executableName,
                                     ArrayRef<mlir::FuncOp> funcOps,
                                     ModuleOp parentModuleOp) {
  assert(!funcOps.empty() && "must have at least one entry function");

  // Create the executable that will contain the outlined region.
  // NOTE: this will get uniquified if we have multiple in the same block.
  OpBuilder parentModuleBuilder(&parentModuleOp.getBody()->back());
  auto executableOp =
      parentModuleBuilder.create<IREE::Flow::ExecutableOp>(loc, executableName);

  // Create the inner ModuleOp that contains the original functions. We need
  // to provide this shim as some ops (like std.call) look for the
  // containing module to provide symbol resolution.
  OpBuilder executableBuilder(executableOp);
  executableBuilder.setInsertionPointToStart(&executableOp.getBlock());
  auto innerModule = executableBuilder.create<mlir::ModuleOp>(loc);
  for (auto funcOp : funcOps) {
    innerModule.push_back(funcOp);
  }

  // Copy all reachable functions into the executable.
  // Linker passes may dedupe these later on.
  OpBuilder innerModuleBuilder = OpBuilder::atBlockEnd(innerModule.getBody());
  innerModuleBuilder.setInsertionPoint(innerModule.getBody(),
                                       ++innerModule.getBody()->begin());

  return executableOp;
}

// Converts a dispatch region op into a dispatch op to the outlined region.
static LogicalResult convertToDispatchOp(DispatchWorkgroupsOp regionOp,
                                         ExecutableOp executableOp,
                                         DispatchEntryOp entryPointOp) {
  // Insert at the same place as the original region.
  OpBuilder builder(regionOp);

  // Perform shape to primitive type expansion.
  // NOTE: this may insert new shape values at |builder|, which is prior to
  // our dispatch operation. All new values that are built can only depend
  // on SSA values that are defined prior to the region op.
  SmallVector<Value, 4> newOperands;
  SmallVector<Value, 4> operandDynamicDims;
  SmallVector<Value, 4> resultDynamicDims;
  for (auto operand : regionOp.operands()) {
    newOperands.push_back(operand);
  }
  for (auto operand : regionOp.operands()) {
    if (operand.getType().isa<ShapedType>()) {
      auto dynamicDims = Shape::buildOrFindDynamicDimsForValue(
          regionOp.getLoc(), operand, builder);
      operandDynamicDims.append(dynamicDims);
      newOperands.append(dynamicDims);
    }
  }
  for (auto result : regionOp.results()) {
    if (result.getType().isa<ShapedType>()) {
      auto dynamicDims = Shape::buildOrFindDynamicDimsForValue(
          regionOp.getLoc(), result, builder);
      resultDynamicDims.append(dynamicDims);
      newOperands.append(dynamicDims);
    }
  }

  // Create the dispatch op to the executable function.
  // Note that we copy the tied operand indices from the workgroups op - it
  // lines up 1:1 with the dispatch once we've outlined things.
  auto dispatchOp = builder.create<DispatchOp>(
      regionOp.getLoc(), entryPointOp, regionOp.workgroup_count(),
      regionOp.getResultTypes(), resultDynamicDims, newOperands,
      operandDynamicDims, regionOp.tied_operandsAttr());

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i).replaceAllUsesWith(dispatchOp.getResult(i));
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

// Converts a dispatch region body to a free-floating function.
// The contents of the function will be updated to propagate shape information
// across the function call boundary and ensure we have all the metadata we need
// on the inside in order to manipulate dynamic shapes.
static mlir::FuncOp createWorkgroupFunc(Location loc, StringRef functionName,
                                        Region &region) {
  // Build function type matching the region signature + the dynamic dims.
  //
  // At this stage we'll insert all dynamic dimension values even if some are
  // duplicates (same value providing the dynamic dimension); we need
  // canonicalization/CSE/etc to run before we can dedupe them.
  SmallVector<Type, 4> operandTypes;
  int64_t totalDynamicDims = 0;
  for (auto &operand : region.getArguments()) {
    if (auto tensorType = operand.getType().dyn_cast<DispatchTensorType>()) {
      operandTypes.push_back(tensorType);
      totalDynamicDims += tensorType.getNumDynamicDims();
    } else {
      // Pass-through.
      operandTypes.push_back(operand.getType());
    }
  }
  auto indexType = IndexType::get(region.getContext());
  for (int64_t i = 0; i < totalDynamicDims; ++i) {
    operandTypes.push_back(indexType);
  }
  auto functionType =
      FunctionType::get(region.getContext(), operandTypes, /*results=*/{});

  // Clone region into the function body.
  auto funcOp = mlir::FuncOp::create(loc, functionName, functionType);
  BlockAndValueMapping mapping;
  region.cloneInto(&funcOp.getBody(), mapping);
  auto *entryBlock = &funcOp.getBody().front();
  for (int64_t i = 0; i < totalDynamicDims; ++i) {
    entryBlock->addArgument(indexType);
  }

  // Insert the shapes for each I/O and tie them together.
  unsigned int dynamicDimArgIndex = region.getNumArguments();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
  for (auto &oldOperand : region.getArguments()) {
    if (auto ioType = oldOperand.getType().dyn_cast<DispatchTensorType>()) {
      // Avoid shape tie noise from fully-static shapes.
      if (ioType.hasStaticShape()) continue;

      // Create the ranked shape type from the dynamic dimension arguments.
      SmallVector<Value, 4> dimValues;
      dimValues.reserve(ioType.getNumDynamicDims());
      for (int64_t i = 0; i < ioType.getNumDynamicDims(); ++i) {
        dimValues.push_back(entryBlock->getArgument(dynamicDimArgIndex++));
      }
      auto shapeOp = entryBuilder.create<Shape::MakeRankedShapeOp>(
          funcOp.getLoc(), ioType.asRankedShapeType(), dimValues);

      // Tie shape to the I/O argument and fix up SSA usage to the tied value.
      auto operand = mapping.lookup<Value>(oldOperand);
      auto tieOp = entryBuilder.create<DispatchTieShapeOp>(
          funcOp.getLoc(), operand.getType(), operand, shapeOp.getResult());
      SmallPtrSet<Operation *, 1> tieOpSet = {tieOp};
      operand.replaceAllUsesExcept(tieOp.result(), tieOpSet);
    }
  }

  // Replace flow.return with std.return.
  // NOTE: in the dispatch workgroups case the return should have no values.
  for (auto &block : funcOp.getBlocks()) {
    if (auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(block.back())) {
      OpBuilder builder(returnOp);
      builder.create<mlir::ReturnOp>(
          returnOp.getLoc(), llvm::to_vector<4>(returnOp.getOperands()));
      returnOp.erase();
    }
  }

  return funcOp;
}

// Outlines a dispatch region into a flow.executable and replaces the region op
// with a dispatch to that outlined executable.
static LogicalResult outlineDispatchWorkgroupsOp(
    std::string namePrefix, DispatchWorkgroupsOp regionOp) {
  // Convert the region to a free-floating function.
  auto workgroupFuncOp =
      createWorkgroupFunc(regionOp.getLoc(), namePrefix, regionOp.body());
  if (!workgroupFuncOp) {
    return failure();
  }

  // Create the executable with the region cloned into it.
  auto parentFuncOp = regionOp->getParentOfType<mlir::FuncOp>();
  auto executableOp =
      createExecutable(regionOp.getLoc(), namePrefix, {workgroupFuncOp},
                       parentFuncOp->getParentOfType<mlir::ModuleOp>());
  executableOp.getOperation()->moveBefore(parentFuncOp);
  executableOp.setPrivate();

  // Add executable entry point pointing at the function.
  OpBuilder builder(executableOp.body());
  auto entryPointOp = builder.create<DispatchEntryOp>(
      regionOp.getLoc(), workgroupFuncOp.getName(),
      SymbolRefAttr::get(workgroupFuncOp),
      builder.getIndexAttr(regionOp.getWorkgroupRank()));

  // Finally convert the dispatch region into a dispatch to the outlined func.
  return convertToDispatchOp(regionOp, executableOp, entryPointOp);
}

}  // namespace

class OutlineDispatchRegionsPass
    : public OutlineDispatchRegionsBase<OutlineDispatchRegionsPass> {
 public:
  OutlineDispatchRegionsPass() = default;

  void runOnOperation() override {
    // Convert each dispatch region into a flow.executable + dispatch op.
    for (auto funcOp : getOperation().getOps<mlir::FuncOp>()) {
      // Outline all of the dispatch regions ops in this function.
      auto dispatchWorkgroupsOps =
          llvm::to_vector<8>(funcOp.getOps<DispatchWorkgroupsOp>());
      for (int i = 0; i < dispatchWorkgroupsOps.size(); ++i) {
        std::string namePrefix =
            funcOp.getName().str() + "_dispatch_" + std::to_string(i);
        if (failed(outlineDispatchWorkgroupsOp(namePrefix,
                                               dispatchWorkgroupsOps[i]))) {
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineDispatchRegionsPass() {
  return std::make_unique<OutlineDispatchRegionsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
