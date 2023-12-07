// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {
namespace {

//===----------------------------------------------------------------------===//
// hal.dispatch.extern
//===----------------------------------------------------------------------===//

// Converts a dispatch region op into a dispatch op to the outlined region.
static LogicalResult
convertDispatchExternToDispatchOp(IREE::HAL::DispatchExternOp dispatchExternOp,
                                  ArrayRef<Attribute> exportRefs) {
  // Insert at the same place as the original region.
  OpBuilder builder(dispatchExternOp);

  // Create the dispatch op to the executable function.
  // Note that we copy the tied operand indices from the workgroups op - it
  // lines up 1:1 with the dispatch once we've outlined things.
  auto dispatchOp = builder.create<IREE::Flow::DispatchOp>(
      dispatchExternOp.getLoc(), dispatchExternOp.getResultTypes(),
      dispatchExternOp.getWorkload(), builder.getArrayAttr(exportRefs),
      dispatchExternOp.getArguments(), dispatchExternOp.getArgumentDims(),
      dispatchExternOp.getResultDims(), dispatchExternOp.getTiedOperandsAttr());
  dispatchOp->setDialectAttrs(dispatchExternOp->getDialectAttrs());
  if (auto bindingsAttr = dispatchExternOp.getBindingsAttr()) {
    dispatchOp->setAttr("hal.interface.bindings", bindingsAttr);
  }

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < dispatchExternOp.getNumResults(); ++i) {
    dispatchExternOp.getResult(i).replaceAllUsesWith(dispatchOp.getResult(i));
  }

  return success();
}

// Outlines a dispatch region into a flow.executable and replaces the region op
// with a dispatch to that outlined executable.
static LogicalResult
outlineDispatchExternOp(std::string name,
                        IREE::HAL::DispatchExternOp dispatchExternOp) {
  // Create the executable that will contain the outlined region.
  // NOTE: this will get uniquified if we have multiple in the same block.
  auto parentFuncOp = dispatchExternOp->getParentOfType<FunctionOpInterface>();
  auto parentModuleOp = parentFuncOp->getParentOfType<mlir::ModuleOp>();
  OpBuilder parentModuleBuilder(&parentModuleOp.getBody()->back());
  auto executableOp = parentModuleBuilder.create<IREE::HAL::ExecutableOp>(
      dispatchExternOp.getLoc(), name);
  executableOp.getOperation()->moveBefore(parentFuncOp);
  executableOp.setPrivate();

  // Add one variant per object target.
  SymbolTable executableSymbolTable(executableOp);
  OpBuilder executableBuilder(executableOp.getBody());
  SmallVector<Attribute> exportRefs;
  for (auto [targetAttr, targetOrdinalAttr, targetObjectsAttr,
             targetConditionRegion] :
       llvm::zip_equal(
           dispatchExternOp.getTargetsAttr()
               .getAsRange<IREE::HAL::ExecutableTargetAttr>(),
           dispatchExternOp.getTargetOrdinalsAttr().getAsRange<IntegerAttr>(),
           dispatchExternOp.getTargetObjectsAttr().getAsRange<ArrayAttr>(),
           dispatchExternOp.getTargetRegions())) {
    // Create the variant for the given target. Note that we may have multiple
    // variants that use the same base targetAttr but have unique condition
    // regions so we rely on the symbol table for uniquing names.
    auto variantOp = executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
        dispatchExternOp.getLoc(), targetAttr.getSymbolNameFragment(),
        targetAttr);
    variantOp.setObjectsAttr(targetObjectsAttr);
    executableSymbolTable.insert(variantOp);

    // Move over optional target condition region to a condition op.
    OpBuilder variantBuilder(variantOp.getBody());
    if (!targetConditionRegion.empty()) {
      auto conditionOp =
          variantBuilder.create<IREE::HAL::ExecutableConditionOp>(
              dispatchExternOp.getLoc());
      IRMapping mapper;
      targetConditionRegion.cloneInto(&conditionOp.getBody(), mapper);
    }

    // Add an export pointing at the entry point function.
    auto exportOp = variantBuilder.create<IREE::HAL::ExecutableExportOp>(
        dispatchExternOp.getLoc(), dispatchExternOp.getExportAttr(),
        targetOrdinalAttr, dispatchExternOp.getLayoutAttr(),
        dispatchExternOp.getWorkgroupSizeAttr(),
        dispatchExternOp.getSubgroupSizeAttr(),
        dispatchExternOp.getWorkgroupLocalMemoryAttr());
    exportOp->setDialectAttrs(dispatchExternOp->getDialectAttrs());
    if (!dispatchExternOp.getWorkgroupCount().empty()) {
      IRMapping mapper;
      dispatchExternOp.getWorkgroupCount().cloneInto(
          &exportOp.getWorkgroupCount(), mapper);
    }

    exportRefs.push_back(
        SymbolRefAttr::get(executableOp.getNameAttr(),
                           {FlatSymbolRefAttr::get(variantOp.getNameAttr()),
                            FlatSymbolRefAttr::get(exportOp.getNameAttr())}));
  }

  // Finally convert the dispatch region into a dispatch to the external
  // exports.
  return convertDispatchExternToDispatchOp(dispatchExternOp, exportRefs);
}

} // namespace

class OutlineDispatchExternsPass
    : public OutlineDispatchExternsBase<OutlineDispatchExternsPass> {
public:
  OutlineDispatchExternsPass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<FunctionOpInterface>()) {
      // Outline all of the dispatch externs ops in this function.
      SmallVector<Operation *> deadOps;
      auto outlineOps = [&](Operation *op) {
        return TypeSwitch<Operation *, WalkResult>(op)
            .Case<IREE::HAL::DispatchExternOp>([&](auto dispatchExternOp) {
              if (failed(outlineDispatchExternOp(
                      ("extern_dispatch_" + llvm::Twine(deadOps.size())).str(),
                      dispatchExternOp))) {
                return WalkResult::interrupt();
              }
              deadOps.push_back(op);
              return WalkResult::advance();
            })
            .Default(WalkResult::advance());
      };
      if (funcOp.walk(outlineOps).wasInterrupted())
        return signalPassFailure();
      for (auto *deadOp : deadOps)
        deadOp->erase();
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineDispatchExternsPass() {
  return std::make_unique<OutlineDispatchExternsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
