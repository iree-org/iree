// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_ANNOTATETARGETDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-annotate-target-devices
//===----------------------------------------------------------------------===//

// Sorts |attrs| in lexigraphical order.
// We have to do this as the PVS elements we source from are unsorted.
static void sortAttributes(SmallVectorImpl<Attribute> &attrs) {
  if (attrs.size() <= 1) {
    return;
  }
  llvm::stable_sort(attrs, [](Attribute lhs, Attribute rhs) {
    std::string lhsStr;
    llvm::raw_string_ostream lhsStream(lhsStr);
    lhs.print(lhsStream);
    std::string rhsStr;
    llvm::raw_string_ostream rhsStream(rhsStr);
    rhs.print(rhsStream);
    return lhsStr < rhsStr;
  });
}

static ArrayAttr
getDeviceSetAttr(MLIRContext *context,
                 std::optional<IREE::HAL::DeviceSet> deviceSet) {
  if (!deviceSet.has_value() || deviceSet->empty()) {
    return ArrayAttr::get(context, {});
  }
  SmallVector<Attribute> targetAttrs;
  llvm::append_range(targetAttrs, deviceSet->getValues());
  sortAttributes(targetAttrs);
  return ArrayAttr::get(context, targetAttrs);
}

static void annotateGlobalOp(IREE::Util::GlobalOpInterface globalOp,
                             DeviceAnalysis &deviceAnalysis) {
  if (isa<IREE::HAL::DeviceType>(globalOp.getGlobalType())) {
    globalOp->setAttr(
        "hal.device.targets",
        getDeviceSetAttr(globalOp.getContext(),
                         deviceAnalysis.lookupDeviceTargets(globalOp)));
  }
}

static void annotateOperandsAndResults(Operation *op,
                                       DeviceAnalysis &deviceAnalysis) {
  SmallVector<Attribute> operandAttrs;
  for (auto operand : op->getOperands()) {
    if (isa<IREE::HAL::DeviceType>(operand.getType())) {
      operandAttrs.push_back(getDeviceSetAttr(
          op->getContext(), deviceAnalysis.lookupDeviceTargets(operand)));
    }
  }
  SmallVector<Attribute> resultAttrs;
  for (auto result : op->getResults()) {
    if (isa<IREE::HAL::DeviceType>(result.getType())) {
      resultAttrs.push_back(getDeviceSetAttr(
          op->getContext(), deviceAnalysis.lookupDeviceTargets(result)));
    }
  }
  if (!operandAttrs.empty()) {
    op->setAttr("hal.device.targets.operands",
                ArrayAttr::get(op->getContext(), operandAttrs));
  }
  if (!resultAttrs.empty()) {
    op->setAttr("hal.device.targets.results",
                ArrayAttr::get(op->getContext(), resultAttrs));
  }
}

static void annotateFuncOp(FunctionOpInterface funcOp,
                           DeviceAnalysis &deviceAnalysis) {
  if (funcOp.empty())
    return;
  for (auto arg : funcOp.front().getArguments()) {
    if (isa<IREE::HAL::DeviceType>(arg.getType())) {
      funcOp.setArgAttr(
          arg.getArgNumber(), "hal.device.targets",
          getDeviceSetAttr(funcOp.getContext(),
                           deviceAnalysis.lookupDeviceTargets(arg)));
    }
  }
  funcOp.walk(
      [&](Operation *op) { annotateOperandsAndResults(op, deviceAnalysis); });
}

struct AnnotateTargetDevicesPass
    : public IREE::HAL::impl::AnnotateTargetDevicesPassBase<
          AnnotateTargetDevicesPass> {
  void runOnOperation() override {
    // Run device analysis on the whole module.
    DeviceAnalysis deviceAnalysis(getOperation());
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Annotate all ops with derived affinities.
    for (auto &op : getOperation().getOps()) {
      if (op.hasTrait<OpTrait::IREE::Util::ObjectLike>())
        continue;
      if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(op)) {
        annotateGlobalOp(globalOp, deviceAnalysis);
      } else if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        annotateFuncOp(funcOp, deviceAnalysis);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
