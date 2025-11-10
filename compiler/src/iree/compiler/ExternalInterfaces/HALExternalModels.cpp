// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/HALExternalModels.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// mlir::ValueBoundsOpInterface
//===----------------------------------------------------------------------===//

template <typename IDOp>
struct IDOpValueBoundsInterface : public ValueBoundsOpInterface::ExternalModel<
                                      IDOpValueBoundsInterface<IDOp>, IDOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto boundOp = cast<IDOp>(op);
    assert(value == boundOp.getResult() && "value must be op result");
    cstr.bound(value) >= 0;
    if (boundOp.getUpperBound()) {
      cstr.bound(value) < boundOp.getUpperBound()->getSExtValue();
    }
  }
};

template <typename CountOp>
struct CountOpValueBoundsInterface
    : public ValueBoundsOpInterface::ExternalModel<
          CountOpValueBoundsInterface<CountOp>, CountOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto boundOp = cast<CountOp>(op);
    assert(value == boundOp.getResult() && "value must be op result");
    cstr.bound(value) >= 1;
    if (boundOp.getUpperBound()) {
      cstr.bound(value) <= boundOp.getUpperBound()->getSExtValue();
    }
  }
};

//===----------------------------------------------------------------------===//
// IREE::Stream::TimelineAwareOpInterface
//===----------------------------------------------------------------------===//

// External model to make util.call ops with fence arguments timeline-aware.
struct TimelineAwareCallExternalModel
    : public IREE::Stream::TimelineAwareOpInterface::ExternalModel<
          TimelineAwareCallExternalModel, IREE::Util::CallOp> {
  static void add(MLIRContext *context) {
    IREE::Util::CallOp::attachInterface<TimelineAwareCallExternalModel>(
        *context);
  }

  // Returns that we participate if we have coarse-fences model with fences.
  bool participatesInTimeline(Operation *op) const {
    auto callOp = cast<IREE::Util::CallOp>(op);

    // Check for HAL ABI convention attribute.
    if (IREE::HAL::ABIConventionAttr::lookupExecutionModel(callOp) !=
        IREE::HAL::ExecutionModel::CoarseFences) {
      return false;
    }

    // Must have fence operands or results to participate.
    for (auto type : callOp.getOperandTypes()) {
      if (isa<IREE::HAL::FenceType>(type)) {
        return true;
      }
    }
    for (auto type : callOp.getResultTypes()) {
      if (isa<IREE::HAL::FenceType>(type)) {
        return true;
      }
    }

    return false;
  }

  // Returns fence operands that will be awaited (excludes signal fence).
  SmallVector<Value> getAwaitFences(Operation *op) const {
    auto callOp = cast<IREE::Util::CallOp>(op);

    // First identify the signal fence.
    Value signalFence = getSignalFence(op);

    // Return all fence operands EXCEPT the signal fence.
    SmallVector<Value> fences;
    for (auto operand : callOp.getOperands()) {
      if (isa<IREE::HAL::FenceType>(operand.getType()) &&
          operand != signalFence) {
        fences.push_back(operand);
      }
    }
    return fences;
  }

  // Returns the signal fence (last fence operand or fence result).
  //
  // TODO(benvanik): better handle wait/signal fences - today we cannot tell
  // what a fence is for and go by convention: last operand if no result, result
  // otherwise. We should really have a way to associate the fences with
  // operands/results and whether they are waits or signals (possibly by having
  // a type parameter like !hal.fence<signal> vs !hal.fence<wait>).
  Value getSignalFence(Operation *op) const {
    auto callOp = cast<IREE::Util::CallOp>(op);

    // Check operands first (in/out fence pattern) - use last fence.
    Value signalFence;
    for (auto operand : callOp.getOperands()) {
      if (isa<IREE::HAL::FenceType>(operand.getType())) {
        signalFence = operand;
      }
    }

    // If no operand fence or if there's a result fence, prefer result.
    for (auto result : callOp.getResults()) {
      if (isa<IREE::HAL::FenceType>(result.getType())) {
        return result;
      }
    }

    return signalFence;
  }

  // Looks for preceding stream.timepoint.import ops for fence operands.
  // We check if imports already exist to avoid duplicates (CSE will handle).
  SmallVector<Value> buildAwaitTimepoints(Operation *op,
                                          OpBuilder &builder) const {
    auto callOp = cast<IREE::Util::CallOp>(op);
    SmallVector<Value> timepoints;
    for (auto operand : callOp.getOperands()) {
      if (!isa<IREE::HAL::FenceType>(operand.getType())) {
        continue;
      }

      // Look for existing import.
      // TODO(benvanik): RegionBranchOpInterface-aware detection (blocks
      // differ).
      IREE::Stream::TimepointImportOp existingImport;
      for (auto *user : operand.getUsers()) {
        if (auto importOp = dyn_cast<IREE::Stream::TimepointImportOp>(user)) {
          if (importOp->getBlock() == callOp->getBlock()) {
            assert(importOp->isBeforeInBlock(callOp) &&
                   "SSA dominance required");
            existingImport = importOp;
            break;
          }
        }
      }
      if (existingImport) {
        timepoints.push_back(existingImport.getResultTimepoint());
      } else {
        // Create new import before the op.
        auto importOp = IREE::Stream::TimepointImportOp::create(
            builder, callOp.getLoc(),
            builder.getType<IREE::Stream::TimepointType>(),
            ValueRange{operand});
        timepoints.push_back(importOp.getResultTimepoint());
      }
    }
    return timepoints;
  }

  // Looks for a signal fence in operands (common pattern for imported funcs).
  // Heuristic: last fence argument is typically the signal fence.
  Value buildResultTimepoint(Operation *op, OpBuilder &builder) const {
    auto callOp = cast<IREE::Util::CallOp>(op);

    Value signalFence;
    for (auto operand : llvm::reverse(callOp.getOperands())) {
      if (isa<IREE::HAL::FenceType>(operand.getType())) {
        signalFence = operand;
        break;
      }
    }
    if (!signalFence) {
      for (auto result : callOp.getResults()) {
        if (isa<IREE::HAL::FenceType>(result.getType())) {
          signalFence = result;
          break;
        }
      }
    }
    if (!signalFence) {
      return {};
    }

    // Check if import already exists.
    // TODO(benvanik): RegionBranchOpInterface-aware detection (blocks differ).
    for (auto *user : signalFence.getUsers()) {
      if (auto importOp = dyn_cast<IREE::Stream::TimepointImportOp>(user)) {
        if (importOp->getBlock() == callOp->getBlock()) {
          assert(callOp->isBeforeInBlock(importOp) && "SSA dominance required");
          return importOp.getResultTimepoint();
        }
      }
    }

    // Create new import after the call.
    auto importOp = IREE::Stream::TimepointImportOp::create(
        builder, callOp.getLoc(),
        builder.getType<IREE::Stream::TimepointType>(),
        ValueRange{signalFence});
    return importOp.getResultTimepoint();
  }
};

} // namespace

void registerHALExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            IREE::HAL::HALDialect *dialect) {
    IREE::HAL::InterfaceWorkgroupIDOp::attachInterface<
        IDOpValueBoundsInterface<IREE::HAL::InterfaceWorkgroupIDOp>>(*context);
    IREE::HAL::InterfaceWorkgroupSizeOp::attachInterface<
        CountOpValueBoundsInterface<IREE::HAL::InterfaceWorkgroupSizeOp>>(
        *context);
    IREE::HAL::InterfaceWorkgroupCountOp::attachInterface<
        CountOpValueBoundsInterface<IREE::HAL::InterfaceWorkgroupCountOp>>(
        *context);
  });

  registry.insert<IREE::Util::UtilDialect>();
  registry.addExtension(
      +[](MLIRContext *context, IREE::Util::UtilDialect *dialect) {
        TimelineAwareCallExternalModel::add(context);
      });
}

} // namespace mlir::iree_compiler
