// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/Range.h"

#include "iree/compiler/Dialect/Util/Analysis/Attributes/PotentialValues.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h" // TODO: Remove
#include "mlir/Dialect/Math/IR/Math.h"

#define DEBUG_TYPE "iree-util-attributes"
using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// FloatRangeStats
//===----------------------------------------------------------------------===//

static FloatRangeStats::TruncationFlag determineTruncationFlag(double value) {
  double truncValue = trunc(value);
  // TODO: I'm sure I need to be doing some ULP comparison.
  return truncValue == value ? FloatRangeStats::TRUNC
                             : FloatRangeStats::TRUNC_UNKNOWN;
}

void FloatRangeStats::addDomainValue(double value) {
  if (!valid) {
    minValue = value;
    maxValue = value;
    truncationFlag = determineTruncationFlag(value);
    valid = true;
  } else {
    minValue = std::min(minValue, value);
    maxValue = std::max(maxValue, value);
    truncationFlag =
        unionTruncationFlag(determineTruncationFlag(value), truncationFlag);
  }
}

std::string FloatRangeStats::getAsStr(AsmState &asmState) const {
  if (!valid)
    return std::string("<<INVALID>>");
  std::string s("[");
  s += std::to_string(minValue);
  s += ", ";
  s += std::to_string(maxValue);
  s += ", ";
  switch (truncationFlag) {
  case TRUNC_UNKNOWN:
    s += "!trunc";
    break;
  case TRUNC:
    s += "TRUNC";
    break;
  }
  s += "]";
  return s;
}

//===----------------------------------------------------------------------===//
// FloatRangeState
//===----------------------------------------------------------------------===//

void FloatRangeState::applyMinf(const FloatRangeStats &lhs,
                                const FloatRangeStats &rhs) {
  assumed += lhs;
  assumed += rhs;
  // Narrow the upper bound and preserve the truncation flag.
  if (assumed.valid) {
    assumed.maxValue = std::min(lhs.maxValue, rhs.maxValue);
  }
}

void FloatRangeState::applyMaxf(const FloatRangeStats &lhs,
                                const FloatRangeStats &rhs) {
  assumed += lhs;
  assumed += rhs;
  // Narrow the lower bound and preserve the truncation flag.
  if (assumed.valid) {
    assumed.minValue = std::max(lhs.minValue, rhs.minValue);
  }
}

void FloatRangeState::applyFloor(const FloatRangeStats &operand) {
  assumed += operand;
  // Apply floor to the bounds and set the truncation flag.
  if (assumed.valid) {
    assumed.minValue = std::floor(assumed.minValue);
    assumed.maxValue = std::floor(assumed.maxValue);
    assumed.truncationFlag = FloatRangeStats::TRUNC;
  }
}

//===----------------------------------------------------------------------===//
// FloatRangeValueElement
//===----------------------------------------------------------------------===//

const char FloatRangeValueElement::ID = 0;

// Returns whether the given type is a valid scalar fp type or a shaped type
// of fp types.
// TODO: getElementTypeOrSelf?
static bool isFpType(Type type) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(type));
}

void FloatRangeValueElement::initializeValue(Value value, DFX::Solver &solver) {
  if (!isFpType(value.getType())) {
    indicatePessimisticFixpoint();
    return;
  }
}

ChangeStatus FloatRangeValueElement::updateValue(Value value,
                                                 DFX::Solver &solver) {
  // TODO: This could be modularized substantially and made common with
  // other attributes.
  FloatRangeState newState = getState();

  // If this is a block argument to a supported parent operation, then
  // remap the value we are working on to the correct input from the
  // parent. This works because the statistics we are accumulating flow
  // layer-wise, and we don't need to pay particular attention to specific
  // loop structures.
  // TODO: We shouldn't need to hard switch on LinalgOp here and should
  // be relying on some kind of concept/interface. It just isn't
  // clear what that would be.
  if (auto valueBlockArg = llvm::dyn_cast<BlockArgument>(value)) {
    Block *ownerBlock = valueBlockArg.getOwner();
    if (auto linalgParent = llvm::dyn_cast_or_null<linalg::LinalgOp>(
            ownerBlock->getParentOp())) {
      value = linalgParent->getOperand(valueBlockArg.getArgNumber());
      LLVM_DEBUG(dbgs() << "  ++ REMAP LINALG BLOCK ARG TO: " << value << "\n");
    }
  }

  // If we can get a full potential value set, then we can derive from that.
  auto pvs = solver.getElementFor<ConstantAttributePVS>(
      *this, Position::forValue(value), DFX::Resolution::OPTIONAL);
  if (pvs.isValidState() && !pvs.isUndefContained()) {
    for (Attribute constValue : pvs.getAssumedSet()) {
      if (auto scalarValue = llvm::dyn_cast<FloatAttr>(constValue)) {
        FloatRangeStats stats;
        stats.addDomainValue(scalarValue.getValueAsDouble());
        newState.setAssumed(stats);
        newState.indicateOptimisticFixpoint();
      } else if (auto elements = llvm::dyn_cast<ElementsAttr>(constValue)) {
        FloatRangeStats stats;
        for (APFloat elementValue : elements.getValues<APFloat>()) {
          stats.addDomainValue(elementValue.convertToDouble());
        }
        newState.setAssumed(stats);
        LLVM_DEBUG(dbgs() << "*** COMPUTED KNOWN RANGE: "
                          << stats.getAsStr(solver.getAsmState()) << "\n");
        newState.indicateOptimisticFixpoint();
      } else {
        // Unknown.
        // TODO
        LLVM_DEBUG(dbgs() << "UNKNOWN ATTRIBUTE: " << constValue << "\n");
        newState.indicatePessimisticFixpoint();
      }
    }
  }

  if (newState.isAtFixpoint()) {
    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
        LLVM_DEBUG(dbgs() << "  WALK: " << result << "\n");
        Operation *definingOp = result.getDefiningOp();
        // TODO: We shouldn't need to hard switch on LinalgOp here and should
        // be relying on some kind of concept/interface. It just isn't
        // clear what that would be.
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(definingOp)) {
          // Because we are working on per-layer statistics, we get to
          // ignore the entire loop structure of the linalg op and just
          // chase the stats up through the terminator (which will have
          // values that match the results).
          Block *loopBody = linalgOp.getBlock();
          assert(!loopBody->empty());
          Operation &terminator = loopBody->back();
          Value loopBodyValue = terminator.getOperand(result.getResultNumber());
          auto inner = solver.getElementFor<FloatRangeValueElement>(
              *this, Position::forValue(loopBodyValue),
              DFX::Resolution::REQUIRED);
          newState ^= inner;
          // Stop traversal if tied OpOperand is not used in the op body.
          if (!linalgOp.payloadUsesValueFromOperand(
                  linalgOp.getDpsInitOperand(result.getResultNumber())))
            return WalkResult::skip();
          return WalkResult::advance();
        } else if (auto minfOp = dyn_cast<arith::MinimumFOp>(definingOp)) {
          auto lhs = solver.getElementFor<FloatRangeValueElement>(
              *this, Position::forValue(minfOp.getLhs()),
              DFX::Resolution::REQUIRED);
          auto rhs = solver.getElementFor<FloatRangeValueElement>(
              *this, Position::forValue(minfOp.getRhs()),
              DFX::Resolution::REQUIRED);

          newState.applyMinf(lhs.getAssumed(), rhs.getAssumed());
          LLVM_DEBUG(dbgs()
                     << "VISITING minf: lhs = "
                     << lhs.getAsStr(solver.getAsmState()) << ", rhs = "
                     << rhs.getAsStr(solver.getAsmState()) << " -> "
                     << newState.getAssumed().getAsStr(solver.getAsmState())
                     << "\n");
          return WalkResult::advance();
        } else if (auto maxfOp = dyn_cast<arith::MaximumFOp>(definingOp)) {
          auto lhs = solver.getElementFor<FloatRangeValueElement>(
              *this, Position::forValue(maxfOp.getLhs()),
              DFX::Resolution::REQUIRED);
          auto rhs = solver.getElementFor<FloatRangeValueElement>(
              *this, Position::forValue(maxfOp.getRhs()),
              DFX::Resolution::REQUIRED);

          newState.applyMaxf(lhs.getAssumed(), rhs.getAssumed());
          LLVM_DEBUG(dbgs()
                     << "VISITING maxf: lhs = "
                     << lhs.getAsStr(solver.getAsmState()) << ", rhs = "
                     << rhs.getAsStr(solver.getAsmState()) << " -> "
                     << newState.getAssumed().getAsStr(solver.getAsmState())
                     << "\n");
          return WalkResult::advance();
        } else if (auto floorOp = dyn_cast<math::FloorOp>(definingOp)) {
          auto operand = solver.getElementFor<FloatRangeValueElement>(
              *this, Position::forValue(floorOp.getOperand()),
              DFX::Resolution::REQUIRED);
          newState.applyFloor(operand.getAssumed());
          LLVM_DEBUG(dbgs()
                     << "VISITING floor: "
                     << operand.getAsStr(solver.getAsmState()) << " -> "
                     << newState.getAssumed().getAsStr(solver.getAsmState())
                     << "\n");
          return WalkResult::advance();
        }

        // Unrecognized op.
        LLVM_DEBUG(dbgs() << "UNRECOGNIZED OP: " << *definingOp
                          << " (signalling pessimistic fixpoint for " << value
                          << ")\n");
        newState.indicatePessimisticFixpoint();
        return WalkResult::advance();
      }) == TraversalResult::INCOMPLETE) {
    newState.indicatePessimisticFixpoint();
  }

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string FloatRangeValueElement::getAsStr(AsmState &asmState) const {
  auto range = getAssumed();
  std::string s("fp-range: ");
  s += range.getAsStr(asmState);
  return s;
}

} // namespace mlir::iree_compiler::IREE::Util
