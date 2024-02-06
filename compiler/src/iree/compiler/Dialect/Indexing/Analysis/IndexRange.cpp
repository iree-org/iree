// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Indexing/Analysis/IndexRange.h"

#include "iree/compiler/Dialect/Indexing/IR/IndexingInterfaces.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/DepGraph.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Position.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#define DEBUG_TYPE "iree-indexing-range"
using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Indexing {

static void
populateOperandBounds(DFX::Solver &solver,
                      const DFX::AbstractElement *queryingElement,
                      Operation *boundingOp,
                      SmallVectorImpl<SaturatedValueRange> &operandRanges) {
  for (auto operand : boundingOp->getOperands()) {
    if (operand.getType().isIntOrIndex()) {
      auto &rangeElement = solver.getOrCreateElementFor<IndexRangeValueElement>(
          Position::forValue(operand), queryingElement,
          DFX::Resolution::OPTIONAL);
      if (rangeElement.getState().isValidState()) {
        operandRanges.push_back(rangeElement.getState().getAssumed());
      } else {
        operandRanges.push_back({});
      }
    } else if (isa<ShapedType>(operand.getType())) {
      auto &rangeElement =
          solver.getOrCreateElementFor<ShapedDimsRangeValueElement>(
              Position::forValue(operand), queryingElement,
              DFX::Resolution::OPTIONAL);
      if (rangeElement.getState().isValidState()) {
        operandRanges.push_back(rangeElement.getState().getAssumed());
      } else {
        operandRanges.push_back({});
      }
    } else {
      operandRanges.push_back({});
    }
  }
}

//===----------------------------------------------------------------------===//
// IndexRangeState
//===----------------------------------------------------------------------===//

const std::string IndexRangeState::getAsStr(AsmState &asmState) const {
  auto range = getAssumed();
  if (!valid) {
    return "<INVALID>";
  }
  std::string s = range.getAsStr();
  return s;
}

//===----------------------------------------------------------------------===//
// IndexRangeValueElement
//===----------------------------------------------------------------------===//

const char IndexRangeValueElement::ID = 0;

void IndexRangeValueElement::initializeValue(Value value, DFX::Solver &solver) {
  auto exit = llvm::make_scope_exit([&]() {
    LLVM_DEBUG({
      llvm::dbgs() << "initializing index value to " << value << ": "
                   << getAsStr(solver.getAsmState()) << "\n";
    });
  });
  if (!value.getType().isIntOrIndex()) {
    indicatePessimisticFixpoint();
    return;
  }
  if (auto boundingOp = value.getDefiningOp<StaticBoundsOpInterface>()) {
    bool isFixed = false;
    std::optional<SaturatedIndexRange> maybeRange =
        boundingOp.initializeRange(value, isFixed);
    if (maybeRange) {
      setAssumed(*maybeRange);
      if (isFixed) {
        indicateOptimisticFixpoint();
      }
      return;
    }
  }
  setWidestIndexRange();
}

ChangeStatus IndexRangeValueElement::updateValue(Value value,
                                                 DFX::Solver &solver) {
  IndexRangeState newState = getState();

  if (auto valueBlockArg = llvm::dyn_cast<BlockArgument>(value)) {
    bool isFirstIncomingArg = true;
    solver.getExplorer().walkIncomingBlockArgument(
        valueBlockArg, [&](Block *block, Value producer) {
          auto &producerElement =
              solver.getOrCreateElementFor<IndexRangeValueElement>(
                  Position::forValue(producer), *this,
                  DFX::Resolution::REQUIRED);
          if (isFirstIncomingArg) {
            newState = producerElement.getState();
            isFirstIncomingArg = false;
          } else {
            newState += producerElement.getState();
          }
          return WalkResult::advance();
        });

    LLVM_DEBUG({
      llvm::dbgs() << "Setting index range for block argument " << value << ": "
                   << newState.getAsStr(solver.getAsmState()) << "\n";
    });

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  Operation *definingOp = value.getDefiningOp();
  if (auto boundingOp = dyn_cast<StaticBoundsOpInterface>(definingOp)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Found bounding producer: ";
      boundingOp.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    SmallVector<SaturatedValueRange> operandRanges;
    populateOperandBounds(solver, *this, boundingOp, operandRanges);
    SaturatedIndexRange newRange =
        boundingOp.getIndexRange(value, operandRanges);
    newState.setAssumed(newRange);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Setting range for scalar value " << value << ": "
                 << newState.getAsStr(solver.getAsmState()) << "\n";
  });

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string IndexRangeValueElement::getAsStr(AsmState &asmState) const {
  auto range = getState();
  std::string s("index-range: ");
  s += range.getAsStr(asmState);
  return s;
}

//===----------------------------------------------------------------------===//
// ShapedDimsRangeState
//===----------------------------------------------------------------------===//

const std::string ShapedDimsRangeState::getAsStr(AsmState &asmState) const {
  auto range = getAssumed();
  if (!valid || range.empty()) {
    return "<INVALID>";
  }
  std::string s;
  for (int i = 0, e = range.size() - 1; i < e; ++i) {
    s += range[i].getAsStr();
    s += ", ";
  }
  s += range.back().getAsStr();
  return s;
}

//===----------------------------------------------------------------------===//
// ShapedDimsRangeValueElement
//===----------------------------------------------------------------------===//

const char ShapedDimsRangeValueElement::ID = 1;

void ShapedDimsRangeValueElement::initializeValue(Value value,
                                                  DFX::Solver &solver) {
  auto exit = llvm::make_scope_exit([&]() {
    LLVM_DEBUG({
      llvm::dbgs() << "initializing shaped value to " << value << ": "
                   << getAsStr(solver.getAsmState()) << "\n";
    });
  });
  if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
    if (shapedType.hasStaticShape()) {
      indicatePessimisticFixpoint();
      return;
    }
    setWidestAssumedDynamicDims(shapedType);
    return;
  }
  indicatePessimisticFixpoint();
}

ChangeStatus ShapedDimsRangeValueElement::updateValue(Value value,
                                                      DFX::Solver &solver) {
  ShapedDimsRangeState newState = getState();

  if (auto valueBlockArg = llvm::dyn_cast<BlockArgument>(value)) {
    bool isFirstIncomingArg = true;
    solver.getExplorer().walkIncomingBlockArgument(
        valueBlockArg, [&](Block *block, Value producer) {
          auto &producerElement =
              solver.getOrCreateElementFor<ShapedDimsRangeValueElement>(
                  Position::forValue(producer), *this,
                  DFX::Resolution::REQUIRED);
          if (isFirstIncomingArg) {
            newState = producerElement.getState();
            isFirstIncomingArg = false;
          } else {
            newState += producerElement.getState();
          }
          return WalkResult::advance();
        });

    LLVM_DEBUG({
      llvm::dbgs() << "Setting shaped dims range for block argument " << value
                   << ": " << newState.getAsStr(solver.getAsmState()) << "\n";
    });

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  Operation *definingOp = value.getDefiningOp();
  if (auto boundingOp = dyn_cast<StaticBoundsOpInterface>(definingOp)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Found bounding producer: ";
      boundingOp.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    SmallVector<SaturatedValueRange> operandRanges;
    populateOperandBounds(solver, *this, boundingOp, operandRanges);
    SaturatedIndexRangeList newRanges;
    newRanges = boundingOp.getDynamicDimRanges(value, operandRanges);
    newState.setAssumed(newRanges);
  }

  // Destination style ops have tied operands that we can try to incorporate
  // additional information from.
  if (auto destinationOp = dyn_cast<DestinationStyleOpInterface>(definingOp)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Found destination passing producer: ";
      destinationOp.print(llvm::dbgs(), solver.getAsmState());
      llvm::dbgs() << "\n";
    });
    auto rangeElement =
        solver.getOrCreateElementFor<ShapedDimsRangeValueElement>(
            Position::forValue(
                destinationOp.getTiedOpOperand(cast<OpResult>(value))->get()),
            *this, DFX::Resolution::OPTIONAL);
    newState ^= rangeElement.getState();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Setting range for shaped value " << value << ": "
                 << newState.getAsStr(solver.getAsmState()) << "\n";
  });

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string
ShapedDimsRangeValueElement::getAsStr(AsmState &asmState) const {
  auto range = getState();
  std::string s("index-range: ");
  s += range.getAsStr(asmState);
  return s;
}

} // namespace mlir::iree_compiler::IREE::Indexing
