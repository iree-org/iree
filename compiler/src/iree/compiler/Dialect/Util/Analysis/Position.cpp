// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Position.h"

namespace mlir::iree_compiler {

// static
const Position Position::EmptyKey(ENC_BLOCK,
                                  llvm::DenseMapInfo<void *>::getEmptyKey(), 0);

// static
const Position
    Position::TombstoneKey(ENC_BLOCK,
                           llvm::DenseMapInfo<void *>::getTombstoneKey(), 0);

void Position::print(llvm::raw_ostream &os) const {
  if (*this == Position::EmptyKey) {
    os << "(empty)";
  } else if (*this == Position::TombstoneKey) {
    os << "(tombstone)";
  } else {
    // Suboptimal printing, but it's not worth instantiating an AsmState.
    // Use the print(os, asmState) version instead of <<.
    switch (enc.getInt()) {
    case Position::ENC_VALUE:
      os << "value";
      break;
    case Position::ENC_RETURNED_VALUE:
      os << "returned value";
      break;
    case Position::ENC_OPERATION: {
      auto symbol = dyn_cast<mlir::SymbolOpInterface>(getOperation());
      os << "op "
         << (symbol ? symbol.getName()
                    : getOperation().getName().getStringRef());
      break;
    }
    case Position::ENC_BLOCK:
      os << "block";
      break;
    }
  }
}

void Position::print(llvm::raw_ostream &os, AsmState &asmState) const {
  if (*this == Position::EmptyKey) {
    os << "(empty)";
  } else if (*this == Position::TombstoneKey) {
    os << "(tombstone)";
  } else {
    switch (enc.getInt()) {
    case Position::ENC_VALUE: {
      getValue().printAsOperand(os, asmState);
      break;
    }
    case Position::ENC_RETURNED_VALUE: {
      auto returnedValue = getReturnedValue();
      os << returnedValue.first->getName().getStringRef();
      auto symbol = dyn_cast<mlir::SymbolOpInterface>(returnedValue.first);
      if (symbol) {
        os << " @" << symbol.getName();
      }
      os << " result " << returnedValue.second;
      break;
    }
    case Position::ENC_OPERATION: {
      os << getOperation().getName().getStringRef();
      auto symbol = dyn_cast<mlir::SymbolOpInterface>(getOperation());
      if (symbol) {
        os << " @" << symbol.getName();
      }
      break;
    }
    case Position::ENC_BLOCK: {
      getBlock().printAsOperand(os, asmState);
      break;
    }
    }
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Position pos) {
  pos.print(os);
  return os;
}

SmallVector<Position> getReturnedValuePositions(Region &region) {
  // Ask the op directly for the results, if possible.
  // This works for closure-like ops that have results in their parent scope.
  if (auto regionOp = dyn_cast<RegionBranchOpInterface>(region.getParentOp())) {
    SmallVector<RegionSuccessor> successors;
    regionOp.getSuccessorRegions(region, successors);
    for (auto &successor : successors) {
      if (successor.isParent()) {
        return llvm::to_vector(getPositions(successor.getSuccessorInputs()));
      }
    }
    assert(false && "should have found a parent successor");
  }

  // Callable ops use synthetic return values as they may have any number of
  // exits and call sites.
  auto *parentOp = region.getParentOp();
  if (auto callableOp = dyn_cast<CallableOpInterface>(parentOp)) {
    unsigned resultCount = callableOp.getResultTypes().size();
    return llvm::map_to_vector(
        llvm::seq(0u, resultCount), [parentOp](unsigned resultIdx) -> Position {
          return Position::forReturnedValue(parentOp, resultIdx);
        });
  }

  // None? Probably want to ensure this doesn't happen.
  return {};
}

} // namespace mlir::iree_compiler
