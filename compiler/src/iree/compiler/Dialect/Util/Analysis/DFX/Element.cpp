// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"

#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-util-dfx"

namespace mlir::iree_compiler::DFX {

ChangeStatus AbstractElement::update(Solver &solver) {
  ChangeStatus changeStatus = ChangeStatus::UNCHANGED;
  if (getState().isAtFixpoint())
    return changeStatus;

  LLVM_DEBUG({
    llvm::dbgs() << "[Solver] updating: ";
    print(llvm::dbgs(), solver.getAsmState());
    llvm::dbgs() << "\n";
  });

  changeStatus = updateImpl(solver);

  LLVM_DEBUG({
    llvm::dbgs() << "[Solver] update " << changeStatus << " ";
    print(llvm::dbgs(), solver.getAsmState());
    llvm::dbgs() << "\n";
  });

  return changeStatus;
}

void AbstractElement::print(llvm::raw_ostream &os, AsmState &asmState) const {
  os << "[";
  os << getName();
  os << "] ";

  if (isValue()) {
    os << "value ";
    getValue().printAsOperand(os, asmState);
  } else if (isReturnedValue()) {
    auto returnedValue = getReturnedValue();
    os << "returned value " << returnedValue.second << " for ";
    returnedValue.first->print(os, asmState);
  } else if (isOperation()) {
    os << "op ";
    getOperation().print(os, asmState);
  } else if (isBlock()) {
    os << "block (TBD)";
  } else {
    os << "<<null>>";
  }

  os << " with state " << getAsStr(asmState);
}

void AbstractElement::printWithDeps(llvm::raw_ostream &os,
                                    AsmState &asmState) const {
  print(os, asmState);
  for (const auto &depElement : deps) {
    auto *element = depElement.getPointer();
    os << "\n  updates ";
    element->print(os, asmState);
  }
  os << '\n';
}

void AbstractElement::dump(AsmState &asmState) const {
  print(llvm::dbgs(), asmState);
}

} // namespace mlir::iree_compiler::DFX
