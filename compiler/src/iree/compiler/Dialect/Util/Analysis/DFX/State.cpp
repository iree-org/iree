// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"

namespace mlir::iree_compiler {

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ChangeStatus status) {
  return os << (status == ChangeStatus::CHANGED ? "changed" : "unchanged");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DFX::AbstractState &state) {
  return os << (!state.isValidState() ? "top"
                                      : (state.isAtFixpoint() ? "fix" : ""));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DFX::PotentialConstantIntValuesState &S) {
  os << "set-state(< {";
  if (!S.isValidState()) {
    os << "full-set";
  } else {
    for (auto &it : S.getAssumedSet())
      os << it << ", ";
    if (S.isUndefContained())
      os << "undef ";
  }
  os << "} >)";
  return os;
}

} // namespace mlir::iree_compiler
