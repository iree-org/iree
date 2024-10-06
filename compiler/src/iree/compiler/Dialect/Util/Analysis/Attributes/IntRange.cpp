// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/IntRange.h"

#include "iree/compiler/Dialect/Util/Analysis/Attributes/PotentialValues.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <limits>

#define DEBUG_TYPE "iree-util-attributes"
using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

// ---------------------------------------------------------------------------//
// GcdState
// ---------------------------------------------------------------------------//

const char ValueGcd::ID = 0;

void ValueGcd::initializeValue(Value value, DFX::Solver &solver) {
  llvm::dbgs() << "INITIALIZE_VALUE: " << value << "\n";
  if (!value.getType().isIndex()) {
    indicatePessimisticFixpoint();
    return;
  }
}

ChangeStatus ValueGcd::updateValue(Value value, DFX::Solver &solver) {
  llvm::dbgs() << "UPDATE_VALUE: " << value << "\n";
  StateType newState = getState();

  if (!newState.isAtFixpoint()) {
    // Scan IR to see if we can infer the gcd.
    if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          // Skip ops.
          if (isa<IREE::Util::AssumeRangeOp>(result.getDefiningOp())) {
            return WalkResult::advance();
          }

          // Handle known ops.
          if (auto assumeDivisibleOp = dyn_cast<IREE::Util::AssumeDivisibleOp>(
                  result.getDefiningOp())) {
            auto divisor = assumeDivisibleOp.getDivisor().getZExtValue();
            newState.takeAssumedMinimum(divisor);
            newState.indicateOptimisticFixpoint();
            return WalkResult::interrupt();
          } else {
            newState.indicatePessimisticFixpoint();
            return WalkResult::interrupt();
          }
        }) == TraversalResult::INCOMPLETE) {
      newState.indicatePessimisticFixpoint();
    }
  }

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string ValueGcd::getAsStr(AsmState &asmState) const {
  std::string s("gcd: ");
  if (getAssumed() == GcdStateType::UNDEF) {
    s += "UNDEF";
  } else {
    s += std::to_string(getAssumed());
  }
  return s;
}

// bool IntegerValueInfo::operator<=(const IntegerValueInfo rhs) const {
//   if (negative) {
//     // lhs negative.
//     if (out_of_range)
//       return true;
//     if (rhs.negative) {
//       // rhs negative.
//       if (out_of_range) {
//         return rhs.out_of_range;
//       } else if (rhs.out_of_range) {
//         return false;
//       } else {
//         return magnitude >= rhs.magnitude;
//       }
//     } else {
//       // rhs positive.
//       return true;
//     }
//   } else {
//     // lhs positive.
//     if (out_of_range)
//       return !rhs.negative && rhs.out_of_range;
//     if (rhs.negative) {
//       // rhs negative.
//       return false;
//     } else {
//       // rhs positive.
//       if (out_of_range) {
//         return !rhs.out_of_range;
//       } else if (rhs.out_of_range) {
//         return true;
//       } else {
//         return magnitude <= rhs.magnitude;
//       }
//     }
//   }
// }

// std::string IntegerValueInfo::to_s() const {
//   if (invalid)
//     return std::string("INVALID");
//   std::string s;
//   s.push_back(negative ? '-' : '+');
//   if (!out_of_range) {
//     s.append(std::to_string(magnitude));
//   } else {
//     s.append("MAX");
//   }
//   return s;
// }

// static bool isIntegerType(Type type) {
//   return llvm::isa<IntegerType, IndexType>(type);
// }

// void IntegerMinValueElement::initializeValue(Value value, DFX::Solver
// &solver) {
//   if (!isIntegerType(value.getType())) {
//     indicatePessimisticFixpoint();
//     return;
//   }
// }

// ChangeStatus IntegerMinValueElement::updateValue(Value value,
//                                                  DFX::Solver &solver) {
//   IntegerMinState newState = getState();

//   if (newState.isAtFixpoint()) {
//     return DFX::clampStateAndIndicateChange(getState(), newState);
//   }

//   return DFX::clampStateAndIndicateChange(getState(), newState);
// }

// const std::string IntegerMinValueElement::getAsStr(AsmState &asmState) const
// {
//   auto info = getAssumed();
//   std::string s("min-info: ");
//   s += info.to_s();
//   return s;
// }

// void IntegerMaxValueElement::initializeValue(Value value, DFX::Solver
// &solver) {
//   if (!isIntegerType(value.getType())) {
//     indicatePessimisticFixpoint();
//     return;
//   }
// }

// ChangeStatus IntegerMaxValueElement::updateValue(Value value,
//                                                  DFX::Solver &solver) {
//   IntegerMaxState newState = getState();

//   if (newState.isAtFixpoint()) {
//     return DFX::clampStateAndIndicateChange(getState(), newState);
//   }

//   return DFX::clampStateAndIndicateChange(getState(), newState);
// }

// const std::string IntegerMaxValueElement::getAsStr(AsmState &asmState) const
// {
//   auto info = getAssumed();
//   std::string s("max-info: ");
//   s += info.to_s();
//   return s;
// }

// void IntegerMaxDivisorElement::initializeValue(Value value,
//                                                DFX::Solver &solver) {
//   if (!isIntegerType(value.getType())) {
//     indicatePessimisticFixpoint();
//     return;
//   }
// }

// ChangeStatus IntegerMaxDivisorElement::updateValue(Value value,
//                                                    DFX::Solver &solver) {
//   IntegerMaxDivisorState newState = getState();

//   if (newState.isAtFixpoint()) {
//     return DFX::clampStateAndIndicateChange(getState(), newState);
//   }

//   return DFX::clampStateAndIndicateChange(getState(), newState);
// }

// const std::string IntegerMaxDivisorElement::getAsStr(AsmState &asmState)
// const {
//   auto state = getAssumed();
//   if (!state) {
//     return std::string("INVALID");
//   } else {
//     return std::to_string(*state);
//   }
// }

} // namespace mlir::iree_compiler::IREE::Util
