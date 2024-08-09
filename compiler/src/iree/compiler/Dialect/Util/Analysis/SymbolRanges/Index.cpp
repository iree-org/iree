// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/SymbolRanges/Index.h"
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-util-index-range"
using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// FunctionalRangeMaps
//===----------------------------------------------------------------------===//

std::unique_ptr<FunctionConstraintSet>
FunctionConstraintSet::constructConstraintSet(Explorer &explorer,
                                              FunctionOpInterface func) {
  FunctionConstraintSet cset(func.getContext());

  explorer.walkValues(func, [&](Value value) {
    TypeSwitch<Type>(value.getType())
        .Case([&](ShapedType shapedType) {
          // Track all dynamic dimensions for the given shaped type.
          for (int64_t i = 0, e = shapedType.getRank(); i < e; ++i) {
            if (shapedType.isDynamicDim(i)) {
              cset.insert(value, i, /*isSymbol=*/false);
            }
          }
        })
        // Value bounds interface does not support integer types so we are
        // restricted to index types here. It's unlikely there will ever be
        // the kind of support we need for standard i64/i32 types so forking
        // the expression tracking for ValueBoundsConstraintSet is TODO.
        .Case([&](IndexType) {
          cset.insert(value, std::nullopt, /*isSymbol=*/false);
        });
    return WalkResult::advance();
  });

  // TODO: Seed the function with constraints on the arguments.
  cset.processWorklist([](Value, std::optional<long>) { return false; });
  cset.projectOut([](ValueDim) { return false; });
  LLVM_DEBUG({
    llvm::dbgs() << "Initialized constraint map for function: " << func;
    cset.cstr.print(llvm::dbgs());
  });
  return std::make_unique<FunctionConstraintSet>(std::move(cset));
}

void FunctionConstraintSet::canonicalize() { cstr.simplify(); }

static ValueDimList collectAnalyzableOperandsDimList(ValueRange inputs) {
  ValueDimList operands;
  for (auto operand : inputs) {
    TypeSwitch<Type>(operand.getType())
        .Case([&](ShapedType shapedType) {
          // Track all dynamic dimensions for the given shaped type.
          for (int64_t i = 0, e = shapedType.getRank(); i < e; ++i) {
            if (shapedType.isDynamicDim(i)) {
              operands.push_back({operand, i});
            }
          }
        })
        .Case([&](IndexType) {
          operands.push_back({operand, std::nullopt});
        });
  }
  return operands;
}

static ValueDimList collectAnalyzableOperandsDimList(Operation *op) {
  return collectAnalyzableOperandsDimList(op->getOperands());
}

SmallVector<AffineExpr>
FunctionConstraintSet::getDimensionList(ValueDimList &operands) {
  return llvm::to_vector(llvm::map_range(
      operands, [&](std::pair<Value, std::optional<long>> v) -> AffineExpr {
        return this->getExpr(v.first, v.second);
      }));
}

void FunctionConstraintSet::incorporateFunctionalRangeBounds(
    Operation *op, FunctionalRangeMaps range) {
  if (range.isInvalid() || range.isUnbounded()) {
    return;
  }

  ValueDimList results = collectAnalyzableOperandsDimList(op->getResults());

  assert(range.lbs.size() == range.ubs.size() && "Invalid function range");
  assert(range.lbs.size() == results.size() && "Invalid range for operation");

  ValueDimList operands = collectAnalyzableOperandsDimList(op);

  for (auto [res, lb, ub] : llvm::zip_equal(results, range.lbs, range.ubs)) {
    assert(lb.getNumResults() == ub.getNumResults() && "Invalid result bounds");

    SmallVector<AffineExpr> dimReplacements = getDimensionList(operands);

    // Simplification for known equal bounds, just set the equality directly.
    if (lb == ub) {
      // Nothing to do for unbounded values.
      if (!lb) {
        continue;
      }

      for (AffineExpr expr : lb.getResults()) {
        AffineExpr boundExpr = expr.replaceDimsAndSymbols(dimReplacements, {});
        if (res.second) {
          this->bound(res.first)[*res.second] == boundExpr;
        } else {
          this->bound(res.first) == boundExpr;
        }
      }
      continue;
    }

    // Add the (closed) lower bounds if present.
    if (lb) {
      for (AffineExpr expr : lb.getResults()) {
        AffineExpr boundExpr = expr.replaceDimsAndSymbols(dimReplacements, {});
        if (res.second) {
          this->bound(res.first)[*res.second] <= boundExpr;
        } else {
          this->bound(res.first) >= boundExpr;
        }
      }
    }

    // Add the (closed) upper bounds if present.
    if (ub) {
      for (AffineExpr expr : ub.getResults()) {
        AffineExpr boundExpr = expr.replaceDimsAndSymbols(dimReplacements, {});
        if (res.second) {
          this->bound(res.first)[*res.second] >= boundExpr;
        } else {
          this->bound(res.first) >= boundExpr;
        }
      }
    }
  }
}

std::pair<AffineMap, AffineMap>
FunctionConstraintSet::getBounds(ValueDimList &operands, Value value,
                                 std::optional<long> dim) {
  int64_t pos = getPos(value, dim);
  // Compute lower and upper bounds for `valueDim`.
  SmallVector<AffineMap> lbvec(1), ubvec(1);
  cstr.getSliceBounds(pos, 1, value.getContext(), &lbvec, &ubvec,
                      /*getClosedUB=*/true);
  AffineMap lb = lbvec.empty() ? AffineMap() : lbvec[0];
  AffineMap ub = ubvec.empty() ? AffineMap() : ubvec[0];

  if (!lb && !ub) {
    return std::make_pair(AffineMap(), AffineMap());
  }

  llvm::DenseMap<int64_t, int64_t> positionToNewDimMap;
  for (auto [idx, valueDim] : llvm::enumerate(operands)) {
    Value v = valueDim.first;
    std::optional<long> dim = valueDim.second;
    int64_t argPos = getPos(v, dim);
    positionToNewDimMap[argPos] = idx;
  }

  // Gather the set of used dimensions to simplify the bound maps.
  SmallVector<AffineExpr> replacementDims;
  for (int64_t i = 0; i < cstr.getNumDimAndSymbolVars(); ++i) {
    // If one of the bounds depend on an SSA value other than one of the
    // operands, that bound is unknown externally.
    if (!positionToNewDimMap.contains(i)) {
      if (lb && lb.isFunctionOfDim(i)) {
        lb = AffineMap();
      }
      if (ub && ub.isFunctionOfDim(i)) {
        ub = AffineMap();
      }
      if (!lb && !ub) {
        return std::make_pair(AffineMap(), AffineMap());
      }
      // Exclude dims that aren't in the set of operands.
      replacementDims.push_back(getAffineConstantExpr(0, value.getContext()));
      continue;
    }
    replacementDims.push_back(
        getAffineDimExpr(positionToNewDimMap[i], value.getContext()));
  }

  // Replace the dimensions and symbols in each bound map with the updated set
  // of expressions. The dimensions and symbols in the resulting maps will
  // be aligned.
  if (lb) {
    lb = lb.replaceDimsAndSymbols(replacementDims, {}, operands.size(),
                                  /*numSymbols=*/0);
  }
  if (ub) {
    ub = ub.replaceDimsAndSymbols(replacementDims, {}, operands.size(),
                                  /*numSymbols=*/0);
  }
  return std::make_pair(lb, ub);
}

std::pair<AffineMap, AffineMap>
FunctionConstraintSet::getUnspecifiedBounds(Value value,
                                            std::optional<long> dim) {
  int64_t pos = getPos(value, dim);
  // Compute lower and upper bounds for `valueDim`.
  SmallVector<AffineMap> lbvec(1), ubvec(1);
  cstr.getSliceBounds(pos, 1, value.getContext(), &lbvec, &ubvec,
                      /*getClosedUB=*/true);
  AffineMap lb = lbvec.empty() ? AffineMap() : lbvec[0];
  AffineMap ub = ubvec.empty() ? AffineMap() : ubvec[0];

  if (!lb && !ub) {
    return std::make_pair(AffineMap(), AffineMap());
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Querying bounds for value " << value;
    if (dim) {
      llvm::dbgs() << " at dim " << *dim;
    }
    llvm::dbgs() << "\n";
    llvm::dbgs() << "Lower bound: " << lb << "\n";
    llvm::dbgs() << "Upper bound: " << ub << "\n";
  });

  // Compute the simplified bound maps.
  SmallVector<AffineExpr> replacementDims, replacementSymbols;
  int64_t numDims = 0, numSymbols = 0;
  Builder b(value.getContext());
  for (int64_t i = 0; i < cstr.getNumDimAndSymbolVars(); ++i) {
    // Skip `value`.
    if (i == pos) {
      if (lb.isFunctionOfDim(i)) {
        lb = AffineMap();
      }
      if (ub.isFunctionOfDim(i)) {
        ub = AffineMap();
      }
    }
    // Check if the position `i` is used in the generated bound. If so, it must
    // be included in the generated affine.apply op.
    bool used = false;
    bool isDim = i < cstr.getNumDimVars();
    if (isDim) {
      if ((lb && lb.isFunctionOfDim(i)) || (ub && ub.isFunctionOfDim(i))) {
        used = true;
      }
    } else {
      if ((lb && lb.isFunctionOfSymbol(i - cstr.getNumDimVars())) ||
          (ub && ub.isFunctionOfSymbol(i - cstr.getNumDimVars()))) {
        used = true;
      }
    }

    if (!used) {
      // Not used: Remove dim/symbol from the result.
      if (isDim) {
        replacementDims.push_back(b.getAffineConstantExpr(0));
      } else {
        replacementSymbols.push_back(b.getAffineConstantExpr(0));
      }
      continue;
    }

    if (isDim) {
      replacementDims.push_back(b.getAffineDimExpr(numDims++));
    } else {
      replacementSymbols.push_back(b.getAffineSymbolExpr(numSymbols++));
    }
  }

  if (lb) {
    lb = lb.replaceDimsAndSymbols(replacementDims, replacementSymbols, numDims,
                                  numSymbols);
  }
  if (ub) {
    ub = ub.replaceDimsAndSymbols(replacementDims, replacementSymbols, numDims,
                                  numSymbols);
  }
  return std::make_pair(lb, ub);
}

std::pair<std::optional<int64_t>, std::optional<int64_t>>
FunctionConstraintSet::getConstantBounds(Value operand,
                                         std::optional<int64_t> dim) {
  int64_t pos = getPos(operand, dim);
  std::optional<int64_t> ub =
      cstr.getConstantBound64(presburger::BoundType::UB, pos);
  std::optional<int64_t> lb =
      cstr.getConstantBound64(presburger::BoundType::LB, pos);
  return std::make_pair(lb, ub);
}

//===----------------------------------------------------------------------===//
// FunctionalRangeMaps
//===----------------------------------------------------------------------===//

std::string FunctionalRangeMaps::getAsStr(AsmState &asmState) const {
  if (!valid)
    return std::string("<<INVALID>>");
  if (isUnbounded())
    return std::string("<<UNBOUNDED>>");
  std::string s;
  llvm::raw_string_ostream stream(s);
  if (!lbs.empty()) {
    stream << "lower bounds: [";
    llvm::interleaveComma(lbs, stream);
    stream << "]\n";
  }
  if (!ubs.empty()) {
    stream << "upper bounds: [";
    llvm::interleaveComma(ubs, stream);
    stream << "]";
  }
  return s;
}

//===----------------------------------------------------------------------===//
// FunctionalRangeOperationElement
//===----------------------------------------------------------------------===//

const char FunctionRangeOperationElement::ID = 0;

void FunctionRangeOperationElement::initializeOperation(
    FunctionOpInterface func, DFX::Solver &solver) {
  assert(!constraints && "Operation element initialized twice");
  constraints =
      FunctionConstraintSet::constructConstraintSet(solver.getExplorer(), func);
  // All functions are initialized to the widest range possible.
}

ChangeStatus
FunctionRangeOperationElement::updateOperation(FunctionOpInterface func,
                                               DFX::Solver &solver) {
  // TODO: Support CFGs.
  if (func.getBlocks().size() != 1) {
    return ChangeStatus::UNCHANGED;
  }

  Block *entry = &func.getBlocks().front();
  Operation *terminator = entry->getTerminator();

  ValueDimList operands =
      collectAnalyzableOperandsDimList(entry->getArguments());
  ValueDimList results = collectAnalyzableOperandsDimList(terminator);

  FunctionalRangeState newState = getState();

  SmallVector<AffineExpr> operandDimReplacements =
      constraints.get()->getDimensionList(operands);

  SmallVector<AffineMap> lbs;
  SmallVector<AffineMap> ubs;

  for (auto [value, maybeDim] : results) {
    auto [lb, ub] = constraints.get()->getBounds(operands, value, maybeDim);
    lbs.push_back(lb);
    ubs.push_back(ub);
  }
  newState.setAssumed(FunctionalRangeMaps(lbs, ubs));

  // Walk all outgoing calls and update the constraint maps with the new
  // functional constraints.
  solver.getExplorer().walkIncomingCalls(func, [&](CallOpInterface call) {
    FunctionOpInterface parent = call->getParentOfType<FunctionOpInterface>();
    auto &callConstraintElement =
        solver.getElementFor<FunctionRangeOperationElement>(
            *this, Position::forOperation(parent), DFX::Resolution::OPTIONAL);
    callConstraintElement.constraints.get()->incorporateFunctionalRangeBounds(
        call, newState.getAssumed());
    return WalkResult::advance();
  });

  return DFX::clampStateAndIndicateChange(getState(), newState);
}

const std::string
FunctionRangeOperationElement::getAsStr(AsmState &asmState) const {
  auto range = getAssumed();
  std::string s("result-ranges: ");
  s += range.getAsStr(asmState);
  return s;
}

//===----------------------------------------------------------------------===//
// IndexRangeAnalysis
//===----------------------------------------------------------------------===//

// Runs analysis and populates the state cache.
// May fail if analysis cannot be completed due to unsupported or unknown IR.
LogicalResult IndexRangeAnalysis::run() {
  // Seed all block arguments throughout the program.
  for (auto funcOp : topLevelOps) {
    solver.getOrCreateElementFor<FunctionRangeOperationElement>(
        Position::forOperation(funcOp));
  }

  // Run solver to completion.
  auto res = solver.run();

  return res;
}

std::pair<AffineMap, AffineMap>
IndexRangeAnalysis::getBounds(Value operand, std::optional<long> dim) {
  FunctionOpInterface parent =
      operand.getParentRegion()->getParentOfType<FunctionOpInterface>();
  auto &parentFunctionConstraints =
      solver.getOrCreateElementFor<FunctionRangeOperationElement>(
          Position::forOperation(parent));
  return parentFunctionConstraints.getBounds(operand, dim);
}

std::pair<std::optional<int64_t>, std::optional<int64_t>>
IndexRangeAnalysis::getConstantBounds(Value operand, std::optional<long> dim) {
  FunctionOpInterface parent =
      operand.getParentRegion()->getParentOfType<FunctionOpInterface>();
  auto &parentFunctionConstraints =
      solver.getOrCreateElementFor<FunctionRangeOperationElement>(
          Position::forOperation(parent));
  return parentFunctionConstraints.getConstantBounds(operand, dim);
}

int64_t IndexRangeAnalysis::getStaticGCD(Value operand,
                                         std::optional<long> dim) {
  auto [lb, ub] = getBounds(operand, dim);
  if (!lb || lb != ub) {
    return 1;
  }
  return lb.getLargestKnownDivisorOfMapExprs();
}

} // namespace mlir::iree_compiler::IREE::Util
