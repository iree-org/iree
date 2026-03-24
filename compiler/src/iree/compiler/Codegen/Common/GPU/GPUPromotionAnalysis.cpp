// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPromotionAnalysis.h"

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler {

using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Lattice value
//===----------------------------------------------------------------------===//

struct PromotionTypeLatticeValue {
  enum class State { None, Concrete, Overdefined };
  State state = State::None;
  Attribute promotionType; // valid when state == Concrete

  PromotionTypeLatticeValue() = default;
  explicit PromotionTypeLatticeValue(Attribute type)
      : state(State::Concrete), promotionType(type) {}

  bool isDefined() const { return state == State::Concrete; }
  bool isOverdefined() const { return state == State::Overdefined; }

  bool operator==(const PromotionTypeLatticeValue &rhs) const {
    return state == rhs.state && promotionType == rhs.promotionType;
  }

  static PromotionTypeLatticeValue meet(const PromotionTypeLatticeValue &lhs,
                                        const PromotionTypeLatticeValue &rhs) {
    if (lhs.isOverdefined() || rhs.isOverdefined()) {
      return getOverdefined();
    }
    if (!lhs.isDefined()) {
      return rhs;
    }
    if (!rhs.isDefined()) {
      return lhs;
    }
    // Both concrete.
    if (lhs.promotionType == rhs.promotionType) {
      return lhs;
    }
    return getOverdefined();
  }

  // Required by Lattice<T> but unused in backward analysis.
  static PromotionTypeLatticeValue join(const PromotionTypeLatticeValue &lhs,
                                        const PromotionTypeLatticeValue &rhs) {
    return meet(lhs, rhs);
  }

  static PromotionTypeLatticeValue getOverdefined() {
    PromotionTypeLatticeValue val;
    val.state = State::Overdefined;
    return val;
  }

  void print(raw_ostream &os) const {
    switch (state) {
    case State::None:
      os << "None";
      break;
    case State::Concrete:
      os << promotionType;
      break;
    case State::Overdefined:
      os << "Overdefined";
      break;
    }
  }
};

//===----------------------------------------------------------------------===//
// Lattice
//===----------------------------------------------------------------------===//

struct PromotionTypeLattice : public Lattice<PromotionTypeLatticeValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromotionTypeLattice)
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

namespace {

/// Returns true for ops through which promotion type should propagate.
static bool isPropagatable(Operation *op) {
  if (op->hasTrait<OpTrait::Elementwise>()) {
    return true;
  }
  return isa<vector::TransposeOp, vector::ShapeCastOp, vector::BroadcastOp,
             IREE::VectorExt::ToLayoutOp>(op);
}

class PromotionTypeAnalysis
    : public SparseBackwardDataFlowAnalysis<PromotionTypeLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  void setToExitState(PromotionTypeLattice *lattice) override {
    // Exit state = None (no promotion type known at exits).
    propagateIfChanged(lattice, lattice->meet(PromotionTypeLatticeValue()));
  }

  LogicalResult
  visitOperation(Operation *op, ArrayRef<PromotionTypeLattice *> operands,
                 ArrayRef<const PromotionTypeLattice *> results) override {
    // Anchor: to_layout with promotion_type discardable attribute.
    if (auto toLayout = dyn_cast<IREE::VectorExt::ToLayoutOp>(op)) {
      if (Attribute pt = toLayout->getAttr(kPromotionTypeAttr)) {
        propagateIfChanged(operands[0],
                           operands[0]->meet(PromotionTypeLatticeValue(pt)));
        return success();
      }
    }

    // Propagate through supported ops: the result's promotion type
    // applies to all operands.
    if (isPropagatable(op)) {
      for (const PromotionTypeLattice *result : results) {
        for (PromotionTypeLattice *operand : operands) {
          meet(operand, *result);
        }
      }
      return success();
    }

    // For unsupported ops: don't propagate (operands stay at exit state).
    return success();
  }

  // Required overrides for pure virtual methods in the base class. No-ops
  // because promotion types only propagate through op data flow, not through
  // control flow edges or call sites.
  void visitBranchOperand(OpOperand &operand) override {}
  void visitCallOperand(OpOperand &operand) override {}
  void
  visitNonControlFlowArguments(RegionSuccessor &successor,
                               ArrayRef<BlockArgument> arguments) override {}
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

PromotionTypeMap analyzePromotionTypes(Operation *root) {
  DataFlowSolver solver;
  SymbolTableCollection symbolTable;
  loadBaselineAnalyses(solver);
  solver.load<PromotionTypeAnalysis>(symbolTable);
  if (failed(solver.initializeAndRun(root))) {
    return PromotionTypeMap();
  }

  PromotionTypeMap result;
  root->walk([&](Operation *op) {
    for (Value res : op->getResults()) {
      const auto *lattice = solver.lookupState<PromotionTypeLattice>(res);
      if (!lattice) {
        continue;
      }
      const PromotionTypeLatticeValue &val = lattice->getValue();
      if (val.isDefined()) {
        result[res] = val.promotionType;
      }
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Test pass
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TESTGPUPROMOTIONANALYSISPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct TestGPUPromotionAnalysisPass final
    : impl::TestGPUPromotionAnalysisPassBase<TestGPUPromotionAnalysisPass> {
  void runOnOperation() override {
    Operation *root = getOperation();
    PromotionTypeMap promotionTypes = analyzePromotionTypes(root);

    root->walk([&](Operation *op) {
      for (OpResult result : op->getOpResults()) {
        auto it = promotionTypes.find(result);
        if (it != promotionTypes.end()) {
          op->emitRemark("promotion type of result #")
              << result.getResultNumber() << " is " << it->second;
        }
      }
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler
