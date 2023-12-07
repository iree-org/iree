// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_ELEMENT_H_
#define IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_ELEMENT_H_

#include "iree/compiler/Dialect/Util/Analysis/DFX/DepGraph.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::DFX {

class Solver;

//===----------------------------------------------------------------------===//
// AbstractElement
//===----------------------------------------------------------------------===//

// Base type for information in the solver framework.
// Each element represents some assumed and known knowledge anchored on a
// specific position in the IR such as a Value or Operation.
class AbstractElement : public Position, public DepGraphNode {
public:
  using StateType = AbstractState;

  AbstractElement(const Position &pos) : Position(pos) {}
  virtual ~AbstractElement() = default;

  // Returns an IR position anchoring this element to the IR.
  const Position &getPosition() const { return *this; };
  Position &getPosition() { return *this; };

  // Returns the internal abstract state for inspection.
  virtual StateType &getState() = 0;
  virtual const StateType &getState() const = 0;

  // Initializes the state with the information in |solver|.
  //
  // This function is called by the solver once all abstract elements
  // have been identified. It can and shall be used for tasks like:
  //  - identify existing knowledge in the IR and use it for the "known state"
  //  - perform any work that is not going to change over time, e.g., determine
  //    a subset of the IR, or elements in-flight, that have to be looked at
  //    in the `updateImpl` method.
  virtual void initialize(Solver &solver) {}

  // Returns the name of the AbstractElement for debug printing.
  virtual const std::string getName() const = 0;
  // Returns the address of the ID of the AbstractElement for type comparison.
  virtual const void *getID() const = 0;

  // Returns true if |node| is of type AbstractElement so that the dyn_cast and
  // cast can use such information to cast an DepGraphNode to an
  // AbstractElement.
  //
  // We eagerly return true here because all DepGraphNodes except for the
  // synthethis node are of type AbstractElement.
  static bool classof(const DepGraphNode *node) { return true; }

  // Returns the human-friendly summarized assumed state as string for
  // debugging.
  virtual const std::string getAsStr(AsmState &asmState) const = 0;

  void print(llvm::raw_ostream &os, AsmState &asmState) const override;
  virtual void printWithDeps(llvm::raw_ostream &os, AsmState &asmState) const;
  void dump(AsmState &asmState) const;

  friend class Solver;

protected:
  // Hook for the solver to trigger an update of the internal state.
  //
  // If this attribute is already fixed this method will return UNCHANGED,
  // otherwise it delegates to `AbstractElement::updateImpl`.
  //
  // Returns CHANGED if the internal state changed, otherwise UNCHANGED.
  ChangeStatus update(Solver &solver);

  // Update/transfer function which has to be implemented by the derived
  // classes.
  //
  // When called the environment has changed and we have to determine if
  // the current information is still valid or adjust it otherwise.
  //
  // Returns CHANGED if the internal state changed, otherwise UNCHANGED.
  virtual ChangeStatus updateImpl(Solver &solver) = 0;
};

//===----------------------------------------------------------------------===//
// OperationElement/TypedOperationElement<T>
//===----------------------------------------------------------------------===//

// An abstract element that represents an operation in the IR.
struct OperationElement : public AbstractElement {
  using AbstractElement::AbstractElement;

  void initialize(Solver &solver) override {
    if (isOperation()) {
      auto &op = getOperation();
      initializeOperation(&op, solver);
    } else {
      getState().indicatePessimisticFixpoint();
    }
  }

  virtual void initializeOperation(Operation *op, Solver &solver) = 0;

  ChangeStatus updateImpl(Solver &solver) override {
    if (isOperation()) {
      auto &op = getOperation();
      return updateOperation(&op, solver);
    } else {
      return getState().indicatePessimisticFixpoint();
    }
  }

  virtual ChangeStatus updateOperation(Operation *op, Solver &solver) = 0;
};

// An abstract element that represents an operation of type OpT in the IR.
template <typename OpT>
struct TypedOperationElement : public AbstractElement {
  using AbstractElement::AbstractElement;

  void initialize(Solver &solver) override {
    if (isOperation()) {
      auto op = dyn_cast<OpT>(getOperation());
      if (op) {
        initializeOperation(op, solver);
        return;
      }
    }
    getState().indicatePessimisticFixpoint();
  }

  virtual void initializeOperation(OpT op, Solver &solver) = 0;

  ChangeStatus updateImpl(Solver &solver) override {
    if (isOperation()) {
      auto op = dyn_cast<OpT>(getOperation());
      if (op)
        return updateOperation(op, solver);
    }
    return getState().indicatePessimisticFixpoint();
  }

  virtual ChangeStatus updateOperation(OpT op, Solver &solver) = 0;
};

//===----------------------------------------------------------------------===//
// ValueElement
//===----------------------------------------------------------------------===//

// An abstract element that represents a value position in the IR.
struct ValueElement : public AbstractElement {
  using AbstractElement::AbstractElement;

  void initialize(Solver &solver) override {
    if (isValue()) {
      auto value = getValue();
      initializeValue(value, solver);
    } else {
      getState().indicatePessimisticFixpoint();
    }
  }

  virtual void initializeValue(Value value, Solver &solver) = 0;

  ChangeStatus updateImpl(Solver &solver) override {
    if (isValue()) {
      auto value = getValue();
      return updateValue(value, solver);
    } else {
      return getState().indicatePessimisticFixpoint();
    }
  }

  virtual ChangeStatus updateValue(Value value, Solver &solver) = 0;
};

} // namespace mlir::iree_compiler::DFX

#endif // IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_ELEMENT_H_
