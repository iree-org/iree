// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_SOLVER_H_
#define IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_SOLVER_H_

#include "iree/compiler/Dialect/Util/Analysis/DFX/DepGraph.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::DFX {

// Fixed point iteration solver ("monotone framework").
// http://symbolaris.com/course/Compilers11/27-monframework.pdf
//
// Forked from the LLVM Attributor: llvm/Transforms/IPO/Attributor.h
// The attributor is an elegant and flexible piece of infra that is tied quite
// tightly to LLVM IR. Here we extract it and generalize it to work with MLIR's
// concepts of positional values, operations, and blocks. Unlike the Attributor
// the solver is only for performing analysis and does no manifestation. We may
// want to extend this to integrate into the MLIR folding framework, though.
//
// Good talks describing how the system works:
//  https://www.youtube.com/watch?v=I4Iv-HefknA
//  https://www.youtube.com/watch?v=CzWkc_JcfS0
//
// This initial fork is to unblock work that requires this framework. Ideally
// we'd upstream this into MLIR proper but there are some missing core
// interfaces that keeps it locked here for the moment: in particular we need
// tied operands (generalized view-like op interface), globals, and reference
// types. We also need a lot of tests :)
//
// NOTE: the solver state - like Explorer - assumes that IR will not be modified
// while it is in-use. Modifying the IR invalidates the state and may lead to
// crashes as pointer references into the IR structure are retained.
class Solver {
public:
  // Creates a solver that uses |explorer| for walking the IR tree and
  // |allocator| for transient allocations of abstract elements.
  explicit Solver(Explorer &explorer, llvm::BumpPtrAllocator &allocator)
      : explorer(explorer), asmState(explorer.getAsmState()),
        allocator(allocator), depGraph(explorer.getAsmState()) {}
  ~Solver();

  // Initialized explorer for walking the IR.
  Explorer &getExplorer() { return explorer; }

  // Shared AsmState that can be used to efficiently print MLIR Values.
  // If this is not used the entire module may need to be walked in order to
  // get the name of a value each time it's printed. Nothing in this framework
  // should do that.
  AsmState &getAsmState() { return asmState; }

  // An allocator whose lifetime is at least as long as the solver.
  llvm::BumpPtrAllocator &getAllocator() { return allocator; }

  // Returns the element of |ElementT| for |pos| and adds a dependency from
  // |queryingElement| to the returned element with the given |resolution|.
  template <typename ElementT>
  const ElementT &getElementFor(const AbstractElement &queryingElement,
                                const Position &pos, Resolution resolution) {
    return getOrCreateElementFor<ElementT>(pos, &queryingElement, resolution,
                                           /*forceUpdate=*/false);
  }

  // Returns the element of |ElementT| for |pos| and adds a dependency from
  // |queryingElement| to the returned element with the given |resolution|.
  // If the element already exists and the solver is in the UPDATE phase it will
  // be updated prior to returning as if another iteration had been performed.
  template <typename ElementT>
  const ElementT &getAndUpdateElementFor(const AbstractElement &queryingElement,
                                         const Position &pos,
                                         Resolution resolution) {
    return getOrCreateElementFor<ElementT>(pos, &queryingElement, resolution,
                                           /*forceUpdate=*/true);
  }

  // Returns the element of |ElementT| for |pos| and optionally adds a
  // dependency from |queryingElement| to the returned element with the given
  // |resolution|.
  //
  // Using this after the solver started running is restricted to only the
  // solver itself. Initial seeding of elements can be done via this function.
  //
  // NOTE: |forceUpdate| is ignored in any stage other than the update stage.
  template <typename ElementT>
  const ElementT &
  getOrCreateElementFor(Position pos, const AbstractElement *queryingElement,
                        Resolution resolution, bool forceUpdate = false,
                        bool updateAfterInit = true) {
    if (auto *elementPtr =
            lookupElementFor<ElementT>(pos, queryingElement, resolution,
                                       /*allowInvalidState=*/true)) {
      if (forceUpdate && phase == Phase::UPDATE) {
        updateElement(*elementPtr);
      }
      return *elementPtr;
    }

    // No matching element found: create one.
    auto &element = ElementT::createForPosition(pos, *this);
    registerElement(element);

    // Avoid too many nested initializations to prevent a stack overflow.
    static const int maxInitializationChainLength = 1024;
    if (initializationChainLength > maxInitializationChainLength) {
      element.getState().indicatePessimisticFixpoint();
      return element;
    }

    // Bootstrap the new element with an initial update to propagate info.
    {
      ++initializationChainLength;
      element.initialize(*this);
      --initializationChainLength;
    }

    // If this is queried after we've performed iteration we force the element
    // to indicate pessimistic fixpoint immediately.
    if (phase == Phase::DONE) {
      element.getState().indicatePessimisticFixpoint();
      return element;
    }

    // Allow seeded elements to declare dependencies that are preserved for
    // use during fixed point iteration.
    if (updateAfterInit) {
      auto oldPhase = phase;
      phase = Phase::UPDATE;
      updateElement(element);
      phase = oldPhase;
    }

    if (queryingElement && element.getState().isValidState()) {
      recordDependence(element, const_cast<AbstractElement &>(*queryingElement),
                       resolution);
    }
    return element;
  }

  // Returns the element of |ElementT| for |pos| if existing and valid.
  template <typename ElementT>
  const ElementT &getOrCreateElementFor(const Position &pos) {
    return getOrCreateElementFor<ElementT>(pos, /*queryingElement=*/nullptr,
                                           Resolution::NONE);
  }

  // Returns the element of |ElementT| for |pos| if existing and valid.
  // |queryingElement| can be nullptr to allow for lookups from outside of the
  // solver system.
  template <typename ElementT>
  ElementT *lookupElementFor(const Position &pos,
                             const AbstractElement *queryingElement = nullptr,
                             Resolution resolution = Resolution::OPTIONAL,
                             bool allowInvalidState = false) {
    static_assert(std::is_base_of<AbstractElement, ElementT>::value,
                  "cannot query an element with a type not derived from "
                  "'AbstractElement'");

    // Lookup the abstract element of type ElementT and if found return it after
    // registering a dependence of queryingElement on the one returned element.
    auto *elementPtr = elementMap.lookup({&ElementT::ID, pos});
    if (!elementPtr)
      return nullptr;
    auto *element = static_cast<ElementT *>(elementPtr);

    // Do not register a dependence on an element with an invalid state.
    if (resolution != Resolution::NONE && queryingElement &&
        element->getState().isValidState()) {
      recordDependence(*element,
                       const_cast<AbstractElement &>(*queryingElement),
                       resolution);
    }

    // Return nullptr if this element has an invalid state.
    if (!allowInvalidState && !element->getState().isValidState()) {
      return nullptr;
    }
    return element;
  }

  // Explicitly record a dependence from |fromElement| to |toElement|,
  // indicating that if |fromElement| changes |toElement| should be updated as
  // well.
  //
  // This method should be used in conjunction with the `getElementFor` method
  // and with the resolution enum passed to the method set to NONE. This can be
  // beneficial to avoid false dependencies but it requires the users of
  // `getElementFor` to explicitly record true dependencies through this method.
  // The |resolution| flag indicates if the dependence is strictly necessary.
  // That means for required dependences if |fromElement| changes to an invalid
  // state |toElement| can be moved to a pessimistic fixpoint because it
  // required information from |fromElement| but none are available anymore.
  void recordDependence(const AbstractElement &fromElement,
                        const AbstractElement &toElement,
                        Resolution resolution);

  // Introduces a new abstract element into the fixpoint analysis.
  //
  // Note that ownership of the element is given to the solver and the solver
  // will invoke delete on destruction of the solver.
  //
  // Elements are identified by their IR position (ElementT::getPosition())
  // and the address of their static member (see ElementT::ID).
  template <typename ElementT>
  ElementT &registerElement(ElementT &element) {
    static_assert(std::is_base_of<AbstractElement, ElementT>::value,
                  "cannot register an element with a type not derived from "
                  "'AbstractElement'!");

    // Put the element in the lookup map structure and the container we use to
    // keep track of all attributes.
    const auto &pos = element.getPosition();
    AbstractElement *&elementPtr = elementMap[{&ElementT::ID, pos}];
    assert(!elementPtr && "element already in map!");
    elementPtr = &element;

    // Register element with the synthetic root only before we are done.
    if (phase == Phase::SEEDING || phase == Phase::UPDATE) {
      depGraph.syntheticRoot.deps.push_back(
          DepGraphNode::DepTy(&element, unsigned(Resolution::REQUIRED)));
    }

    return element;
  }

  // Runs the solver until either it converges to a fixed point or exceeds the
  // maximum iteration count. Returns success() if it converges in time.
  LogicalResult run();

  // Prints the constraint dependency graph to |os|.
  void print(llvm::raw_ostream &os);
  // Dumps a .dot of the constraint dependency graph to a file.
  void dumpGraph();

protected:
  friend DepGraph;

  Explorer &explorer;
  AsmState &asmState;
  llvm::BumpPtrAllocator &allocator;

  // This method will do fixpoint iteration until a fixpoint or the maximum
  // iteration count is reached.
  //
  // If the maximum iteration count is reached this method will
  // indicate pessimistic fixpoint on elements that transitively depend on
  // elements that were still scheduled for an update.
  LogicalResult runTillFixpoint();

  // Runs update on |element| and tracks the dependencies queried while doing
  // so. Also adjusts the state if we know further updates are not necessary.
  ChangeStatus updateElement(AbstractElement &element);

  // Remembers the dependences on the top of the dependence stack such that they
  // may trigger further updates.
  void rememberDependences();

  // Maximum number of fixed point iterations or None for default.
  std::optional<unsigned> maxFixpointIterations;

  // A flag that indicates which stage of the process we are in.
  enum class Phase {
    // Initial elements are being registered to seed the graph.
    SEEDING,
    // Fixed point iteration is running.
    UPDATE,
    // Iteration has completed; does not indicate whether it coverged.
    DONE,
  } phase = Phase::SEEDING;

  // The current initialization chain length. Tracked to avoid stack overflows
  // during recursive initialization.
  unsigned initializationChainLength = 0;

  using ElementMapKeyTy = std::pair<const char *, Position>;
  DenseMap<ElementMapKeyTy, AbstractElement *> elementMap;

  // Element dependency graph indicating the resolution constraints across
  // elements.
  DepGraph depGraph;

  // Information about a dependence:
  //   If fromElement is changed toElement needs to be updated as well.
  struct DepInfo {
    const AbstractElement *fromElement;
    const AbstractElement *toElement;
    Resolution resolution;
  };

  // The dependence stack is used to track dependences during an
  // `AbstractElement::update` call. As `AbstractElement::update` can be
  // recursive we might have multiple vectors of dependences in here. The stack
  // size, should be adjusted according to the expected recursion depth and the
  // inner dependence vector size to the expected number of dependences per
  // abstract element. Since the inner vectors are actually allocated on the
  // stack we can be generous with their size.
  using DependenceVector = SmallVector<DepInfo, 8>;
  SmallVector<DependenceVector *, 16> dependenceStack;
};

} // namespace mlir::iree_compiler::DFX

#endif // IREE_COMPILER_DIALECT_UTIL_ANALYSIS_DFX_SOLVER_H_
