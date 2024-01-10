// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include <type_traits>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

namespace mlir::iree_compiler {

// Calls a collection of callbacks for an operation.
// For concrete ops each callback is called only if its concrete op type
// matches the given operation.
//
// Operation* op = ...;
// OpVisitorCollection visitors = ...;
// visitors.emplaceVisitors<MyVisitor1, MyVisitor1>(
//   visitorConstructorArg1,
//   visitorConstructorArg2);
// visitors.insertVisitors(
//   [](ConcreteOp op) { ... }, // Call only for ConcreteOp
//   [](Operation* op) { ... } // Call for all op types
// );
// visitors(op);
struct OpVisitorCollection {
  void operator()(Operation *op) {
    for (auto &fn : everyOpFns) {
      fn(op);
    }

    auto it = opFnMap.find(op->getName().getTypeID());
    if (it == opFnMap.end()) {
      return;
    }

    for (auto &fn : it->second) {
      fn(op);
    }
  }

  template <typename Op, typename Fn>
  void insertVisitor(Fn &&fn) {
    opFnMap[TypeID::get<Op>()].emplace_back(
        ConcreteOpFn<Op, Fn>(std::forward<Fn>(fn)));
  }

  template <typename Fn,
            typename = std::enable_if_t<!std::is_invocable_v<Fn, Operation *>>>
  void insertVisitor(Fn &&fn) {
    insertVisitor<
        std::decay_t<typename llvm::function_traits<Fn>::template arg_t<0>>>(
        std::forward<Fn>(fn));
  }

  template <typename Fn,
            typename = std::enable_if_t<std::is_invocable_v<Fn, Operation *>>,
            typename = void>
  void insertVisitor(Fn &&fn) {
    everyOpFns.emplace_back(std::forward<Fn>(fn));
  }

  template <typename Fn, typename... RestFns>
  void insertVisitors(Fn &&fn, RestFns &&...restFns) {
    insertVisitor(fn);
    insertVisitors(std::forward<RestFns>(restFns)...);
  }

  template <typename Fn>
  void insertVisitors(Fn &&fn) {
    insertVisitor(fn);
  }

  template <typename... Fns, typename... ConstructorArgs>
  void emplaceVisitors(ConstructorArgs &&...args) {
    (emplaceVisitor<Fns>(std::forward<ConstructorArgs>(args)...), ...);
  }

private:
  template <typename Fn, typename... ConstructorArgs>
  void emplaceVisitor(ConstructorArgs &&...args) {
    insertVisitor(Fn(std::forward<ConstructorArgs>(args)...));
  }

  template <typename Op, typename Fn>
  struct ConcreteOpFn {

    template <typename FnArg>
    ConcreteOpFn(FnArg &&fn) : fn(fn) {}

    void operator()(Operation *op) { fn(llvm::cast<Op>(op)); }

  private:
    Fn fn;
  };

  DenseMap<TypeID, SmallVector<std::function<void(Operation *)>>> opFnMap;
  SmallVector<std::function<void(Operation *)>> everyOpFns;
};

} // namespace mlir::iree_compiler
