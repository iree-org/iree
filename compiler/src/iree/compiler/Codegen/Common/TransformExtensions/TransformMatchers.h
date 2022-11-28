// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_TRANSFORMMATCHERS_H_
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_TRANSFORMMATCHERS_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler::IREE::transform_dialect {

/// A tag indicating the shape being static or dynamic, for use with the
/// structured op matcher.
enum class ShapeKind { Static, Dynamic };

/// A placeholder indicating the structured op matcher to check the predicate
/// for all dimensions.
struct AllDims {};

/// A placeholder indicating the structured op matcher to check the predicate
/// for all operands of the relevant kind.
struct AllOperands {};

/// A tag indicating to look for any user of the operation's result that would
/// satisfy the predicate.
struct HasAnyUse {};

/// Base class for predicate parameters that can be described with the single
/// value. Concrete predicate parameters should inherit this and forward the
/// constructor via `using Base::Base`.
template <typename T>
struct SingleValuePredicateParam {
  using Base = SingleValuePredicateParam<T>;
  explicit SingleValuePredicateParam(T value) : value(value) {}
  const T value;
};

/// Indicates that the dimension must be divisible by the given value.
struct DivisibleBy : public SingleValuePredicateParam<int64_t> {
  using Base::Base;
};

/// Indicates that the number of entities must be equal to the given value.
struct NumEqualsTo : public SingleValuePredicateParam<size_t> {
  using Base::Base;
};

/// Indicates that the bit width of the elemental type must be equal to the give
/// value.
struct ElementTypeBitWidth : public SingleValuePredicateParam<size_t> {
  using Base::Base;
};

/// Predicate tag indicating that the affine map is a permutation.
struct IsPermutation {};

/// Indicates that the match optional. The matcher is still expected to run and
/// capture if successful. The parameter can be set to false
struct OptionalMatch : public SingleValuePredicateParam<bool> {
  OptionalMatch() : Base(true) {}
  explicit OptionalMatch(bool set) : Base(set) {}
};

/// Predicate tag indicating that the reduction is produced by a single combiner
/// operation.
struct SingleCombinerReduction {};

class StructuredOpMatcher;
StructuredOpMatcher m_StructuredOp();

/// Structured op matcher with additional predicates attachable through the
/// fluent, a.k.a. chainable, API. Note that public API must *not* accept
/// additional callbacks even; new predicates should be added instead when
/// necessary. Not only this decreases the depth of the callback stack and
/// increases readability, it also allows us to port the matcher to a
/// declarative format using PDL and/or Transform dialect in the future. The
/// latter will become impossible with arbitrary C++ callbacks.
class StructuredOpMatcher {
  friend StructuredOpMatcher m_StructuredOp();
  using PredicateFn = std::function<bool(linalg::LinalgOp)>;

  /// Matches a structured operation if the given predicate is satisfied.
  StructuredOpMatcher(PredicateFn &&firstPredicate) {
    predicates.push_back(std::move(firstPredicate));
  }

 public:
  /// Matches any structured operation, i.e., operation with LinalgOp interface.
  StructuredOpMatcher() {}

  /// Creates a matcher for a structured operation with one of the given types.
  template <typename... OpType>
  static StructuredOpMatcher create() {
    return StructuredOpMatcher(
        [](linalg::LinalgOp op) { return isa<OpType...>(op.getOperation()); });
  }

  /// Returns the matched operation if the match was successful.
  linalg::LinalgOp getCaptured() const { return captured; }

  /// Matches the given operation, hook for `matchPattern`.
  bool match(Operation *op);

  /// Adds a predicate checking that the given iteration space dimension is
  /// static/dynamic. The dimension index may be negative, in which case
  /// dimensions are counted from the last one (i.e. Python-style), or be an
  /// AllDims tag, in which case all dimensions are checked. This may be
  /// eventually extended to slices and/or lists of dimensions.
  StructuredOpMatcher &dim(int64_t dimension, ShapeKind kind);
  StructuredOpMatcher &dim(AllDims tag, ShapeKind kind);

  /// Adds a predicate checking that the given iteration space dimension has the
  /// given iterator type, e.g., parallel or reduction. The dimension index may
  /// be negative, in which case dimensions are counted from the last one
  /// (i.e. Python-style), or be an AllDims tag, in which case all dimensions
  /// are checked. This may be eventually extended to slices and/or lists of
  /// dimensions.
  StructuredOpMatcher &dim(int64_t dimension, utils::IteratorType kind);
  StructuredOpMatcher &dim(AllDims tag, utils::IteratorType kind);

  /// Adds a predicate checking that the given iteration space dimension is
  /// statically known to be divisible by the given value. The dimension index
  /// may be negative, in which case dimensions are counted from the last one
  /// (i.e. Python-style).
  StructuredOpMatcher &dim(int64_t dimension, DivisibleBy divisibleBy);

  /// Adds a predicate checking that the structured op has the given number of
  /// inputs.
  StructuredOpMatcher &input(NumEqualsTo num) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      return linalgOp.getNumDpsInputs() == num.value;
    });
    return *this;
  }

  /// Adds a predicate that recursively applies other predicates to the
  /// operation defining the `position`-th operand. The position may be
  /// negative, in which case positions are counted from the last one
  /// (i.e. Python-style). When the match is optional, the predicate check
  /// succeeds as long as the `position` is in bounds. The matcher is executed
  /// if there is a defining operation for the input operand.
  template <typename T>
  std::enable_if_t<
      llvm::is_detected<::mlir::detail::has_operation_or_value_matcher_t, T,
                        Operation *>::value,
      StructuredOpMatcher &>
  input(int64_t position, T &operandMatcher,
        OptionalMatch optional = OptionalMatch(false)) {
    predicates.push_back([position, optional,
                          &operandMatcher](linalg::LinalgOp linalgOp) -> bool {
      int64_t transformedPosition =
          position >= 0 ? position : linalgOp.getNumDpsInputs() + position;
      if (transformedPosition >= linalgOp.getNumDpsInputs()) return false;

      Operation *definingOp = linalgOp.getDpsInputOperand(transformedPosition)
                                  ->get()
                                  .getDefiningOp();
      // We MUST run the matcher at this point, even if the match is optional,
      // to allow for capture.
      if (operandMatcher.match(definingOp)) return true;
      return optional.value;
    });
    return *this;
  }

  /// Adds a predicate checking that all input operands of the structured op
  /// have a permutation indexing map.
  StructuredOpMatcher &input(AllOperands tag, IsPermutation);

  /// Adds a predicate checking that the structured op has the given number of
  /// outputs.
  StructuredOpMatcher &output(NumEqualsTo num) {
    predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
      return linalgOp.getNumDpsInits() == num.value;
    });
    return *this;
  }

  /// Adds a predicate checking that all output operands of the structured op
  /// have a permutation indexing map.
  StructuredOpMatcher &output(AllOperands tag, IsPermutation);

  /// Adds a predicate checking that the bit width of the elemental type of the
  /// structured op output at the given position is equal to the given value.
  StructuredOpMatcher &output(int64_t position, ElementTypeBitWidth width);

  /// Adds a predicate checking that the output of the structured op is produced
  /// by a reduction with a single-operation combinator (such as addf or mulf,
  /// but not a compare+select pair).
  StructuredOpMatcher &output(int64_t position, SingleCombinerReduction tag);

  /// Adds a predicate that recursively applies other predicates to the
  /// operation defining the init/out operand corresponding to `position`-th
  /// output. The position may be negative, in which case positions are counted
  /// from the last one (i.e. Python-style). When the match is optional, the
  /// predicate check succeeds as long as the `position` is in bounds. The
  /// matcher executed if there is a defining operation for the output operand.
  template <typename T>
  std::enable_if_t<
      llvm::is_detected<::mlir::detail::has_operation_or_value_matcher_t, T,
                        Operation *>::value,
      StructuredOpMatcher &>
  output(int64_t position, T &operandMatcher,
         OptionalMatch optional = OptionalMatch(false)) {
    predicates.push_back([position, optional,
                          &operandMatcher](linalg::LinalgOp linalgOp) -> bool {
      int64_t transformedPosition =
          position >= 0 ? position : linalgOp.getNumDpsInits() + position;
      if (transformedPosition >= linalgOp.getNumDpsInits()) return false;

      Operation *definingOp = linalgOp.getDpsInitOperand(transformedPosition)
                                  ->get()
                                  .getDefiningOp();
      if (!definingOp) return optional.value;
      // We MUST run the matcher at this point, even if the match is optional,
      // to allow for capture.
      if (operandMatcher.match(definingOp)) return true;
      return optional.value;
    });
    return *this;
  }

  /// Adds a predicate that recursively applies to users of the `position`-th
  /// result of the structured op. Succeeds if any user matches the predicate.
  /// When the match is optional, the predicate check succeeds as long as the
  /// `position` is in bounds, after running the given matcher.
  template <typename T>
  std::enable_if_t<
      llvm::is_detected<::mlir::detail::has_operation_or_value_matcher_t, T,
                        Operation *>::value,
      StructuredOpMatcher &>
  result(int64_t position, HasAnyUse tag, T &resultUserMatcher,
         OptionalMatch optional = OptionalMatch(false)) {
    predicates.push_back([&resultUserMatcher, optional,
                          position](linalg::LinalgOp linalgOp) -> bool {
      int64_t transformedPosition =
          position >= 0 ? position : linalgOp->getNumResults() + position;
      if (transformedPosition >= linalgOp->getNumResults()) return false;

      // We MUST run the matcher at this point, even if the match is optional,
      // to allow for capture.
      if (llvm::any_of(linalgOp->getResult(transformedPosition).getUsers(),
                       [&resultUserMatcher](Operation *op) {
                         return resultUserMatcher.match(op);
                       })) {
        return true;
      }
      return optional.value;
    });
    return *this;
  }

 private:
  /// Additional predicates to be checked on the structured op.
  SmallVector<PredicateFn> predicates;

  /// Matched value.
  linalg::LinalgOp captured = nullptr;
};

/// Creates a matcher of an arbitrary structured op.
inline StructuredOpMatcher m_StructuredOp() { return StructuredOpMatcher(); }

/// Creates a matcher of a structured op with kinds provided as template
/// arguments.
template <typename... OpType>
inline StructuredOpMatcher m_StructuredOp() {
  return StructuredOpMatcher::create<OpType...>();
}

}  // namespace mlir::iree_compiler::IREE::transform_dialect

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_TRANSFORMMATCHERS_H_
