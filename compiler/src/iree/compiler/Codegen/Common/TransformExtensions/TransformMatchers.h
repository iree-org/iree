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

#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler::IREE::transform_dialect {

//===---------------------------------------------------------------------===//
// StructuredOpMatcher and predicates.
//===---------------------------------------------------------------------===//

class StructuredOpMatcher;
StructuredOpMatcher m_StructuredOp();

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

/// Indicates that it suffices for only a subset of an operand or result value
/// to be used.
struct SubsetOf {
  explicit SubsetOf(StructuredOpMatcher &matcher) : matcher(matcher) {}
  StructuredOpMatcher &matcher;
};

namespace detail {
template <typename T>
using has_reset_capture_t = decltype(std::declval<T>().resetCapture());
}  // namespace detail

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
  using CaptureResetFn = std::function<void()>;

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
      if (!definingOp) return optional.value;
      // We MUST run the matcher at this point, even if the match is optional,
      // to allow for capture.
      if (operandMatcher.match(definingOp)) return true;
      return optional.value;
    });
    recordNestedMatcher(operandMatcher);
    return *this;
  }

  /// Adds a predicate checking that all input operands of the structured op
  /// have a permutation indexing map.
  StructuredOpMatcher &input(AllOperands tag, IsPermutation);

  /// Adds a predicate that recursively applies another predicate to the
  /// operation defining the `position`-th input operand, looking through any
  /// "subsetting" operation such as "tensor.extract_slice".
  StructuredOpMatcher &input(int64_t position, SubsetOf subset);

  /// Adds a predicate that recursively applies another predicate to the
  /// operation defining the `position`-th output operand, looking through any
  /// "subsetting" operation such as "tensor.extract_slice".
  StructuredOpMatcher &output(int64_t position, SubsetOf subset);

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
    recordNestedMatcher(operandMatcher);
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
    recordNestedMatcher(resultUserMatcher);
    return *this;
  }

  /// Adds a predicate that recursively applies to users of the `positions`-th
  /// result, looking through any "subsetting" operation such as
  /// "tensor.extract_slice". Succeeds if any user matches the predicate.
  /// When the match is optional, the predicate check succeeds as long as the
  /// `position` is in bounds, after running the given matcher.
  StructuredOpMatcher &result(int64_t position, HasAnyUse tag, SubsetOf subset,
                              OptionalMatch optional = OptionalMatch(false));

  /// Resets the captured value to null. This should be called if the same
  /// pattern needs to be applied more than once as it may keep captured values
  /// for optional nested predicates from the previous application.
  void resetCapture() {
    captured = nullptr;
    for (const CaptureResetFn &fn : captureResetFns) fn();
  }

 private:
  /// Informs the matcher that it has another, nested matcher. Practically,
  /// records the captured value cleanup function so it runs when required.
  template <typename T>
  std::enable_if_t<llvm::is_detected<detail::has_reset_capture_t, T>::value>
  recordNestedMatcher(T &nested) {
    captureResetFns.push_back([&nested] { nested.resetCapture(); });
  }
  template <typename T>
  std::enable_if_t<!llvm::is_detected<detail::has_reset_capture_t, T>::value>
  recordNestedMatcher(T &nested) {}

  /// Additional predicates to be checked on the structured op.
  SmallVector<PredicateFn> predicates;

  /// Callbacks to reset captures of nested matchers.
  SmallVector<CaptureResetFn> captureResetFns;

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

//===---------------------------------------------------------------------===//
// MatchCallback functionality.
//===---------------------------------------------------------------------===//

/// Additional results of the C++ callback usable in the `match_callback`
/// transform operation. Conceptually, a list of lists of payload operations to
/// be associated with each result handle.
class MatchCallbackResult {
 public:
  /// Returns the number of lists of payload operations.
  unsigned getNumPayloadGroups() const { return payloadGroupLengths.size(); }

  /// Returns the `position`-th list of payload operations.
  ArrayRef<Operation *> getPayloadGroup(unsigned position) const;

  /// Adds a new list of payload operations to the list of lists. The new list
  /// must not contain null operations.
  template <typename Range>
  unsigned addPayloadGroup(Range operations) {
    int64_t originalLength = payloadOperations.size();
    assert(llvm::all_of(operations, [](Operation *op) -> bool { return op; }) &&
           "null operation");
    llvm::append_range(payloadOperations, operations);
    payloadGroupLengths.push_back(payloadOperations.size() - originalLength);
    return payloadGroupLengths.size() - 1;
  }
  void addPayloadGroup(ArrayRef<Operation *> operations) {
    addPayloadGroup<ArrayRef<Operation *>>(operations);
  }

  /// Adds a new singleton list of payload operation to the list of lists if the
  /// operation is non-null, adds an empty list otherwise. Useful for results of
  /// optional matches.
  void addPotentiallyEmptyPayloadGroup(Operation *op) {
    if (!op)
      addPayloadGroup(ArrayRef<Operation *>());
    else
      addPayloadGroup(ArrayRef<Operation *>(op));
  }

 private:
  /// The flat list of all payload opreations. `payloadGroupLengths` can be used
  /// to compute the sublist that corresponds to one nested list.
  // TODO: if somebody implements such a flattened vector generically, use it.
  SmallVector<Operation *> payloadOperations;
  SmallVector<int64_t> payloadGroupLengths;
};

/// A transform state extension that maintains the mapping between callback
/// names as strings usable in `match_callback` and their implementations.
class MatchCallbacksRegistry : public transform::TransformState::Extension {
 public:
  using MatchCallbackFn = std::function<DiagnosedSilenceableFailure(
      MatchCallbackResult &, Location, const transform::TransformState &,
      ValueRange)>;

  /// Constructs the extension.
  MatchCallbacksRegistry(transform::TransformState &state)
      : transform::TransformState::Extension(state) {}

  /// Registers the given function as a callback with the given name. The name
  /// must not be already present in the registry. The callback must be
  /// convertible to MatchCallbackFn.
  template <typename Fn>
  void registerCallback(StringRef name, Fn &&fn) {
    bool succeeded = callbacks.try_emplace(name, std::forward<Fn>(fn)).second;
    (void)succeeded;
    assert(succeeded && "adding a callback with a repeated name");
  }

  /// Returns a pointer to the implementation of the callback with the given
  /// name, or null if it is not present in the registry.
  const MatchCallbackFn *get(StringRef name) const {
    auto iter = callbacks.find(name);
    if (iter == callbacks.end()) return nullptr;
    return &iter->getValue();
  }

 private:
  llvm::StringMap<MatchCallbackFn> callbacks;
};

//===---------------------------------------------------------------------===//
// Case-specific matcher builders.
//===---------------------------------------------------------------------===//

/// Creates a group of matchers for:
///
///     trailing(reduction(leading(), fill()))
///
/// where trailing and leading are elementwise operations whose presence is
/// optional. Each matcher will capture the corresponding operation.
void makeGPUReductionMatcher(StructuredOpMatcher &reduction,
                             StructuredOpMatcher &fill,
                             StructuredOpMatcher &leading,
                             StructuredOpMatcher &trailing);

/// Creates a group of matchers for:
///
///     trailing(
///       combiner_reduction(
///         parallel_reduction(leading(), parallel_fill()),
///         original_fill())))
///
/// where trailing and leading are elementwise operations whose presence is
/// optional, and with subsetting ops potentially present on the operand use-def
/// chains.
void makeGPUSplitReductionMatcher(StructuredOpMatcher &parallel_reduction,
                                  StructuredOpMatcher &combiner_reduction,
                                  StructuredOpMatcher &parallel_fill,
                                  StructuredOpMatcher &original_fill,
                                  StructuredOpMatcher &leading,
                                  StructuredOpMatcher &trailing);

}  // namespace mlir::iree_compiler::IREE::transform_dialect

#endif  // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_TRANSFORMMATCHERS_H_
