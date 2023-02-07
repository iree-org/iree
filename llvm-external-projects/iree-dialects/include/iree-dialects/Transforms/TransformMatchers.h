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

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace transform_ext {

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

/// A predicate indicating the structured op matcher to check the predicate for
/// all dimensions except the specified ones.
struct AllDimsExcept {
  explicit AllDimsExcept(std::initializer_list<int64_t> range) {
    llvm::append_range(exceptions, range);
  }
  ArrayRef<int64_t> getExcluded() const { return llvm::ArrayRef(exceptions); }

private:
  SmallVector<int64_t> exceptions;
};

/// A placeholder indicating the structured op matcher to check the predicate
/// for all operands of the relevant kind.
struct AllOperands {};

/// Base class for single-value captures. Concrete captures should inherit this
/// and forward the constructor via `using Base::Base`.
template <typename T>
struct CaptureStaticValue {
  using Base = CaptureStaticValue<T>;
  explicit CaptureStaticValue(T &value) : value(value) {}
  T &value;
};

/// Captures the (static) size of the dimension.
struct CaptureDim : public CaptureStaticValue<int64_t> {
  using Base::Base;
};

/// Captures the (static) sizes of multiple dimensions.
struct CaptureDims : public CaptureStaticValue<SmallVector<int64_t>> {
  using Base::Base;
};

/// Captures the rank of the operation.
struct CaptureRank : public CaptureStaticValue<int64_t> {
  using Base::Base;
};

/// Captures the bitwidth of an element type.
struct CaptureElementTypeBitWidth : public CaptureStaticValue<int64_t> {
  using Base::Base;
};

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

/// Indicates that the number of entities must be greater than the given value.
struct NumGreaterEqualTo : public SingleValuePredicateParam<size_t> {
  using Base::Base;
};

/// Indicates that the number of entities must be greater than the given value.
struct NumLowerEqualTo : public SingleValuePredicateParam<size_t> {
  using Base::Base;
};

/// Indicates that the bit width of the elemental type must be equal to the give
/// value.
struct ElementTypeBitWidth : public SingleValuePredicateParam<size_t> {
  using Base::Base;
};

/// Predicate tag indicating that the affine map is a permutation.
struct IsPermutation {};

/// Predicate tag indicating that the affine map is a projected permutation.
struct IsProjectedPermutation {};

/// Indicates that the match optional. The matcher is still expected to run and
/// capture if successful. The parameter can be set to false
struct OptionalMatch : public SingleValuePredicateParam<bool> {
  OptionalMatch() : Base(true) {}
  explicit OptionalMatch(bool set) : Base(set) {}
};

/// Predicate tag indicating that the reduction is produced by a single combiner
/// operation.
struct SingleCombinerReduction {};

namespace detail {
template <typename T>
using has_reset_capture_t = decltype(std::declval<T>().resetCapture());
template <typename T>
using has_get_capture_t = decltype(std::declval<T>().getCaptured());
} // namespace detail

/// Base class for op matchers that capture the matched operation.
class CapturingOpMatcher {
public:
  virtual ~CapturingOpMatcher() = default;

  /// Resets the captured value to null. This should be called if the same
  /// pattern needs to be applied more than once as it may keep captured values
  /// for optional nested predicates from the previous application.
  void resetCapture() {
    captured = nullptr;
    SmallVector<CapturingOpMatcher *> nested;
    getAllNested(nested);
    for (CapturingOpMatcher *matcher : nested) {
      matcher->captured = nullptr;
    }
  }

  /// Returns the matched operation if the match was successful.
  Operation *getCaptured() const { return captured; }

protected:
  /// Informs the matcher that it has another, nested matcher. Derived classes
  /// must call this to keep track of nested matchers for capture resetting
  /// purposes.
  template <typename T>
  void recordNestedMatcher(T &nested) {
    if constexpr (std::is_base_of_v<CapturingOpMatcher, T>)
      nestedCapturingMatchers.push_back(&nested);
  }

  /// Appends all nested capturing matchers, excluding this one, to `nested`.
  void getAllNested(SmallVectorImpl<CapturingOpMatcher *> &nested);

private:
  /// A list of (recursively) nested capturing matchers that should be reset
  /// when the current matcher is.
  SmallVector<CapturingOpMatcher *> nestedCapturingMatchers;

protected:
  /// Matched value.
  linalg::LinalgOp captured = nullptr;
};

/// Structured op matcher with additional predicates attachable through the
/// fluent, a.k.a. chainable, API. Note that public API must *not* accept
/// additional callbacks even; new predicates should be added instead when
/// necessary. Not only this decreases the depth of the callback stack and
/// increases readability, it also allows us to port the matcher to a
/// declarative format using PDL and/or Transform dialect in the future. The
/// latter will become impossible with arbitrary C++ callbacks.
class StructuredOpMatcher : public CapturingOpMatcher {
  friend StructuredOpMatcher m_StructuredOp();
  using PredicateFn = std::function<bool(linalg::LinalgOp)>;
  using CaptureResetFn = std::function<void()>;
  using GetCapturedFn = std::function<Operation *()>;

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
    return StructuredOpMatcher([](linalg::LinalgOp op) {
      debugOutputForCreate(ArrayRef<StringRef>{OpType::getOperationName()...});
      return isa<OpType...>(op.getOperation());
    });
  }

  /// Matches the given operation, hook for `matchPattern`.
  bool match(Operation *op);

  //===-------------------------------------------------------------------===//
  // Constraints on op rank and dims.
  //===-------------------------------------------------------------------===//
  /// Adds a predicate checking that the given rank must be greater than some
  /// constant value.
  // TODO: Base class, derived class and proper API.
  StructuredOpMatcher &rank(NumGreaterEqualTo minRank);
  StructuredOpMatcher &rank(NumLowerEqualTo maxRank);

  /// Adds a predicate checking that the given iteration space dimension is
  /// static/dynamic. The dimension index may be negative, in which case
  /// dimensions are counted from the last one (i.e. Python-style), or be an
  /// AllDims tag, in which case all dimensions are checked. This may be
  /// eventually extended to slices and/or lists of dimensions.
  StructuredOpMatcher &dim(int64_t dimension, ShapeKind kind) {
    return dim(SmallVector<int64_t>{dimension}, kind);
  }
  StructuredOpMatcher &dim(SmallVector<int64_t> &&dimensions, ShapeKind kind);
  StructuredOpMatcher &dim(AllDims tag, ShapeKind kind);

  /// Adds a predicate checking that the given iteration space dimension has the
  /// given iterator type, e.g., parallel or reduction. The dimension index may
  /// be negative, in which case dimensions are counted from the last one
  /// (i.e. Python-style), or be an AllDims tag, in which case all dimensions
  /// are checked. This may be eventually extended to slices and/or lists of
  /// dimensions.
  StructuredOpMatcher &dim(int64_t dimension, utils::IteratorType kind) {
    return dim(SmallVector<int64_t>{dimension}, kind);
  }
  // Ownership may get tricky here so we wrap in an explicit vector.
  StructuredOpMatcher &dim(SmallVector<int64_t> &&dimensions,
                           utils::IteratorType kind);
  StructuredOpMatcher &dim(AllDims tag, utils::IteratorType kind);
  StructuredOpMatcher &dim(AllDimsExcept &&dimensions,
                           utils::IteratorType kind);

  /// Adds a predicate checking that the given iteration space dimension is
  /// statically known to be divisible by the given value. The dimension index
  /// may be negative, in which case dimensions are counted from the last one
  /// (i.e. Python-style).
  StructuredOpMatcher &dim(int64_t dimension, DivisibleBy divisibleBy);

  //===-------------------------------------------------------------------===//
  // Capture directives.
  //===-------------------------------------------------------------------===//
  StructuredOpMatcher &rank(CaptureRank capture);
  StructuredOpMatcher &dim(int64_t dimension, CaptureDim capture);
  StructuredOpMatcher &dim(AllDims tag, CaptureDims captures);

  //===-------------------------------------------------------------------===//
  // Constraints on input operands.
  //===-------------------------------------------------------------------===//
  /// Adds a predicate checking that the structured op has the given number of
  /// inputs.
  StructuredOpMatcher &input(NumEqualsTo num);

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
    addInputMatcher(
        position,
        [&operandMatcher](Operation *op) { return operandMatcher.match(op); },
        optional);
    recordNestedMatcher(operandMatcher);
    return *this;
  }

  /// Adds a predicate checking that all input operands of the structured op
  /// have a permutation indexing map.
  StructuredOpMatcher &input(AllOperands tag, IsPermutation);

  /// Adds a predicate checking that all input operands of the structured op
  /// have a projected permutation indexing map.
  StructuredOpMatcher &input(AllOperands tag, IsProjectedPermutation);

  /// Adds a predicate checking that the bit width of the elemental type of the
  /// structured op input at the given position is equal to the given value.
  StructuredOpMatcher &input(int64_t position, ElementTypeBitWidth width);

  /// Capture the elemental type bitwidth of input operand `position`.
  StructuredOpMatcher &input(int64_t position,
                             CaptureElementTypeBitWidth width);

  //===-------------------------------------------------------------------===//
  // Constraints on adjacent ops.
  //===-------------------------------------------------------------------===//

  /// Adds a predicate checking that all ops implementing TilingInterface in the
  /// parent of the given type (e.g., a function or a module) were matched by
  /// this or nested matchers. This is useful to ensure that the matcher covered
  /// the entire parent region, not just a parent of it. This predicate **must**
  /// be added *after* all the other predicates that capture.
  template <typename OpTy>
  StructuredOpMatcher &allTilableOpsCaptured() {
    SmallVector<CapturingOpMatcher *> copy;
    copy.push_back(this);
    getAllNested(copy);
    predicates.push_back([copy = std::move(copy)](linalg::LinalgOp linalgOp) {
      Operation *parent = linalgOp->getParentOfType<OpTy>();
      return checkAllTilableMatched(parent, linalgOp, copy);
    });
    return *this;
  }

  //===-------------------------------------------------------------------===//
  // Constraints on output operands.
  //===-------------------------------------------------------------------===//

  /// Adds a predicate checking that the structured op has the given number of
  /// outputs.
  StructuredOpMatcher &output(NumEqualsTo num);

  /// Adds a predicate checking that all output operands of the structured op
  /// have a permutation indexing map.
  StructuredOpMatcher &output(AllOperands tag, IsPermutation);

  /// Adds a predicate checking that all output operands of the structured op
  /// have a projected permutation indexing map.
  StructuredOpMatcher &output(AllOperands tag, IsProjectedPermutation);

  /// Adds a predicate checking that the bit width of the elemental type of the
  /// structured op output at the given position is equal to the given value.
  StructuredOpMatcher &output(int64_t position, ElementTypeBitWidth width);

  /// Capture the elemental type bitwidth of output operand `position`.
  StructuredOpMatcher &output(int64_t position,
                              CaptureElementTypeBitWidth width);

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
    addOutputMatcher(
        position,
        [&operandMatcher](Operation *op) { return operandMatcher.match(op); },
        optional);
    recordNestedMatcher(operandMatcher);
    return *this;
  }

  //===-------------------------------------------------------------------===//
  // Constraints on results.
  //===-------------------------------------------------------------------===//

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
    addResultMatcher(
        position, tag,
        [&resultUserMatcher](Operation *op) {
          return resultUserMatcher.match(op);
        },
        optional);
    recordNestedMatcher(resultUserMatcher);
    return *this;
  }

private:
  /// Checks that `matchers` captured all tilable ops nested in `parent` except
  /// for `linalgOp`. This is an implementation detail of allTilableOpsCaptured.
  static bool checkAllTilableMatched(Operation *parent,
                                     linalg::LinalgOp linalgOp,
                                     ArrayRef<CapturingOpMatcher *> matchers);

  /// Produce the debug output for `create` method in a non-templated way.
  static void debugOutputForCreate(ArrayRef<StringRef> opNames);

  /// Non-template implementations of nested predicate builders for inputs,
  /// outputs and results. Should not be called directly.
  void addInputMatcher(int64_t position,
                       std::function<bool(Operation *)> matcher,
                       OptionalMatch optional);
  void addOutputMatcher(int64_t position,
                        std::function<bool(Operation *)> matcher,
                        OptionalMatch optional);
  void addResultMatcher(int64_t position, HasAnyUse tag,
                        std::function<bool(Operation *)> matcher,
                        OptionalMatch optional);

  /// Additional predicates to be checked on the structured op.
  SmallVector<PredicateFn> predicates;
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
  int64_t getNumPayloadGroups() const { return payloadGroupLengths.size(); }

  /// Returns the `position`-th list of payload operations.
  ArrayRef<Operation *> getPayloadGroup(int64_t position) const;

  /// Adds a new list of payload operations to the list of lists. The new list
  /// must not contain null operations.
  template <typename Range>
  int64_t addPayloadGroup(Range operations) {
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
    if (iter == callbacks.end())
      return nullptr;
    return &iter->getValue();
  }

private:
  llvm::StringMap<MatchCallbackFn> callbacks;
};

//===---------------------------------------------------------------------===//
// Case-specific matcher builders.
//===---------------------------------------------------------------------===//

struct MatchedReductionCaptures {
  int64_t reductionRank = 0;
  int64_t maybeLeadingRank = 0;
  int64_t maybeTrailingRank = 0;
  SmallVector<int64_t> leadingOpSizes = {};
  SmallVector<int64_t> reductionOpSizes = {};
  SmallVector<int64_t> trailingOpSizes = {};
  int64_t reductionOutputElementalTypeBitWidth = 0;
  int64_t maybeLeadingOutputElementalTypeBitWidth = 0;
  int64_t maybeTrailingOutputElementalTypeBitWidth = 0;
};

/// Creates a group of matchers for:
///
///     trailing(reduction(leading(), fill()))
///
/// where trailing and leading are elementwise operations whose presence is
/// optional. Each matcher will capture the corresponding operation.
void makeReductionMatcher(StructuredOpMatcher &reduction,
                          StructuredOpMatcher &fill,
                          StructuredOpMatcher &leading,
                          StructuredOpMatcher &trailing,
                          MatchedReductionCaptures &captures);

} // namespace transform_ext
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_TRANSFORMMATCHERS_H_
