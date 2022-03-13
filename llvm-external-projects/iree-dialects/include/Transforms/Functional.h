//===- Functional.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_FUNCTIONAL_H
#define MLIR_REWRITE_FUNCTIONAL_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace functional {

/// A "Functional Pattern" is a callable concept that accepts as its first
/// argument an operation or operation interface and as its second argument a
/// `RewriterBase` or `PatternRewriter`. Beyond these, it can accept additional
/// parameters of any type.
///
/// A functional pattern is expected to return a type convertible to
/// `LogicalResult`. If the result is a `FailureOr<T>`, then `T` is forwarded to
/// subsequent patterns in sequences. Additionally, if `T` is a pair or tuple,
/// its elements are unpacked and passed as separate arguments to subsequent
/// patterns.
template <typename PatternT>
struct PatternConcept {
  static constexpr bool verify() {
    using Traits = llvm::function_traits<std::decay_t<PatternT>>;
    static_assert(Traits::num_args >= 2,
                  "Patterns must have at least two arguments.");

    using OpT = typename Traits::template arg_t<0>;
    static_assert(std::is_convertible<OpT, Operation *>::value,
                  "The first argument of a pattern must be Operation * or "
                  "convertible to Operation *");

    using RewriterT = typename Traits::template arg_t<1>;
    static_assert(std::is_convertible<PatternRewriter &, RewriterT>::value,
                  "The second argument of a pattern must be convertible from "
                  "PatternRewriter & (e.g. RewriterBase &)");

    using ResultT = typename Traits::result_t;
    static_assert(
        std::is_convertible<ResultT, LogicalResult>::value,
        "The result of a pattern must be convertible to LogicalResult");

    return true;
  }
};

namespace detail {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // end namespace detail

/// Apply a pattern directly on an operation. This function instantiates a
/// simple pattern rewriter and calls the pattern directly on the operation with
/// the given arguments.
template <typename OpT, typename PatternT, typename... Args,
          bool = PatternConcept<PatternT>::verify()>
auto applyAt(OpT op, PatternT &&pattern, Args &&...args) {
  detail::SimpleRewriter rewriter(op->getContext());
  rewriter.setInsertionPoint(op);
  return pattern(op, rewriter, std::forward<Args>(args)...);
}

/// Given a scope, apply a pattern with the given arguments until the first
/// successful match and return the result. This function instantiates a simple
/// pattern rewriter.
template <typename PatternT, typename... Args,
          bool = PatternConcept<PatternT>::verify()>
auto applyOnceIn(Operation *scope, PatternT &&pattern, Args &&...args) {
  assert(scope->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "scope is not isolated from above");
  using Traits = llvm::function_traits<std::decay_t<PatternT>>;
  using OpT = typename Traits::template arg_t<0>;

  detail::SimpleRewriter rewriter(scope->getContext());
  typename Traits::result_t result = failure();
  scope->walk([pattern = std::forward<PatternT>(pattern), &result, &rewriter,
               &args...](OpT op) {
    rewriter.setInsertionPoint(op);
    result = pattern(op, rewriter, std::forward<Args>(args)...);
    return failed(result) ? WalkResult::advance() : WalkResult::interrupt();
  });
  return result;
}

namespace detail {
/// If a pattern returns `FailureOr<Type>`, unpack the nested value of `Type`.
/// Otherwise, just return the whole value.
template <typename ReturnType>
struct UnpackFailureOr {
  using type = ReturnType;
  /// Base case. If a pattern does not return `FailureOr`, just forward the
  /// whole result. Usually, this is a `LogicalResult`.
  static type unpack(type &&value) { return value; }
};
template <typename NestedType>
struct UnpackFailureOr<FailureOr<NestedType>> {
  using type = NestedType;
  /// Specialized case for `FailureOr`. Assumes that the pattern succeeded.
  /// Return the contained value.
  static type unpack(FailureOr<type> &&value) { return std::move(*value); }
};
} // end namespace detail

/// Given a scope, apply a pattern once per operation in the scope, saving the
/// result of each match. The result list is empty if the pattern failed to
/// match at all.
template <typename PatternT, typename... Args,
          bool = PatternConcept<PatternT>::verify()>
auto applyForEachIn(Operation *scope, PatternT &&pattern, Args &&...args) {
  using Traits = llvm::function_traits<std::decay_t<PatternT>>;
  using Unpack = detail::UnpackFailureOr<typename Traits::result_t>;

  assert(scope->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "scope is not isolated from above");

  detail::SimpleRewriter rewriter(scope->getContext());
  // A list of all the results.
  SmallVector<typename Unpack::type> results;
  scope->walk([pattern = std::forward<PatternT>(pattern), &rewriter, &results,
               &args...](Operation *op) {
    rewriter.setInsertionPoint(op);
    auto result = pattern(op, rewriter, std::forward<Args>(args)...);
    // If the pattern applied, unpack the result and store it.
    if (succeeded(result))
      results.push_back(Unpack::unpack(std::move(result)));
  });
  return results;
}

/// Apply a pattern directly on an operation, for each operation in a list.
template <typename ListT, typename PatternT, typename... Args,
          bool = PatternConcept<PatternT>::verify()>
auto applyForEach(ListT &&list, PatternT &&pattern, Args &&...args) {
  using Traits = llvm::function_traits<std::decay_t<PatternT>>;
  using Unpack = detail::UnpackFailureOr<typename Traits::result_t>;

  // A list of all the results.
  SmallVector<typename Unpack::type> results;
  for (auto op : list) {
    auto result = applyAt(op, std::forward<PatternT>(pattern),
                          std::forward<Args>(args)...);
    // The pattern applied, unpack the result and store it.
    if (succeeded(result))
      results.push_back(Unpack::unpack(std::move(result)));
  }
  return results;
}

namespace detail {
/// Utility struct for handling functional patterns that may operate on generic
/// `Operation *` or a more specific interface or op type. In the base case,
/// patterns need to check that the correct type was passed and may need to cast
/// to that type.
template <typename OpT>
struct IsaOr {
  static bool apply(Operation *op) { return isa<OpT>(op); }
  static OpT cast(Operation *op) { return ::mlir::cast<OpT>(op); }
  static OpT dyn_cast(Operation *op) { return ::mlir::dyn_cast<OpT>(op); }
};
/// In the special case, nothing needs to be done. Just pass the generic op
/// directly into the pattern.
template <>
struct IsaOr<Operation *> {
  static bool apply(Operation *op) { return true; }
  static Operation *cast(Operation *op) { return op; }
  static Operation *dyn_cast(Operation *op) { return op; }
};

/// A sequence here is a tuple of unique functions. However, when constructing
/// the sequence, the result types of subsequent functions are not visible. In
/// order to generically pass around the entire sequence, it is stored as a list
/// of opaque pointers.
using OpaqueSeq = ArrayRef<void *>;

/// Unpack a non-tuple return type into a tuple.
template <typename ResultT>
struct UnpackIntoTuple {
  static auto apply(ResultT &&result) {
    return std::make_tuple(std::forward<ResultT>(result));
  }
};
/// Unpack a pair into a tuple.
template <typename FirstT, typename SecondT>
struct UnpackIntoTuple<std::pair<FirstT, SecondT>> {
  static auto apply(std::pair<FirstT, SecondT> &&result) {
    return std::make_tuple(std::move(result.first), std::move(result.second));
  }
};
/// If the result type is already a tuple, just forward it.
template <typename... ResultTs>
struct UnpackIntoTuple<std::tuple<ResultTs...>> {
  static auto apply(std::tuple<ResultTs...> &&result) {
    return std::forward<std::tuple<ResultTs...>>(result);
  }
};

/// Utility function for calling another pattern in the sequence where the
/// arguments are packed into a tuple. Similar to `std::apply`, except the first
/// argument is passed as the operation.
template <typename PatternT, typename OpT, typename Args, size_t... Indices>
auto callNextPattern(PatternT &&pattern, OpT op, PatternRewriter &rewriter,
                     Args &&args, std::index_sequence<Indices...>) {
  rewriter.setInsertionPoint(op);
  return pattern(op, rewriter, std::move(std::get<Indices + 1>(args))...);
}

/// A pattern sequence is implemented as a tuple of unique functions.
template <typename... UniqueFunctionTs>
struct GenericSequence : public std::tuple<UniqueFunctionTs...> {
  /// Inherit the tuple constructor.
  using std::tuple<UniqueFunctionTs...>::tuple;
  /// The number of patterns in the sequence.
  static constexpr size_t NumPatterns = sizeof...(UniqueFunctionTs);

  /// Get the equivalent tuple type, for use with tuple type utilities.
  template <typename...>
  struct GenericToTupleType;
  template <typename... Ts>
  struct GenericToTupleType<GenericSequence<Ts...>> {
    using type = std::tuple<Ts...>;
  };

  /// Create a new sequence that contains all the patterns of an existing
  /// sequence but appended with a new pattern. The previous sequence is
  /// invalidated.
  template <typename ResultT, typename PrevT, size_t... Indices>
  static auto moveInto(PrevT &&prev, std::index_sequence<Indices...>) {
    return GenericSequence<
        std::tuple_element_t<Indices,
                             typename GenericToTupleType<PrevT>::type>...,
        llvm::unique_function<ResultT(Operation *, PatternRewriter &,
                                      OpaqueSeq)>>(
        std::move(std::get<Indices>(prev))...,
        // Populate an empty function and define it later.
        llvm::unique_function<ResultT(Operation *, PatternRewriter &,
                                      OpaqueSeq)>());
  }

  /// Chain a pattern with another pattern. When calling `seq.then(...)`, the
  /// results of the previous pattern are passed to the subsequent pattern as
  /// follows:
  /// - if the pattern failed, then sequence execution is aborted
  /// - the first result must be convertible to `Operation *`, and is passed as
  ///   the operation into the next pattern
  /// - additional results are passed as arguments
  template <typename PatternT, bool = PatternConcept<PatternT>::verify()>
  auto then(PatternT &&pattern) {
    using Traits = llvm::function_traits<std::decay_t<PatternT>>;
    using OpT = typename Traits::template arg_t<0>;
    using ResultT = typename Traits::result_t;

    // Get the type of the previous pattern fucntion.
    using PrevT =
        std::remove_reference_t<decltype(std::get<NumPatterns - 1>(*this))>;
    // Copy all the patterns into a new sequence.
    auto seq = moveInto<ResultT>(std::move(*this),
                                 std::make_index_sequence<NumPatterns>());
    // Get a reference to the n
    auto &next = std::get<NumPatterns>(seq);
    // Each pattern in the sequence calls the previous pattern, except the first
    // pattern.
    next = [pattern = std::forward<PatternT>(pattern)](
               Operation *op, PatternRewriter &rewriter,
               OpaqueSeq opaqueSeq) -> ResultT {
      // FIXME: this is a hack to get around knowing all the return types.
      auto prevResult =
          (*(PrevT *)opaqueSeq[NumPatterns - 1])(op, rewriter, opaqueSeq);
      if (failed(prevResult))
        return failure();

      // The previous pattern succeeded. Unpack the results into a tuple to pass
      // as arguments to the next pattern.
      auto args =
          UnpackIntoTuple<std::remove_reference_t<decltype(*prevResult)>>::
              apply(std::move(*prevResult));
      // Grab the first result value as the operation.
      Operation *nextOp = std::get<0>(args);
      if (!detail::IsaOr<OpT>::apply(nextOp))
        return failure();
      // Call the next pattern.
      return callNextPattern(
          pattern, detail::IsaOr<OpT>::cast(nextOp), rewriter, std::move(args),
          std::make_index_sequence<std::tuple_size<decltype(args)>::value -
                                   1>());
    };
    return seq;
  }

  // Convert a generic sequence into an opaque sequence.
  template <typename SequenceT, size_t... Indices>
  static auto getAsOpaqueSeq(SequenceT &seq, std::index_sequence<Indices...>) {
    SmallVector<void *> opaqueSeq = {(void *)&std::get<Indices>(seq)...};
    return opaqueSeq;
  }

  // Implement the call operator using tail-recursion.
  auto operator()(Operation *op, PatternRewriter &rewriter) {
    return std::get<NumPatterns - 1>(*this)(
        op, rewriter,
        getAsOpaqueSeq(*this, std::make_index_sequence<NumPatterns>()));
  }
};

/// The starting point for constructing a pattern sequence.
struct SequenceBuilder {
  /// The first pattern of the sequence receives arguments directly from the
  /// caller and does not recurse.
  template <typename PatternT, typename... Args,
            bool = PatternConcept<PatternT>::verify()>
  auto begin(PatternT &&pattern, Args &&...args) {
    using Traits = llvm::function_traits<std::decay_t<PatternT>>;
    using OpT = typename Traits::template arg_t<0>;
    using ResultT = typename Traits::result_t;

    /// Create the function that calls the pattern.
    llvm::unique_function<ResultT(Operation *, PatternRewriter &, OpaqueSeq)>
        call = [pattern = std::forward<PatternT>(pattern),
                &args...](Operation *op, PatternRewriter &rewriter,
                          OpaqueSeq) -> ResultT {
      if (!detail::IsaOr<OpT>::apply(op))
        return failure();
      rewriter.setInsertionPoint(op);
      return pattern(detail::IsaOr<OpT>::cast(op), rewriter,
                     std::forward<Args>(args)...);
    };

    /// Insert it into a generic sequence and return.
    return GenericSequence<llvm::unique_function<ResultT(
        Operation *, PatternRewriter &, OpaqueSeq)>>(std::move(call));
  }
};

} // end namespace detail

struct SequenceBuilder : public detail::SequenceBuilder {};

} // end namespace functional
} // end namespace mlir

#endif // MLIR_REWRITE_FUNCTIONAL_H
