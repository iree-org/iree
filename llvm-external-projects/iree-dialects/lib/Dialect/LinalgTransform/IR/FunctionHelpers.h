//===-- FunctionalHelpers.h - Function rewrite helpers --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

namespace mlir {
namespace linalg {

// Pure C++ functional patterns requires some type plumbing.
namespace detail {
template <typename OpT>
struct ConvertOrForward {
  static OpT to(LinalgOp op) { return cast<OpT>(op.getOperation()); }
  static LinalgOp from(OpT op) { return cast<LinalgOp>(op.getOperation()); }
};
template <>
struct ConvertOrForward<LinalgOp> {
  static LinalgOp to(LinalgOp op) { return op; }
  static LinalgOp from(LinalgOp op) { return op; }
};
} // namespace detail

/// Wrap a call to a Linalg pattern where the input is a `LinalgOp` and the
/// output is a `LinalgOp`.
template <typename FunctionalLinalgPattern, typename... Args>
auto callLinalgPattern(Args &&...args) {
  FunctionalLinalgPattern pattern(std::forward<Args>(args)...);
  using Traits = llvm::function_traits<
      decltype(&FunctionalLinalgPattern::returningMatchAndRewrite)>;
  using OpT = typename Traits::template arg_t<0>;
  return
      [pattern = std::move(pattern)](
          LinalgOp linalgOp, PatternRewriter &rewriter) -> FailureOr<LinalgOp> {
        OpT op = detail::ConvertOrForward<OpT>::to(linalgOp);
        auto result = pattern.returningMatchAndRewrite(op, rewriter);
        if (failed(result))
          return failure();
        return detail::ConvertOrForward<decltype(*result)>::from(*result);
      };
}

} // namespace linalg
} // namespace mlir
