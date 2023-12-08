// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

LinalgTransformationFilter::LinalgTransformationFilter(
    const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {
  if (f)
    filters.push_back(f);
}

LogicalResult LinalgTransformationFilter::checkAndNotify(RewriterBase &rewriter,
                                                         Operation *op) const {
  if (llvm::any_of(filters,
                   [&](const FilterFunction &f) { return failed(f(op)); }))
    return failure();

  auto attr = op->template getAttrOfType<StringAttr>(
      LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty() || matchByDefault)
      return success();

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction)
    if (attr.getValue() == filter)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void LinalgTransformationFilter::replaceLinalgTransformationFilter(
    RewriterBase &rewriter, Operation *op) const {
  if (replacement.has_value())
    op->setAttr(LinalgTransforms::kLinalgTransformMarker, replacement.value());
  else
    op->removeAttr(
        rewriter.getStringAttr(LinalgTransforms::kLinalgTransformMarker));
}

bool LinalgTransformationFilter::hasReplacementFilter(Operation *op) const {
  if (!replacement)
    return false;
  auto attr = op->getAttr(LinalgTransforms::kLinalgTransformMarker)
                  .dyn_cast<StringAttr>();
  return attr && attr == *replacement;
}

namespace detail {
#define GEN_PASS_REGISTRATION
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h.inc" // IWYU pragma: export
} // namespace detail

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

void IREE::LinalgExt::registerPasses() {
  IREE::LinalgExt::detail::registerPasses();
}
