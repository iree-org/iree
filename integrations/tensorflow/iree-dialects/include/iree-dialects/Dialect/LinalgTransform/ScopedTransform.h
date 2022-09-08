// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_SCOPEDTRANSFORM_H
#define IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_SCOPEDTRANSFORM_H

#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"

namespace mlir {
namespace linalg {
namespace transform {
ScopeOp wrapInScope(Operation *op);
FailureOr<SmallVector<Operation *>> unwrapScope(ScopeOp scope);

template <typename TransformT>
auto scoped(Operation *target, TransformT &&transform) {
  auto scope = wrapInScope(target);
  Operation &op = *scope.getBody().front().begin();
  auto result = transform(scope, &op);
  if (failed(unwrapScope(scope)) || failed(result))
    return decltype(result)(failure());
  return result;
}
} // namespace transform
} // namespace linalg
} // namespace mlir

#endif // IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_SCOPEDTRANSFORM_H
