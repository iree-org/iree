// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/Captures.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::IREE::HAL {

ValueOrigin categorizeValue(Value value) {
  // If this is a captured argument of an execution region then look up to the
  // SSA value that was captured.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto closureOp = dyn_cast<IREE::Util::ClosureOpInterface>(
            blockArg.getOwner()->getParentOp())) {
      return categorizeValue(
          closureOp.getClosureOperands()[blockArg.getArgNumber()]);
    }
  }

  // If we wanted to pull in entire IR slices this would have to use a
  // worklist (selects of globals based on globals, etc). For now this analysis
  // only looks at the value provided.
  auto *definingOp = value.getDefiningOp();
  if (definingOp && definingOp->hasTrait<OpTrait::ConstantLike>()) {
    // Op producing the value is constant-like and we should be able to
    // outline it by cloning.
    return ValueOrigin::LocalConstant;
  } else if (auto loadOp =
                 dyn_cast_if_present<IREE::Util::GlobalLoadOp>(definingOp)) {
    // We only support immutable global loads - mutable ones are dynamic
    // values that may change over time and we can't memoize with them.
    if (loadOp.isGlobalImmutable()) {
      return ValueOrigin::ImmutableGlobal;
    } else {
      return ValueOrigin::MutableGlobal;
    }
  } else {
    // Dynamic value that is only available at the memoization site.
    return ValueOrigin::Unknown;
  }
}

} // namespace mlir::iree_compiler::IREE::HAL
