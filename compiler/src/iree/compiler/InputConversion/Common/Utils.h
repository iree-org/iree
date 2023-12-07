// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace mlir::iree_compiler {

Value sumReduceDimensionSubset(ImplicitLocOpBuilder &rewriter, Value val,
                               Type accETy, ArrayRef<bool> is_reduction);

} // namespace mlir::iree_compiler
