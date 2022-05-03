// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

namespace mlir {
namespace iree_compiler {

/// Expose the implementation of the set num workgroups pass as a free function
/// because passes are surprisingly hard to apply reliably when they need to
/// anchor on special (i.e. non-Module) ops.
LogicalResult setNumWorkgroupsImpl(IREE::HAL::ExecutableVariantOp variantOp,
                                   ArrayRef<int64_t> workloadPerWorkgroup);

}  // namespace iree_compiler
}  // namespace mlir
