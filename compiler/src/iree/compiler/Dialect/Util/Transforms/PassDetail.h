// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_UTIL_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

#define GEN_PASS_CLASSES
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"  // IWYU pragma: keep

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_UTIL_TRANSFORMS_PASS_DETAIL_H_
