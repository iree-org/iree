// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASS_DETAIL_H_
#define IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASS_DETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

#define GEN_PASS_CLASSES
#include "iree_tf_compiler/TFL/Passes.h.inc"  // IWYU pragma: keep

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TENSORFLOW_IREE_TF_COMPILER_TFL_PASS_DETAIL_H_
