// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_MHLO_PASSDETAIL_H_
#define IREE_COMPILER_INPUTCONVERSION_MHLO_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

#define GEN_PASS_CLASSES
#include "iree/compiler/InputConversion/MHLO/Passes.h.inc"

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_INPUTCONVERSION_MHLO_PASSDETAIL_H_
