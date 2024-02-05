// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOSA_IREE_INPUTCONVERSION_PASSDETAIL_H_
#define TOSA_IREE_INPUTCONVERSION_PASSDETAIL_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_CLASSES
#include "tosa-iree/InputConversion/Passes.h.inc"

} // namespace mlir::iree_compiler

#endif // TOSA_IREE_INPUTCONVERSION_PASSDETAIL_H_
