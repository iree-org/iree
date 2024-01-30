// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STABLEHLO_IREE_CONVERSION_PASSDETAIL_H_
#define STABLEHLO_IREE_CONVERSION_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DECL
#include "stablehlo-iree/Conversion/Passes.h.inc"

} // namespace mlir::iree_compiler::stablehlo

#endif // STABLEHLO_IREE_CONVERSION_PASSDETAIL_H_
