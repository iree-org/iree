// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_TMTENSOR_PASSDETAIL_H_
#define IREE_COMPILER_INPUTCONVERSION_TMTENSOR_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace TMTensor {

#define GEN_PASS_CLASSES
#include "iree/compiler/InputConversion/TMTensor/Passes.h.inc"

}  // namespace TMTensor
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_INPUTCONVERSION_TMTENSOR_PASSDETAIL_H_
