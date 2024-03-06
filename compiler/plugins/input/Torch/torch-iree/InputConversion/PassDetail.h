// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORCH_IREE_INPUTCONVERSION_PASSDETAIL_H_
#define TORCH_IREE_INPUTCONVERSION_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_CLASSES
#include "compiler/plugins/input/Torch/torch-iree/InputConversion/Passes.h.inc"

} // namespace mlir::iree_compiler::TorchInput

#endif // TORCH_IREE_INPUTCONVERSION_PASSDETAIL_H_
