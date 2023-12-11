// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSIONS_PASSES_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSIONS_PASSES_H_

#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertHALToVMPass(IREE::VM::TargetOptions targetOptions);

inline void registerHALConversionPasses() {
  createConvertHALToVMPass(IREE::VM::TargetOptions::FromFlags::get());
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSIONS_PASSES_H_
