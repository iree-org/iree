// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOVM_PATTERNS_H_
#define IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOVM_PATTERNS_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Populates conversion patterns from the HAL dialect to the VM dialect.
void populateHALToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                             RewritePatternSet &patterns,
                             TypeConverter &typeConverter);

// Creates a !vm.buffer containing all of the |constantValues|.
Value createPackedConstantBuffer(Location loc, ValueRange constantValues,
                                 OpBuilder &builder);

// Creates a vm.rodata containing the contents of a hal.executable.binary.
IREE::VM::RodataOp
createExecutableBinaryRodata(IREE::HAL::ExecutableBinaryOp binaryOp,
                             OpBuilder &builder);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_HAL_CONVERSION_HALTOVM_PATTERNS_H_
