// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOSTATUSCHECK_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOSTATUSCHECK_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::stablehlo {

// Fails the invocation with FAILED_PRECONDITION and |message| unless the i1
// |condition| holds.
void emitStatusCheck(OpBuilder &builder, Location loc, Value condition,
                     StringRef message);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOSTATUSCHECK_H_
