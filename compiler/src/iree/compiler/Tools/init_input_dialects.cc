// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/init_input_dialects.h"

#ifdef IREE_HAVE_STABLEHLO_INPUT
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif // IREE_HAVE_STABLEHLO_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#endif // IREE_HAVE_TOSA_INPUT

namespace mlir {
namespace iree_compiler {

void registerInputDialects(DialectRegistry &registry) {
#ifdef IREE_HAVE_STABLEHLO_INPUT
  registry.insert<mlir::chlo::ChloDialect, mlir::stablehlo::StablehloDialect>();
#endif // IREE_HAVE_STABLEHLO_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
  registry.insert<tosa::TosaDialect>();
#endif // IREE_HAVE_TOSA_INPUT
}

} // namespace iree_compiler
} // namespace mlir
