// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/init_input_passes.h"

#include "iree/compiler/InputConversion/Common/Passes.h"

#ifdef IREE_HAVE_STABLEHLO_INPUT
#include "iree/compiler/InputConversion/StableHLO/Passes.h"
#endif // IREE_HAVE_STABLEHLO_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#endif // IREE_HAVE_TOSA_INPUT

namespace mlir {
namespace iree_compiler {

void registerInputPasses() {
  registerCommonInputConversionPasses();

#ifdef IREE_HAVE_STABLEHLO_INPUT
  stablehlo::registerStableHLOConversionPasses();
#endif // IREE_HAVE_STABLEHLO_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
  registerTOSAConversionPasses();
  registerTosaToArithPass();
  registerTosaToLinalgPass();
  registerTosaToTensorPass();
#endif // IREE_HAVE_TOSA_INPUT
}

} // namespace iree_compiler
} // namespace mlir
