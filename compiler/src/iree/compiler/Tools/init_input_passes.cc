// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/init_input_passes.h"

#include "iree/compiler/InputConversion/Common/Passes.h"

#ifdef IREE_HAVE_MHLO_DIALECTS
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#endif  // IREE_HAVE_MHLO_DIALECTS
#ifdef IREE_HAVE_TORCH_DIALECTS
#include "iree/compiler/InputConversion/TMTensor/Passes.h"
#endif  // IREE_HAVE_TORCH_DIALECTS
#ifdef IREE_HAVE_TOSA_DIALECTS
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#endif  // IREE_HAVE_TOSA_DIALECTS

namespace mlir {
namespace iree_compiler {

void registerInputPasses() {
  registerCommonInputConversionPasses();

#ifdef IREE_HAVE_MHLO_DIALECTS
  MHLO::registerMHLOConversionPasses();
#endif  // IREE_HAVE_MHLO_DIALECTS
#ifdef IREE_HAVE_TORCH_DIALECTS
  TMTensor::registerTMTensorConversionPasses();
#endif
#ifdef IREE_HAVE_TOSA_DIALECTS
  registerTOSAConversionPasses();
  registerTosaToArithPass();
  registerTosaToLinalgPass();
  registerTosaToTensorPass();
#endif  // IREE_HAVE_TOSA_DIALECTS
}

}  // namespace iree_compiler
}  // namespace mlir
