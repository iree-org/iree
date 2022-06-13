// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/init_input_dialects.h"

#ifdef IREE_HAVE_MHLO_DIALECTS
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#endif  // IREE_HAVE_MHLO_DIALECTS
#ifdef IREE_HAVE_TORCH_DIALECTS
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#endif
#ifdef IREE_HAVE_TOSA_DIALECTS
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#endif  // IREE_HAVE_TOSA_DIALECTS

namespace mlir {
namespace iree_compiler {

void registerInputDialects(DialectRegistry &registry) {
#ifdef IREE_HAVE_MHLO_DIALECTS
  registry.insert<mlir::chlo::ChloDialect, mlir::mhlo::MhloDialect>();
#endif  // IREE_HAVE_MHLO_DIALECTS
#ifdef IREE_HAVE_TORCH_DIALECTS
  registry.insert<mlir::torch::TMTensor::TMTensorDialect>();
#endif  // IREE_HAVE_TORCH_DIALECTS
#ifdef IREE_HAVE_TOSA_DIALECTS
  registry.insert<tosa::TosaDialect>();
#endif  // IREE_HAVE_TOSA_DIALECTS
}

}  // namespace iree_compiler
}  // namespace mlir
