// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Dialects.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

//===--------------------------------------------------------------------===//
// IREELinalgTransform
//===--------------------------------------------------------------------===//

void mlirIREELinalgTransformRegisterPasses() {
  mlir::linalg::transform::registerDropSchedulePass();
}

//===--------------------------------------------------------------------===//
// TransformDialect
//===--------------------------------------------------------------------===//

void ireeRegisterTransformExtensions(MlirContext context) {
  MLIRContext *ctx = unwrap(context);
  DialectRegistry registry;
  registry
      .addExtensions<mlir::transform_ext::StructuredTransformOpsExtension>();
  ctx->appendDialectRegistry(registry);
}

void mlirIREETransformRegisterPasses() {
  mlir::linalg::transform::registerDropSchedulePass();
}
