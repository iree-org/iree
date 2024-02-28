// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_C_DIALECTS_H
#define IREE_DIALECTS_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// IREEDialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(IREEInput, iree_input);

//===--------------------------------------------------------------------===//
// LinalgTransform
//===--------------------------------------------------------------------===//

/// Register all passes for LinalgTransform.
MLIR_CAPI_EXPORTED void mlirIREELinalgTransformRegisterPasses();

//===--------------------------------------------------------------------===//
// TransformDialect
//===--------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Transform, transform);

MLIR_CAPI_EXPORTED void ireeRegisterTransformExtensions(MlirContext context);

/// Register all passes for the transform dialect.
MLIR_CAPI_EXPORTED void mlirIREETransformRegisterPasses();

#ifdef __cplusplus
}
#endif

#endif // IREE_DIALECTS_C_DIALECTS_H
