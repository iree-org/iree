// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECTS_IREE_CODEGEN_H
#define IREE_COMPILER_DIALECTS_IREE_CODEGEN_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// The following C API is **NOT STABLE** and likely to change in the future.
// It mirrors the IREE Codegen Dialect which is not stable itself.

MLIR_CAPI_EXPORTED bool
ireeAttributeIsACodegenDispatchLoweringPassPipelineAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirTypeID
ireeCodegenDispatchLoweringPassPipelineAttrGetTypeID(void);

MLIR_CAPI_EXPORTED MlirAttribute ireeCodegenDispatchLoweringPassPipelineAttrGet(
    MlirContext mlirCtx, uint32_t value);

MLIR_CAPI_EXPORTED
uint32_t
ireeCodegenDispatchLoweringPassPipelineAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_DIALECTS_IREE_CODEGEN_H
