// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_C_UTILS_H
#define IREE_DIALECTS_C_UTILS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: Upstream C/Python APIs for symbol table.
// Looks up the referrent operation with the given flat symbol, starting from
// a specific op.
MLIR_CAPI_EXPORTED MlirOperation
ireeLookupNearestSymbolFrom(MlirOperation fromOp, MlirAttribute symbolRefAttr);

#ifdef __cplusplus
}
#endif

#endif // IREE_DIALECTS_C_UTILS_H
