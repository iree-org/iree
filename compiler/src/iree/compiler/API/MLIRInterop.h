// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Interop facilities between the IREE compiler embedding API (embedding_api.h)
// and the MLIR C-API. This provides access to the MlirContext and ability to
// load in-memory modules without parsing.
// Stability: This API, like the MLIR C-API, offers no ABI stability guarantees,
// and API stability is best effort.

#ifndef IREE_COMPILER_API_MLIR_INTEROP_H
#define IREE_COMPILER_API_MLIR_INTEROP_H

#include "iree/compiler/embedding_api.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers all dialects and extensions known to the IREE compiler.
MLIR_CAPI_EXPORTED void ireeCompilerRegisterDialects(
    MlirDialectRegistry registry);

// Gets the MlirContext that the session manages. The context is owned by the
// session and valid until it is destroyed.
MLIR_CAPI_EXPORTED MlirContext
ireeCompilerSessionGetContext(iree_compiler_session_t *session);

// Imports an externally built MlirModule into the invocation as an alternative
// to parsing with |ireeCompilerInvocationParseSource|.
// Ownership of the moduleOp is transferred to the invocation, regardless of
// whether the call succeeds or fails.
// On failure, returns false and issues diagnostics.
MLIR_CAPI_EXPORTED bool ireeCompilerInvocationImportModule(
    iree_compiler_invocation_t *inv, MlirOperation moduleOp);

#ifdef __cplusplus
}
#endif

#endif  // IREE_COMPILER_API_MLIR_INTEROP_H
