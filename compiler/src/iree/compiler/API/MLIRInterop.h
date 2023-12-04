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
MLIR_CAPI_EXPORTED void
ireeCompilerRegisterDialects(MlirDialectRegistry registry);

// Performs post-creation initialization of an externally derived context.
// This configures things such as threading behavior. Dialect registration
// is done via ireeCompilerRegisterDialects.
MLIR_CAPI_EXPORTED void ireeCompilerInitializeContext(MlirContext context);

// Gets the MlirContext that the session manages. The context is owned by the
// session and valid until it is destroyed.
// This implicitly "activates" the session: make sure that any configuration
// (flags, etc) has been done prior. Activation is lazy and is usually done
// on the first use of the context (i.e. for parsing a source in an
// invocation) but API access like this forces it.
// Returns a NULL context if it has already been stolen or if activation fails.
MLIR_CAPI_EXPORTED MlirContext
ireeCompilerSessionBorrowContext(iree_compiler_session_t *session);

// Gets the MlirContext that the session manages, releasing it for external
// management of its lifetime. If the context has already been released,
// then {nullptr} is returned. Upon return, it is up to the caller to destroy
// the context and ensure that its lifetime extends at least as long as the
// session remains in use.
// This implicitly "activates" the session: make sure that any configuration
// (flags, etc) has been done prior. Activation is lazy and is usually done
// on the first use of the context (i.e. for parsing a source in an
// invocation) but API access like this forces it.
// Returns a NULL context if it has already been stolen or if activation fails.
MLIR_CAPI_EXPORTED MlirContext
ireeCompilerSessionStealContext(iree_compiler_session_t *session);

// Same as |ireeCompilerInvocationStealModule| but ownership of the module
// remains with the caller, who is responsible for ensuring that it is not
// destroyed before the invocation is destroyed.
MLIR_CAPI_EXPORTED bool
ireeCompilerInvocationImportBorrowModule(iree_compiler_invocation_t *inv,
                                         MlirOperation moduleOp);

// Imports an externally built MlirModule into the invocation as an alternative
// to parsing with |ireeCompilerInvocationParseSource|.
// Ownership of the moduleOp is transferred to the invocation, regardless of
// whether the call succeeds or fails.
// On failure, returns false and issues diagnostics.
MLIR_CAPI_EXPORTED bool
ireeCompilerInvocationImportStealModule(iree_compiler_invocation_t *inv,
                                        MlirOperation moduleOp);

// Exports the owned module from the invocation, transferring ownership to the
// caller.
MLIR_CAPI_EXPORTED
MlirOperation
ireeCompilerInvocationExportStealModule(iree_compiler_invocation_t *inv);

#ifdef __cplusplus
}
#endif

#endif // IREE_COMPILER_API_MLIR_INTEROP_H
