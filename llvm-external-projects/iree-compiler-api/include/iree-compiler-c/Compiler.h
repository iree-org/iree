// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_COMPILER_API_COMPILER_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_COMPILER_API_COMPILER_H

#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage) \
  struct name {                            \
    storage *ptr;                          \
  };                                       \
  typedef struct name name

DEFINE_C_API_STRUCT(IreeCompilerOptions, void);
#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void ireeCompilerRegisterTargetBackends();

//===----------------------------------------------------------------------===//
// Compiler options.
//===----------------------------------------------------------------------===//

// Creates and destroys a compiler options structure.
MLIR_CAPI_EXPORTED IreeCompilerOptions ireeCompilerOptionsCreate();
MLIR_CAPI_EXPORTED void ireeCompilerOptionsDestroy(IreeCompilerOptions options);

MLIR_CAPI_EXPORTED void ireeCompilerOptionsSetInputDialectMHLO(
    IreeCompilerOptions options);
MLIR_CAPI_EXPORTED void ireeCompilerOptionsSetInputDialectTOSA(
    IreeCompilerOptions options);
MLIR_CAPI_EXPORTED void ireeCompilerOptionsSetInputDialectXLA(
    IreeCompilerOptions options);
MLIR_CAPI_EXPORTED void ireeCompilerOptionsAddTargetBackend(
    IreeCompilerOptions options, const char *targetBackend);

//===----------------------------------------------------------------------===//
// Compiler stages.
//===----------------------------------------------------------------------===//

// Builds a pass pipeline to cleanup MHLO dialect input derived from XLA.
// A pass pipeline of this plus ireeCompilerBuildMHLOImportPassPipeline is
// equivalent to ireeCompilerOptionsSetInputDialectXLA as a one-shot.
MLIR_CAPI_EXPORTED void ireeCompilerBuildXLACleanupPassPipeline(
    MlirOpPassManager passManager);

// Builds a pass pipeline to lower IREE-compatible MHLO functions and ops to
// be a legal input to IREE. This performs the standalone work that
// ireeCompilerOptionsSetInputDialectMHLO will do as a one-shot. Notably, this
// requires that XLA control flow has been legalized to SCF or CFG and that
// no tuples are in the input program.
MLIR_CAPI_EXPORTED void ireeCompilerBuildMHLOImportPassPipeline(
    MlirOpPassManager passManager);

// Builds a pass pipeline to lower IREE-compatible TOSA function and ops to
// be a legal input to IREE. This performs that standalone work that
// ireeCompilerOptionsSetInputDialectTOSA will do as a one-shot.
MLIR_CAPI_EXPORTED void ireeCompilerBuildTOSAImportPassPipeline(
    MlirOpPassManager passManager);

// Builds a pass manager for transforming from an input module op to the IREE VM
// dialect. This represents the primary compilation stage with serialization to
// specific formats following.
MLIR_CAPI_EXPORTED void ireeCompilerBuildIREEVMPassPipeline(
    IreeCompilerOptions options, MlirOpPassManager passManager);

// Translates a module op derived from the ireeCompilerBuildIREEVMPassPipeline
// to serialized bytecode. The module op may either be an outer builtin ModuleOp
// wrapping a VM::ModuleOp or a VM::ModuleOp.
MLIR_CAPI_EXPORTED MlirLogicalResult ireeCompilerTranslateModuletoVMBytecode(
    IreeCompilerOptions options, MlirOperation moduleOp,
    MlirStringCallback dataCallback, void *dataUserObject);

#ifdef __cplusplus
}
#endif

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_COMPILER_API_COMPILER_H
