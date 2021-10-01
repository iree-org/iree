// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_C_DIALECTS_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// IREEDialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(IREE, iree);

//===----------------------------------------------------------------------===//
// IREEPyDMDialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(IREEPyDM, iree_pydm);

#define DEFINE_C_API_STRUCT(name, storage) \
  struct name {                            \
    storage *ptr;                          \
  };                                       \
  typedef struct name name

DEFINE_C_API_STRUCT(IREEPyDMSourceBundle, void);
DEFINE_C_API_STRUCT(IREEPyDMLoweringOptions, void);
#undef DEFINE_C_API_STRUCT

/// Creates a PyDM source bundle from an ASM string.
MLIR_CAPI_EXPORTED IREEPyDMSourceBundle
ireePyDMSourceBundleCreateAsm(MlirStringRef asmString);

/// Creates a PyDM source bundle from a file path.
MLIR_CAPI_EXPORTED IREEPyDMSourceBundle
ireePyDMSourceBundleCreateFile(MlirStringRef filePath);

/// Destroys a created source bundle.
MLIR_CAPI_EXPORTED void ireePyDMSourceBundleDestroy(
    IREEPyDMSourceBundle bundle);

MLIR_CAPI_EXPORTED bool mlirTypeIsAIREEPyDMPrimitiveType(MlirType type);

#define IREEPYDM_DECLARE_NULLARY_TYPE(Name)                         \
  MLIR_CAPI_EXPORTED bool mlirTypeIsAIREEPyDM##Name(MlirType type); \
  MLIR_CAPI_EXPORTED MlirType mlirIREEPyDM##Name##TypeGet(MlirContext ctx);

IREEPYDM_DECLARE_NULLARY_TYPE(Bool)
IREEPYDM_DECLARE_NULLARY_TYPE(Bytes)
IREEPYDM_DECLARE_NULLARY_TYPE(Integer)
IREEPYDM_DECLARE_NULLARY_TYPE(ExceptionResult)
IREEPYDM_DECLARE_NULLARY_TYPE(FreeVarRef)
IREEPYDM_DECLARE_NULLARY_TYPE(List)
IREEPYDM_DECLARE_NULLARY_TYPE(None)
IREEPYDM_DECLARE_NULLARY_TYPE(Real)
IREEPYDM_DECLARE_NULLARY_TYPE(Str)
IREEPYDM_DECLARE_NULLARY_TYPE(Tuple)
IREEPYDM_DECLARE_NULLARY_TYPE(Type)

#undef IREEPYDM_DECLARE_NULLARY_TYPE

MLIR_CAPI_EXPORTED bool mlirTypeIsAIREEPyDMObject(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirIREEPyDMObjectTypeGet(MlirContext context,
                                                      MlirType primitive);

/// Creates a lowering options struct.
MLIR_CAPI_EXPORTED IREEPyDMLoweringOptions ireePyDMLoweringOptionsCreate();

/// Sets the RTL link source bundle to the lowering options.
MLIR_CAPI_EXPORTED void ireePyDMLoweringOptionsLinkRtl(
    IREEPyDMLoweringOptions options, IREEPyDMSourceBundle source);

/// Destroys a created lowering options struct.
MLIR_CAPI_EXPORTED void ireePyDMLoweringOptionsDestroy(
    IREEPyDMLoweringOptions options);

/// Builds a pass pipeline which lowers the iree_pydm dialect to IREE.
MLIR_CAPI_EXPORTED void mlirIREEPyDMBuildLowerToIREEPassPipeline(
    MlirOpPassManager passManager, IREEPyDMLoweringOptions options);

#ifdef __cplusplus
}
#endif

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_C_DIALECTS_H
