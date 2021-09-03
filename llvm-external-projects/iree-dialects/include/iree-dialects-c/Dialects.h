// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_C_DIALECTS_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_C_DIALECTS_H

#include "mlir-c/IR.h"
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

MLIR_CAPI_EXPORTED bool mlirTypeIsAIREEPyDMPrimitiveType(MlirType type);

#define IREEPYDM_DECLARE_NULLARY_TYPE(Name)                         \
  MLIR_CAPI_EXPORTED bool mlirTypeIsAIREEPyDM##Name(MlirType type); \
  MLIR_CAPI_EXPORTED MlirType mlirIREEPyDM##Name##TypeGet(MlirContext ctx);

IREEPYDM_DECLARE_NULLARY_TYPE(Bool)
IREEPYDM_DECLARE_NULLARY_TYPE(Bytes)
IREEPYDM_DECLARE_NULLARY_TYPE(Integer)
IREEPYDM_DECLARE_NULLARY_TYPE(ExceptionResult)
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

#ifdef __cplusplus
}
#endif

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_C_DIALECTS_H
