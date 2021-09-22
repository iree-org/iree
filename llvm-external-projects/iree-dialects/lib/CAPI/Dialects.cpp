// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Dialects.h"

#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Dialect.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/CAPI/Wrap.h"

//===----------------------------------------------------------------------===//
// IREEDialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(IREE, iree, mlir::iree::IREEDialect)

//===----------------------------------------------------------------------===//
// IREEPyDMDialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(IREEPyDM, iree_pydm,
                                      mlir::iree_pydm::IREEPyDMDialect)

bool mlirTypeIsAIREEPyDMPrimitiveType(MlirType type) {
  return unwrap(type).isa<mlir::iree_pydm::PrimitiveType>();
}

#define IREEPYDM_DEFINE_NULLARY_TYPE(Name)                      \
  bool mlirTypeIsAIREEPyDM##Name(MlirType type) {               \
    return unwrap(type).isa<mlir::iree_pydm::Name##Type>();     \
  }                                                             \
  MlirType mlirIREEPyDM##Name##TypeGet(MlirContext ctx) {       \
    return wrap(mlir::iree_pydm::Name##Type::get(unwrap(ctx))); \
  }

IREEPYDM_DEFINE_NULLARY_TYPE(Bool)
IREEPYDM_DEFINE_NULLARY_TYPE(Bytes)
IREEPYDM_DEFINE_NULLARY_TYPE(Integer)
IREEPYDM_DEFINE_NULLARY_TYPE(ExceptionResult)
IREEPYDM_DEFINE_NULLARY_TYPE(FreeVarRef)
IREEPYDM_DEFINE_NULLARY_TYPE(List)
IREEPYDM_DEFINE_NULLARY_TYPE(None)
IREEPYDM_DEFINE_NULLARY_TYPE(Real)
IREEPYDM_DEFINE_NULLARY_TYPE(Str)
IREEPYDM_DEFINE_NULLARY_TYPE(Tuple)
IREEPYDM_DEFINE_NULLARY_TYPE(Type)

bool mlirTypeIsAIREEPyDMObject(MlirType type) {
  return unwrap(type).isa<mlir::iree_pydm::ObjectType>();
}

MlirType mlirIREEPyDMObjectTypeGet(MlirContext ctx, MlirType primitive) {
  if (!primitive.ptr) {
    return wrap(mlir::iree_pydm::ObjectType::get(unwrap(ctx), nullptr));
  }

  auto cppType = unwrap(primitive).cast<mlir::iree_pydm::PrimitiveType>();
  return wrap(mlir::iree_pydm::ObjectType::get(unwrap(ctx), cppType));
}

void mlirIREEPyDMBuildLowerToIREEPassPipeline(MlirOpPassManager passManager) {
  auto *passManagerCpp = unwrap(passManager);
  // TODO: Should be a pass pipeline, not loose passes in the C impl.
  passManagerCpp->addPass(mlir::iree_pydm::createConvertIREEPyDMToIREEPass());
}
