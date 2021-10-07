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
#include "mlir/Support/LLVM.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// IREEDialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(IREE, iree, mlir::iree::IREEDialect)

//===----------------------------------------------------------------------===//
// IREEPyDMDialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(IREEPyDM, iree_pydm,
                                      mlir::iree_pydm::IREEPyDMDialect)

DEFINE_C_API_PTR_METHODS(IREEPyDMSourceBundle, mlir::iree_pydm::SourceBundle)
DEFINE_C_API_PTR_METHODS(IREEPyDMLoweringOptions,
                         mlir::iree_pydm::LowerToIREEOptions)

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

// Non-nullary Type constructors from the above.
MlirType mlirIREEPyDMIntegerTypeGetExplicit(MlirContext ctx, int bitWidth,
                                            bool isSigned) {
  return wrap(
      mlir::iree_pydm::IntegerType::get(unwrap(ctx), bitWidth, isSigned));
}

// ObjectType.
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

// LowerToIREE Pass Pipeline.
void mlirIREEPyDMBuildLowerToIREEPassPipeline(MlirOpPassManager passManager,
                                              IREEPyDMLoweringOptions options) {
  auto *passManagerCpp = unwrap(passManager);
  mlir::iree_pydm::buildLowerToIREEPassPipeline(*passManagerCpp,
                                                *unwrap(options));
}

// SourceBundle
IREEPyDMSourceBundle ireePyDMSourceBundleCreateAsm(MlirStringRef asmString) {
  auto bundle = std::make_unique<mlir::iree_pydm::SourceBundle>();
  bundle->asmBlob = std::make_shared<std::string>(unwrap(asmString));
  return wrap(bundle.release());
}

IREEPyDMSourceBundle ireePyDMSourceBundleCreateFile(MlirStringRef filePath) {
  auto bundle = std::make_unique<mlir::iree_pydm::SourceBundle>();
  bundle->asmFilePath = std::string(unwrap(filePath));
  return wrap(bundle.release());
}

void ireePyDMSourceBundleDestroy(IREEPyDMSourceBundle bundle) {
  delete unwrap(bundle);
}

// LoweringOptions
IREEPyDMLoweringOptions ireePyDMLoweringOptionsCreate() {
  return wrap(new mlir::iree_pydm::LowerToIREEOptions);
}

void ireePyDMLoweringOptionsLinkRtl(IREEPyDMLoweringOptions options,
                                    IREEPyDMSourceBundle source) {
  unwrap(options)->linkRtlSource = *unwrap(source);
}

void ireePyDMLoweringOptionsDestroy(IREEPyDMLoweringOptions options) {
  delete unwrap(options);
}
