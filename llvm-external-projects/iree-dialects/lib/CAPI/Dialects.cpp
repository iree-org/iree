// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Dialects.h"

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMDialect.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

//===----------------------------------------------------------------------===//
// IREEDialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    IREEInput, iree_input, mlir::iree_compiler::IREE::Input::IREEInputDialect)

//===--------------------------------------------------------------------===//
// IREELinalgExt
//===--------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    IREELinalgExt, iree_linalg_ext,
    mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect)

//===--------------------------------------------------------------------===//
// IREELinalgTransform
//===--------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    IREELinalgTransform, iree_linalg_transform,
    mlir::linalg::transform::LinalgTransformDialect)

//===----------------------------------------------------------------------===//
// IREEPyDMDialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(IREEPyDM, iree_pydm,
                                      PYDM::IREEPyDMDialect)

DEFINE_C_API_PTR_METHODS(IREEPyDMSourceBundle, PYDM::SourceBundle)
DEFINE_C_API_PTR_METHODS(IREEPyDMLoweringOptions, PYDM::LowerToIREEOptions)

void mlirIREEPyDMRegisterPasses() { PYDM::registerPasses(); }

bool mlirTypeIsAIREEPyDMPrimitiveType(MlirType type) {
  return unwrap(type).isa<PYDM::PrimitiveType>();
}

#define IREEPYDM_DEFINE_NULLARY_TYPE(Name)                \
  bool mlirTypeIsAIREEPyDM##Name(MlirType type) {         \
    return unwrap(type).isa<PYDM::Name##Type>();          \
  }                                                       \
  MlirType mlirIREEPyDM##Name##TypeGet(MlirContext ctx) { \
    return wrap(PYDM::Name##Type::get(unwrap(ctx)));      \
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
  return wrap(PYDM::IntegerType::get(unwrap(ctx), bitWidth, isSigned));
}

MlirType mlirIREEPyDMRealTypeGetExplicit(MlirType fpType) {
  auto fpTypeCpp = unwrap(fpType).cast<FloatType>();
  return wrap(PYDM::RealType::get(fpTypeCpp.getContext(), fpTypeCpp));
}

// ObjectType.
bool mlirTypeIsAIREEPyDMObject(MlirType type) {
  return unwrap(type).isa<PYDM::ObjectType>();
}

MlirType mlirIREEPyDMObjectTypeGet(MlirContext ctx, MlirType primitive) {
  if (!primitive.ptr) {
    return wrap(PYDM::ObjectType::get(unwrap(ctx), nullptr));
  }

  auto cppType = unwrap(primitive).cast<PYDM::PrimitiveType>();
  return wrap(PYDM::ObjectType::get(unwrap(ctx), cppType));
}

MLIR_CAPI_EXPORTED void mlirIREEPyDMBuildPostImportPassPipeline(
    MlirOpPassManager passManager) {
  auto *passManagerCpp = unwrap(passManager);
  PYDM::buildPostImportPassPipeline(*passManagerCpp);
}

// LowerToIREE Pass Pipeline.
void mlirIREEPyDMBuildLowerToIREEPassPipeline(MlirOpPassManager passManager,
                                              IREEPyDMLoweringOptions options) {
  auto *passManagerCpp = unwrap(passManager);
  PYDM::buildLowerToIREEPassPipeline(*passManagerCpp, *unwrap(options));
}

// SourceBundle
IREEPyDMSourceBundle ireePyDMSourceBundleCreateAsm(MlirStringRef asmString) {
  auto bundle = std::make_unique<PYDM::SourceBundle>();
  bundle->asmBlob = std::make_shared<std::string>(unwrap(asmString));
  return wrap(bundle.release());
}

IREEPyDMSourceBundle ireePyDMSourceBundleCreateFile(MlirStringRef filePath) {
  auto bundle = std::make_unique<PYDM::SourceBundle>();
  bundle->asmFilePath = std::string(unwrap(filePath));
  return wrap(bundle.release());
}

void ireePyDMSourceBundleDestroy(IREEPyDMSourceBundle bundle) {
  delete unwrap(bundle);
}

// LoweringOptions
IREEPyDMLoweringOptions ireePyDMLoweringOptionsCreate() {
  return wrap(new PYDM::LowerToIREEOptions);
}

void ireePyDMLoweringOptionsLinkRtl(IREEPyDMLoweringOptions options,
                                    IREEPyDMSourceBundle source) {
  unwrap(options)->linkRtlSource = *unwrap(source);
}

void ireePyDMLoweringOptionsDestroy(IREEPyDMLoweringOptions options) {
  delete unwrap(options);
}
