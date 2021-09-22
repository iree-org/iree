// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Dialects.h"
#include "iree-dialects-c/Utils.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_ireeDialects, m) {
  m.doc() = "iree-dialects main python extension";

  auto irModule = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"));
  auto typeClass = irModule.attr("Type");

  //===--------------------------------------------------------------------===//
  // Utils
  //===--------------------------------------------------------------------===//

  m.def(
      "lookup_nearest_symbol_from",
      [](MlirOperation fromOp, MlirAttribute symbol) {
        if (!mlirAttributeIsASymbolRef(symbol)) {
          throw std::invalid_argument("expected a SymbolRefAttr");
        }
        return ireeLookupNearestSymbolFrom(fromOp, symbol);
      },
      py::arg("fromOp"), py::arg("symbol"));

  // TODO: Upstream this into the main Python bindings.
  m.def(
      "emit_error",
      [](MlirLocation loc, std::string message) {
        mlirEmitError(loc, message.c_str());
      },
      py::arg("loc"), py::arg("message"));

  //===--------------------------------------------------------------------===//
  // IREEDialect
  //===--------------------------------------------------------------------===//
  auto iree_m = m.def_submodule("iree");
  iree_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__iree__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // IREEPyDMDialect
  //===--------------------------------------------------------------------===//
  auto iree_pydm_m = m.def_submodule("iree_pydm");

  iree_pydm_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__iree_pydm__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  iree_pydm_m.def(
      "build_lower_to_iree_pass_pipeline",
      [](MlirPassManager passManager) {
        MlirOpPassManager opPassManager =
            mlirPassManagerGetAsOpPassManager(passManager);
        mlirIREEPyDMBuildLowerToIREEPassPipeline(opPassManager);
      },
      py::arg("pass_manager"));

#define DEFINE_IREEPYDM_NULLARY_TYPE(Name)                                 \
  mlir_type_subclass(iree_pydm_m, #Name "Type", mlirTypeIsAIREEPyDM##Name, \
                     typeClass)                                            \
      .def_classmethod(                                                    \
          "get",                                                           \
          [](py::object cls, MlirContext context) {                        \
            return cls(mlirIREEPyDM##Name##TypeGet(context));              \
          },                                                               \
          py::arg("cls"), py::arg("context") = py::none());

  DEFINE_IREEPYDM_NULLARY_TYPE(Bool)
  DEFINE_IREEPYDM_NULLARY_TYPE(Bytes)
  DEFINE_IREEPYDM_NULLARY_TYPE(ExceptionResult)
  DEFINE_IREEPYDM_NULLARY_TYPE(FreeVarRef)
  DEFINE_IREEPYDM_NULLARY_TYPE(Integer)
  DEFINE_IREEPYDM_NULLARY_TYPE(List)
  DEFINE_IREEPYDM_NULLARY_TYPE(None)
  DEFINE_IREEPYDM_NULLARY_TYPE(Real)
  DEFINE_IREEPYDM_NULLARY_TYPE(Str)
  DEFINE_IREEPYDM_NULLARY_TYPE(Tuple)
  DEFINE_IREEPYDM_NULLARY_TYPE(Type)

  mlir_type_subclass(iree_pydm_m, "ObjectType", mlirTypeIsAIREEPyDMObject,
                     typeClass)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext context) {
            return cls(mlirIREEPyDMObjectTypeGet(context, {nullptr}));
          },
          py::arg("cls"), py::arg("context") = py::none())
      .def_classmethod(
          "get_typed",
          [](py::object cls, MlirType type) {
            if (!mlirTypeIsAIREEPyDMPrimitiveType(type)) {
              throw std::invalid_argument(
                  "expected a primitive type when constructing object");
            }
            MlirContext context = mlirTypeGetContext(type);
            return cls(mlirIREEPyDMObjectTypeGet(context, type));
          },
          py::arg("cls"), py::arg("type"));
}
