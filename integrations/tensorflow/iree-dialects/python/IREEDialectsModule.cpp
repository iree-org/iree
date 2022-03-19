// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Dialects.h"
#include "iree-dialects-c/Utils.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

namespace {

struct PyIREEPyDMSourceBundle {
  PyIREEPyDMSourceBundle(IREEPyDMSourceBundle wrapped) : wrapped(wrapped) {}
  PyIREEPyDMSourceBundle(PyIREEPyDMSourceBundle &&other)
      : wrapped(other.wrapped) {
    other.wrapped.ptr = nullptr;
  }
  PyIREEPyDMSourceBundle(const PyIREEPyDMSourceBundle &) = delete;
  ~PyIREEPyDMSourceBundle() {
    if (wrapped.ptr) ireePyDMSourceBundleDestroy(wrapped);
  }
  IREEPyDMSourceBundle wrapped;
};

struct PyIREEPyDMLoweringOptions {
  PyIREEPyDMLoweringOptions() : wrapped(ireePyDMLoweringOptionsCreate()) {}
  PyIREEPyDMLoweringOptions(PyIREEPyDMLoweringOptions &&other)
      : wrapped(other.wrapped) {
    other.wrapped.ptr = nullptr;
  }
  PyIREEPyDMLoweringOptions(const PyIREEPyDMLoweringOptions &) = delete;
  ~PyIREEPyDMLoweringOptions() {
    if (wrapped.ptr) ireePyDMLoweringOptionsDestroy(wrapped);
  }
  IREEPyDMLoweringOptions wrapped;
};

}  // namespace

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
  auto iree_m = m.def_submodule("iree_input");
  iree_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__iree_input__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // IREELinalgExt
  //===--------------------------------------------------------------------===//
  auto iree_linalg_ext_m = m.def_submodule("iree_linalg_ext");
  iree_linalg_ext_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__iree_linalg_ext__();
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
  mlirIREEPyDMRegisterPasses();

  py::class_<PyIREEPyDMSourceBundle>(
      iree_pydm_m, "SourceBundle", py::module_local(),
      "Contains raw assembly source or a reference to a file")
      .def_static(
          "from_asm",
          [](std::string asmBlob) {
            return PyIREEPyDMSourceBundle(ireePyDMSourceBundleCreateAsm(
                {asmBlob.data(), asmBlob.size()}));
          },
          py::arg("asm_blob"),
          "Creates a SourceBundle from an ASM blob (string or bytes)")
      .def_static(
          "from_file",
          [](std::string asmFile) {
            return PyIREEPyDMSourceBundle(ireePyDMSourceBundleCreateFile(
                {asmFile.data(), asmFile.size()}));
          },
          py::arg("asm_file"),
          "Creates a SourceBundle from a file containing ASM");
  py::class_<PyIREEPyDMLoweringOptions>(iree_pydm_m, "LoweringOptions",
                                        py::module_local(),
                                        "Lowering options to compile to IREE")
      .def(py::init<>())
      .def(
          "link_rtl",
          [](PyIREEPyDMLoweringOptions &self,
             PyIREEPyDMSourceBundle &sourceBundle) {
            ireePyDMLoweringOptionsLinkRtl(self.wrapped, sourceBundle.wrapped);
          },
          "Enables linking against a runtime-library module");

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
      [](MlirPassManager passManager, PyIREEPyDMLoweringOptions &options) {
        MlirOpPassManager opPassManager =
            mlirPassManagerGetAsOpPassManager(passManager);
        mlirIREEPyDMBuildLowerToIREEPassPipeline(opPassManager,
                                                 options.wrapped);
      },
      py::arg("pass_manager"), py::arg("link_rtl_asm") = py::none());

  iree_pydm_m.def(
      "build_post_import_pass_pipeline",
      [](MlirPassManager passManager) {
        MlirOpPassManager opPassManager =
            mlirPassManagerGetAsOpPassManager(passManager);
        mlirIREEPyDMBuildPostImportPassPipeline(opPassManager);
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
  DEFINE_IREEPYDM_NULLARY_TYPE(List)
  DEFINE_IREEPYDM_NULLARY_TYPE(None)
  DEFINE_IREEPYDM_NULLARY_TYPE(Str)
  DEFINE_IREEPYDM_NULLARY_TYPE(Tuple)
  DEFINE_IREEPYDM_NULLARY_TYPE(Type)

  // IntegerType.
  mlir_type_subclass(iree_pydm_m, "IntegerType", mlirTypeIsAIREEPyDMInteger,
                     typeClass)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext context) {
            return cls(mlirIREEPyDMIntegerTypeGet(context));
          },
          py::arg("cls"), py::arg("context") = py::none())
      .def_classmethod(
          "get_explicit",
          [](py::object cls, int bitWidth, bool isSigned, MlirContext context) {
            return cls(mlirIREEPyDMIntegerTypeGetExplicit(context, bitWidth,
                                                          isSigned));
          },
          py::arg("cls"), py::arg("bit_width"), py::arg("is_signed") = true,
          py::arg("context") = py::none());

  // RealType.
  mlir_type_subclass(iree_pydm_m, "RealType", mlirTypeIsAIREEPyDMReal,
                     typeClass)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext context) {
            return cls(mlirIREEPyDMRealTypeGet(context));
          },
          py::arg("cls"), py::arg("context") = py::none())
      .def_classmethod(
          "get_explicit",
          [](py::object cls, MlirType fpType) {
            // TODO: Add a C-API for generically checking for FloatType.
            if (!mlirTypeIsAF32(fpType) && !mlirTypeIsAF64(fpType) &&
                !mlirTypeIsAF16(fpType) && !mlirTypeIsABF16(fpType)) {
              throw std::invalid_argument("expected a floating point type");
            }
            return cls(mlirIREEPyDMRealTypeGetExplicit(fpType));
          },
          py::arg("cls"), py::arg("fp_type"));

  // ObjectType.
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
