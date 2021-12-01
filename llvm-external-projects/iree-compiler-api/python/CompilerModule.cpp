// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-compiler-c/Compiler.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

namespace {

struct PyCompilerOptions {
  PyCompilerOptions() : options(ireeCompilerOptionsCreate()) {}
  PyCompilerOptions(const PyCompilerOptions &) = delete;
  void operator=(const PyCompilerOptions &) = delete;
  PyCompilerOptions(PyCompilerOptions &&other) : options(other.options) {
    other.options = {nullptr};
  }
  ~PyCompilerOptions() {
    if (options.ptr) ireeCompilerOptionsDestroy(options);
  }
  IreeCompilerOptions options;
};

/// Accumulates int a python file-like object, either writing text (default)
/// or binary.
class PyFileAccumulator {
 public:
  PyFileAccumulator(pybind11::object fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire();
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        pybind11::bytes pyBytes(part.data, part.length);
        accum->pyWriteFunction(pyBytes);
      } else {
        pybind11::str pyStr(part.data,
                            part.length);  // Decodes as UTF-8 by default.
        accum->pyWriteFunction(pyStr);
      }
    };
  }

 private:
  pybind11::object pyWriteFunction;
  bool binary;
};

}  // namespace

static const char BUILD_MHLO_IMPORT_PASS_PIPELINE_DOCSTRING[] =
    R"(Populates MHLO import passes on a PassManager.

This enables standalone access to the import pipeline that can be run as part
of main compilation with the `set_input_dialect_mhlo()` option. It is provided
seprately to facilitate integration with frontend workflows.

This pipeline requires IREE-compatible MHLO input: MHLO control flow must have
been legalized to SCF or CFG and tuples must not exist. See the
`build_xla_cleanup_pass_pipeline` for assistance if interoping with such IR.
)";

static const char BUILD_TOSA_IMPORT_PASS_PIPELINE_DOCSTRING[] =
    R"(Populates TOSA import passes on a PassManager.

This enables standalone access to the import pipeline that can be run as part
of main compilation with the `set_input_dialect_tosa()` option. It is provided
seprately to facilitate integration with frontend workflows.
)";

static const char BUILD_IREE_VM_PASS_PIPELINE_DOCSTRING[] =
    R"(Populates VM compilation pass on a PassManager.

This is the primary interface to IREE's backend compiler, providing compilation
from a supported set of input dialects to the `vm` dialect, representing
IREE's lowest level representation.
)";

static const char BUILD_XLA_CLEANUP_PASS_PIPELINE_DOCSTRING[] =
    R"(Populates passes to cleanup XLA-imported MHLO to comply with IREE.

Combining this pipeline with `build_mhlo_import_pass_pipeline()` provides
standalone access to the import pipeline that can be run as part of main
compilation with the `set_input_dialect_xla()` option. It is provided
separately to facilitate integration with frontend workflows.
)";

static const char TRANSLATE_MODULE_TO_VM_BYTECODE_DOCSTRING[] =
    R"(Given a `vm.module` translate it to VM bytecode.

The provided `file` argument must be a valid `IO` object, capable of having
binary data written to it.
)";

PYBIND11_MODULE(_ireeCompilerDriver, m) {
  m.doc() = "iree-compiler driver api";
  ireeCompilerRegisterTargetBackends();

  py::class_<PyCompilerOptions>(m, "CompilerOptions",
                                "Options for the IREE backend compiler.")
      .def(py::init<>())
      .def(
          "set_input_dialect_mhlo",
          [](PyCompilerOptions &self) {
            ireeCompilerOptionsSetInputDialectMHLO(self.options);
          },
          "Sets the input type to the 'mhlo' dialect")
      .def(
          "set_input_dialect_tosa",
          [](PyCompilerOptions &self) {
            ireeCompilerOptionsSetInputDialectTOSA(self.options);
          },
          "Sets the input type to the 'tosa' dialect")
      .def(
          "set_input_dialect_xla",
          [](PyCompilerOptions &self) {
            ireeCompilerOptionsSetInputDialectTOSA(self.options);
          },
          "Sets the input type to the 'mhlo' dialect with XLA compatibility "
          "cleanups")
      .def(
          "add_target_backend",
          [](PyCompilerOptions &self, const std::string &targetBackend) {
            ireeCompilerOptionsAddTargetBackend(self.options,
                                                targetBackend.c_str());
          },
          py::arg("target_backend"),
          "Adds a target backend (i.e. 'cpu', 'vulkan-spirv', etc)");
  m.def(
      "build_mhlo_import_pass_pipeline",
      [](MlirPassManager passManager) {
        MlirOpPassManager opPassManager =
            mlirPassManagerGetAsOpPassManager(passManager);
        ireeCompilerBuildMHLOImportPassPipeline(opPassManager);
      },
      py::arg("pass_manager"), BUILD_MHLO_IMPORT_PASS_PIPELINE_DOCSTRING);
  m.def(
      "build_tosa_import_pass_pipeline",
      [](MlirPassManager passManager) {
        MlirOpPassManager opPassManager =
            mlirPassManagerGetAsOpPassManager(passManager);
        ireeCompilerBuildTOSAImportPassPipeline(opPassManager);
      },
      py::arg("pass_manager"), BUILD_TOSA_IMPORT_PASS_PIPELINE_DOCSTRING);
  m.def(
      "build_xla_cleanup_pass_pipeline",
      [](MlirPassManager passManager) {
        MlirOpPassManager opPassManager =
            mlirPassManagerGetAsOpPassManager(passManager);
        ireeCompilerBuildXLACleanupPassPipeline(opPassManager);
      },
      py::arg("pass_manager"), BUILD_XLA_CLEANUP_PASS_PIPELINE_DOCSTRING);
  m.def(
      "build_iree_vm_pass_pipeline",
      [](PyCompilerOptions &options, MlirPassManager passManager) {
        MlirOpPassManager opPassManager =
            mlirPassManagerGetAsOpPassManager(passManager);
        ireeCompilerBuildIREEVMPassPipeline(options.options, opPassManager);
      },
      py::arg("options"), py::arg("pass_manager"),
      BUILD_IREE_VM_PASS_PIPELINE_DOCSTRING);

  m.def(
      "translate_module_to_vm_bytecode",
      [](PyCompilerOptions &options, MlirModule module, py::object file) {
        PyFileAccumulator accum(file, /*binary=*/true);
        MlirOperation operation = mlirModuleGetOperation(module);
        auto result = ireeCompilerTranslateModuletoVMBytecode(
            options.options, operation, accum.getCallback(),
            accum.getUserData());
        if (mlirLogicalResultIsFailure(result)) {
          throw std::runtime_error("failure translating module to bytecode");
        }
      },
      py::arg("options"), py::arg("module"), py::arg("file"),
      TRANSLATE_MODULE_TO_VM_BYTECODE_DOCSTRING);
}
