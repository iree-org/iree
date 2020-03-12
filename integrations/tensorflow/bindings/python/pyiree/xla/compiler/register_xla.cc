// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "integrations/tensorflow/bindings/python/pyiree/xla/compiler/register_xla.h"

#include <string>
#include <vector>

#include "bindings/python/pyiree/common/status_utils.h"
#include "bindings/python/pyiree/compiler/compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_module_importer.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace iree {
namespace python {

namespace {

CompilerModuleBundle LoadXlaModuleProto(
    std::shared_ptr<CompilerContextBundle> context_bundle,
    xla::XlaComputation& computation,
    const std::vector<std::string>& exported_names) {
  auto context = context_bundle->mlir_context();
  mlir::Builder builder(context);

  // TODO(suderman): Figure out how to transport a location detail in.
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

  xla::HloModuleImporter importer(module.get());
  auto result = importer.Import(computation.proto());
  if (!result.ok()) {
    std::stringstream msg;
    msg << "Failed to convert HLO Module Proto to MLIR: " << result;
    throw RaisePyError(PyExc_RuntimeError, msg.str().c_str());
  }

  for (auto func : module->getOps<mlir::FuncOp>()) {
    func.setAttr("iree.module.export", mlir::UnitAttr::get(func.getContext()));
  }

  return CompilerModuleBundle(context_bundle, module.release());
}

}  // namespace

void SetupXlaBindings(pybind11::module m) {
  m.def("load_xla_module_proto", &LoadXlaModuleProto,
        py::arg("compiler_context"), py::arg("computation"),
        py::arg("exported_names") = std::vector<std::string>());
}

}  // namespace python
}  // namespace iree
