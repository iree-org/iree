// Copyright 2019 Google LLC
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

#include "bindings/python/pyiree/tf_interop/register_tensorflow.h"

#include <string>
#include <vector>

#include "bindings/python/pyiree/compiler.h"
#include "bindings/python/pyiree/status_utils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "third_party/llvm/llvm/include/llvm/Support/raw_ostream.h"

using namespace mlir;  // NOLINT

using tensorflow::ConvertSavedModelToMlir;
using tensorflow::RunOptions;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;

namespace iree {
namespace python {

namespace {

CompilerModuleBundle LoadSavedModel(
    std::shared_ptr<CompilerContextBundle> context_bundle,
    const std::string& saved_model_dir,
    const std::vector<std::string>& exported_names) {
  SessionOptions session_options;
  RunOptions run_options;
  SavedModelBundle bundle;
  std::unordered_set<std::string> tags{"serve"};
  auto load_status = LoadSavedModel(
      session_options, run_options,
      std::string(saved_model_dir.data(), saved_model_dir.length()), tags,
      &bundle);
  if (!load_status.ok()) {
    std::stringstream msg;
    msg << "Failed to load saved model '" << saved_model_dir
        << "': " << load_status;
    throw RaisePyError(PyExc_RuntimeError, msg.str().c_str());
  }

  // TODO(laurenzo): Fix the upstream ConvertSavedModelToMlir() to take a const
  // span of external names.
  std::vector<std::string> mutable_exported_names = exported_names;
  auto module_or =
      ConvertSavedModelToMlir(bundle, context_bundle->mlir_context(),
                              absl::MakeSpan(mutable_exported_names));
  if (!module_or.status().ok()) {
    std::stringstream msg;
    msg << "Failed to load saved model '" << saved_model_dir
        << "': " << load_status;
    throw RaisePyError(PyExc_RuntimeError, msg.str().c_str());
  }
  return CompilerModuleBundle(context_bundle,
                              module_or.ConsumeValueOrDie().release());
}

}  // namespace

void SetupTensorFlowBindings(pybind11::module m) {
  m.def("load_saved_model", &LoadSavedModel, py::arg("compiler_context"),
        py::arg("saved_model_dir"),
        py::arg("exported_names") = std::vector<std::string>());
}

}  // namespace python
}  // namespace iree
