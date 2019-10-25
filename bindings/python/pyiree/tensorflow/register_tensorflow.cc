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

#include "bindings/python/pyiree/tensorflow/register_tensorflow.h"

#include <string>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

using namespace mlir;  // NOLINT

namespace iree {
namespace python {

namespace {

std::string ImportSavedModelToMlirAsm(const std::string& saved_model_dir,
                                      std::vector<std::string> exported_names,
                                      std::vector<std::string> tags) {
  std::unordered_set<std::string> tags_set;
  for (const auto& tag : tags) {
    tags_set.insert(tag);
  }

  MLIRContext context;
  auto module = tensorflow::SavedModelToMlirImport(
      saved_model_dir, tags_set, absl::MakeSpan(exported_names), &context);

  // Print to asm.
  std::string asm_output;
  llvm::raw_string_ostream sout(asm_output);
  OpPrintingFlags print_flags;
  module->print(sout, print_flags);
  return sout.str();
}

}  // namespace

void SetupTensorFlowBindings(pybind11::module m) {
  m.def("import_saved_model_to_mlir_asm", &ImportSavedModelToMlirAsm,
        py::arg("saved_model_dir"),
        py::arg("exported_names") = std::vector<std::string>(),
        py::arg("tags") = std::vector<std::string>({std::string("serve")}));
}

}  // namespace python
}  // namespace iree
