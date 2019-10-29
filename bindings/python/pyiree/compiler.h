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

#ifndef IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_
#define IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_

#include <string>

#include "bindings/python/pyiree/binding.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"

namespace iree {
namespace python {

class CompilerContextBundle;
class CompilerModuleBundle;

// Wraps an MLIR module and its producing context.
class CompilerModuleBundle {
 public:
  CompilerModuleBundle(std::shared_ptr<CompilerContextBundle> context,
                       mlir::ModuleOp module_op)
      : context_(std::move(context)), module_op_(std::move(module_op)) {}

  mlir::ModuleOp& module_op() { return module_op_; }
  std::string ToAsm();

  // Runs one or more pass pipelines (as is mlir::parsePassPipeline).
  void RunPassPipeline(const std::vector<std::string>& pipelines);

  // Compiles the MLIR module to an IREE sequencer module.
  std::shared_ptr<OpaqueBlob> CompileToSequencerBlob();

 private:
  std::shared_ptr<CompilerContextBundle> context_;
  mlir::ModuleOp module_op_;
};

// Bundle of MLIRContext related things that facilitates interop with
// Python.
class CompilerContextBundle
    : public std::enable_shared_from_this<CompilerContextBundle> {
 public:
  CompilerContextBundle();
  ~CompilerContextBundle();

  mlir::MLIRContext* mlir_context() { return &mlir_context_; }

  CompilerModuleBundle ParseAsm(const std::string& asm_text);

  // Consumes/clears diagnostics.
  std::string ConsumeDiagnosticsAsString();
  void ClearDiagnostics();

 private:
  mlir::MLIRContext mlir_context_;
  std::vector<mlir::Diagnostic> diagnostics_;
};

void SetupCompilerBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_
