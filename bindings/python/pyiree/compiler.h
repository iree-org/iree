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
  std::string ToAsm(bool enableDebugInfo, bool prettyForm,
                    int64_t largeElementLimit);

  // Runs one or more pass pipelines (as is mlir::parsePassPipeline).
  void RunPassPipeline(const std::vector<std::string>& pipelines);

  // Compiles the MLIR module to an IREE sequencer module.
  std::shared_ptr<OpaqueBlob> CompileToSequencerBlob(
      bool print_mlir, std::vector<std::string> target_backends);

 private:
  std::shared_ptr<CompilerContextBundle> context_;
  mlir::ModuleOp module_op_;
};

// Registers to receive diagnostics for a scope.
// When this goes out of scope, any remaining diagnostics will be added to
// the parent.
class DiagnosticCapture {
 public:
  DiagnosticCapture(mlir::MLIRContext* mlir_context, DiagnosticCapture* parent);
  ~DiagnosticCapture();
  DiagnosticCapture(DiagnosticCapture&& other);

  std::vector<mlir::Diagnostic>& diagnostics() { return diagnostics_; }

  // Consumes/clears diagnostics.
  std::string ConsumeDiagnosticsAsString(const char* error_message);
  void ClearDiagnostics();

 private:
  mlir::MLIRContext* mlir_context_;
  DiagnosticCapture* parent_;
  std::vector<mlir::Diagnostic> diagnostics_;
  mlir::DiagnosticEngine::HandlerID handler_id_;
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

  // Gets the default diagnostic capture.
  DiagnosticCapture& DefaultDiagnosticCapture() { return default_capture_; }

  // Creates a new diagnostic region.
  // Note that this only supports one deep at present.
  DiagnosticCapture CaptureDiagnostics() {
    return DiagnosticCapture(&mlir_context_, &default_capture_);
  }

  // Consumes/clears diagnostics.
  std::string ConsumeDiagnosticsAsString() {
    return default_capture_.ConsumeDiagnosticsAsString(nullptr);
  }
  void ClearDiagnostics() { default_capture_.ClearDiagnostics(); }

 private:
  mlir::MLIRContext mlir_context_;
  DiagnosticCapture default_capture_;
};

void SetupCompilerBindings(pybind11::module m);

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_PYIREE_COMPILER_H_
