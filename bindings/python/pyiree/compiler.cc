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

#include "bindings/python/pyiree/compiler.h"

#include <stdexcept>

#include "bindings/python/pyiree/binding.h"
#include "bindings/python/pyiree/status_utils.h"
#include "iree/compiler/Translation/Sequencer/SequencerModuleTranslation.h"
#include "iree/schemas/module_def_generated.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace py = pybind11;

using namespace mlir;
using namespace mlir::iree_compiler;

using llvm::MemoryBuffer;
using llvm::MemoryBufferRef;
using llvm::StringRef;

namespace iree {
namespace python {

namespace {

OwningModuleRef parseMLIRModuleFromString(StringRef contents,
                                          MLIRContext* context) {
  std::unique_ptr<MemoryBuffer> contents_buffer;
  if (contents.back() == 0) {
    // If it has a nul terminator, just use as-is.
    contents_buffer = MemoryBuffer::getMemBuffer(contents.drop_back());
  } else {
    // Otherwise, make a copy.
    contents_buffer = MemoryBuffer::getMemBufferCopy(contents, "EMBED");
  }

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(contents_buffer), llvm::SMLoc());
  OwningModuleRef mlir_module = parseSourceFile(source_mgr, context);
  return mlir_module;
}

}  // namespace

DiagnosticCapture::DiagnosticCapture(mlir::MLIRContext* mlir_context,
                                     DiagnosticCapture* parent)
    : mlir_context_(mlir_context), parent_(parent) {
  handler_id_ = mlir_context_->getDiagEngine().registerHandler(
      [&](Diagnostic& d) -> LogicalResult {
        diagnostics_.push_back(std::move(d));
        return success();
      });
}
DiagnosticCapture::~DiagnosticCapture() {
  if (mlir_context_) {
    mlir_context_->getDiagEngine().eraseHandler(handler_id_);
    if (parent_) {
      for (auto& d : diagnostics_) {
        parent_->diagnostics_.push_back(std::move(d));
      }
    }
  }
}

DiagnosticCapture::DiagnosticCapture(DiagnosticCapture&& other) {
  mlir_context_ = other.mlir_context_;
  parent_ = other.parent_;
  diagnostics_.swap(other.diagnostics_);
  handler_id_ = other.handler_id_;
  other.mlir_context_ = nullptr;
}

std::string DiagnosticCapture::ConsumeDiagnosticsAsString(
    const char* error_message) {
  std::string s;
  llvm::raw_string_ostream sout(s);
  bool first = true;
  if (error_message) {
    sout << error_message;
    first = false;
  }
  for (auto& d : diagnostics_) {
    if (!first) {
      sout << "\n\n";
    } else {
      first = false;
    }

    switch (d.getSeverity()) {
      case DiagnosticSeverity::Note:
        sout << "[NOTE]";
        break;
      case DiagnosticSeverity::Warning:
        sout << "[WARNING]";
        break;
      case DiagnosticSeverity::Error:
        sout << "[ERROR]";
        break;
      case DiagnosticSeverity::Remark:
        sout << "[REMARK]";
        break;
      default:
        sout << "[UNKNOWN]";
    }
    // Message.
    sout << ": " << d << "\n\t";

    // Location.
    d.getLocation().print(sout);
  }

  diagnostics_.clear();
  return sout.str();
}

void DiagnosticCapture::ClearDiagnostics() { diagnostics_.clear(); }

CompilerContextBundle::CompilerContextBundle()
    : default_capture_(&mlir_context_, nullptr) {}
CompilerContextBundle::~CompilerContextBundle() = default;

CompilerModuleBundle CompilerContextBundle::ParseAsm(
    const std::string& asm_text) {
  // Arrange to get a view that includes a terminating null to avoid additional
  // copy.
  const char* asm_chars = asm_text.c_str();
  StringRef asm_sr(asm_chars, asm_text.size() + 1);

  auto diag_capture = CaptureDiagnostics();
  auto module_ref = parseMLIRModuleFromString(asm_sr, mlir_context());
  if (!module_ref) {
    throw RaiseValueError(
        diag_capture.ConsumeDiagnosticsAsString("Error parsing ASM").c_str());
  }
  return CompilerModuleBundle(shared_from_this(), module_ref.release());
}

std::string CompilerModuleBundle::ToAsm() {
  // Print to asm.
  std::string asm_output;
  llvm::raw_string_ostream sout(asm_output);
  OpPrintingFlags print_flags;
  module_op().print(sout, print_flags);
  return sout.str();
}

std::shared_ptr<OpaqueBlob> CompilerModuleBundle::CompileToSequencerBlob() {
  auto diag_capture = context_->CaptureDiagnostics();
  auto module_blob =
      mlir::iree_compiler::translateMlirToIreeSequencerModule(module_op());
  if (module_blob.empty()) {
    throw RaiseValueError(
        diag_capture
            .ConsumeDiagnosticsAsString("Failed to translate MLIR module")
            .c_str());
  }
  return std::make_shared<OpaqueByteVectorBlob>(std::move(module_blob));
}

void CompilerModuleBundle::RunPassPipeline(
    const std::vector<std::string>& pipelines) {
  mlir::PassManager pm(context_->mlir_context());

  // Parse the pass pipelines.
  std::string error;
  llvm::raw_string_ostream error_stream(error);
  for (const auto& pipeline : pipelines) {
    if (failed(mlir::parsePassPipeline(pipeline, pm, error_stream))) {
      throw RaiseValueError(error_stream.str().c_str());
    }
  }

  // Run them.
  auto diag_capture = context_->CaptureDiagnostics();
  if (failed(pm.run(module_op_))) {
    throw RaisePyError(
        PyExc_RuntimeError,
        diag_capture.ConsumeDiagnosticsAsString("Error running pass pipelines:")
            .c_str());
  }
}

void SetupCompilerBindings(pybind11::module m) {
  py::class_<CompilerContextBundle, std::shared_ptr<CompilerContextBundle>>(
      m, "CompilerContext")
      .def(py::init<>([]() {
        // Need explicit make_shared to avoid UB with enable_shared_from_this.
        return std::make_shared<CompilerContextBundle>();
      }))
      .def("parse_asm", &CompilerContextBundle::ParseAsm)
      .def("get_diagnostics",
           &CompilerContextBundle::ConsumeDiagnosticsAsString)
      .def("clear_diagnostics", &CompilerContextBundle::ClearDiagnostics);
  py::class_<CompilerModuleBundle>(m, "CompilerModule")
      .def("to_asm", &CompilerModuleBundle::ToAsm)
      .def("compile_to_sequencer_blob",
           &CompilerModuleBundle::CompileToSequencerBlob)
      .def("run_pass_pipeline", &CompilerModuleBundle::RunPassPipeline);
}

}  // namespace python
}  // namespace iree
