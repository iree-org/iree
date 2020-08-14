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

#include "bindings/python/pyiree/compiler/compiler.h"

#include <stdexcept>
#include <string>

#include "bindings/python/pyiree/common/binding.h"
#include "bindings/python/pyiree/common/status_utils.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/tools/init_compiler_modules.h"
#include "iree/tools/init_iree_dialects.h"
#include "iree/tools/init_iree_passes.h"
#include "iree/tools/init_mlir_dialects.h"
#include "iree/tools/init_mlir_passes.h"
#include "iree/tools/init_targets.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace py = pybind11;

using namespace mlir;
using namespace mlir::iree_compiler;

using mlir::iree_compiler::IREE::HAL::TargetOptions;
using mlir::iree_compiler::IREE::VM::BytecodeOutputFormat;
using mlir::iree_compiler::IREE::VM::BytecodeTargetOptions;

using llvm::MemoryBuffer;
using llvm::MemoryBufferRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::StringRef;

namespace iree {
namespace python {

/* static */ std::mutex CompilerContextBundle::static_config_lock_;
/* static */ absl::optional<std::string>
    CompilerContextBundle::default_crash_reproducer_path_;

namespace {

bool LLVMOnceInit() {
  // Enable LLVM's signal handler to get nice stack traces.
  llvm::sys::SetOneShotPipeSignalFunction(
      llvm::sys::DefaultOneShotPipeSignalHandler);
  llvm::sys::PrintStackTraceOnErrorSignal("pyiree");

  // Register built-in MLIR dialects.
  mlir::registerMlirDialects();

  // Register IREE dialects, compiler module dialects, and HAL target backends.
  mlir::iree_compiler::registerIreeDialects();
  mlir::iree_compiler::registerIreeCompilerModuleDialects();
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerVMTargets();

  // Depending on the build environment the MLIR Passes may already be
  // registered. Conditionally register passes until re-registration is
  // supported.
#ifdef IREE_REGISTER_MLIR_PASSES
  mlir::registerMlirPasses();
#endif
  // Register IREE dialects.
  mlir::iree_compiler::registerAllIreePasses();

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();

  std::string program_name = "pyiree";
  std::vector<const char*> default_options = {program_name.c_str(), nullptr};
  llvm::cl::ParseCommandLineOptions(1, default_options.data());
  return true;
}

void SetupLLVMModule(pybind11::module m) {
  m.def("print_help_message", []() { llvm::cl::PrintHelpMessage(); });
  m.def(
      "add_option",
      [](std::string name, absl::optional<std::string> value) {
        auto options_map = llvm::cl::getRegisteredOptions();
        auto found_it = options_map.find(name);
        if (found_it == options_map.end()) {
          std::string message = "Unknown LLVM option: ";
          message.append(name);
          throw RaiseValueError(message.c_str());
        }

        std::string value_sr = value ? *value : "";
        found_it->getValue()->addOccurrence(1, name, value_sr);
      },
      py::arg("name"), py::arg("value") = absl::optional<std::string>());
  m.def(
      "reset_option",
      [](std::string name) {
        auto options_map = llvm::cl::getRegisteredOptions();
        auto found_it = options_map.find(name);
        if (found_it == options_map.end()) {
          std::string message = "Unknown LLVM option: ";
          message.append(name);
          throw RaiseValueError(message.c_str());
        }
        found_it->getValue()->setDefault();
      },
      py::arg("name"));
}

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

// Custom location printer that prints prettier, multi-line file output
// suitable for human readable error messages. The standard printer just prints
// a long nested expression not particularly human friendly). Note that there
// is a location pretty printer in the MLIR AsmPrinter. It is private and
// doesn't do any path shortening, which seems to make long Python stack traces
// a bit easier to scan.
void PrintLocation(Location loc, raw_ostream& out) {
  TypeSwitch<Location>(loc)
      .Case<OpaqueLoc>(
          [&](OpaqueLoc loc) { PrintLocation(loc.getFallbackLocation(), out); })
      .Case<UnknownLoc>([&](UnknownLoc) { out << "  [unknown location]\n"; })
      .Case<FileLineColLoc>([&](FileLineColLoc line_col_loc) {
        StringRef this_filename = line_col_loc.getFilename();
        auto slash_pos = this_filename.find_last_of("/\\");
        // We print both the basename and extended names with a structure like
        // `foo.py:35:4`. Even though technically the line/col
        // information is redundant to include in both names, having it on both
        // makes it easier to paste the paths into an editor and jump to the
        // exact location.
        std::string line_col_suffix =
            ":" + std::to_string(line_col_loc.getLine()) + ":" +
            std::to_string(line_col_loc.getColumn());
        bool has_basename = false;
        StringRef basename = this_filename;
        if (slash_pos != StringRef::npos) {
          has_basename = true;
          basename = this_filename.substr(slash_pos + 1);
        }
        out << "  at: " << basename << line_col_suffix;
        if (has_basename) {
          // When running through bazel, such as in our e2e test suite,
          // the paths involved can be quite large, and will have a very long
          // prefix before the sandboxed "runfiles" directory that the program
          // runs in. Trim off that long prefix. By convention, the path names
          // with this prefix dropped will correspond to the path in the source
          // directory, which is probably what we want anyway.
          StringRef kRunfiles(".runfiles/");
          StringRef extended_name = this_filename;
          auto runfiles_pos = extended_name.rfind(kRunfiles);
          if (runfiles_pos != StringRef::npos) {
            extended_name =
                extended_name.drop_front(runfiles_pos + kRunfiles.size());
          }
          // Print out two tabs, as basenames usually vary in length by more
          // than one tab width.
          out << "\t\t( " << extended_name << line_col_suffix << " )";
        }
        out << "\n";
      })
      .Case<NameLoc>([&](NameLoc name_loc) {
        out << "  @'" << name_loc.getName() << "':\n";
        auto child_loc = name_loc.getChildLoc();
        if (!child_loc.isa<UnknownLoc>()) {
          out << "(...\n";
          PrintLocation(child_loc, out);
          out << ")\n";
        }
      })
      .Case<CallSiteLoc>([&](CallSiteLoc call_site) {
        PrintLocation(call_site.getCaller(), out);
        PrintLocation(call_site.getCallee(), out);
      });
}

std::string DiagnosticCapture::ConsumeDiagnosticsAsString(
    const char* error_message) {
  std::string s;
  raw_string_ostream sout(s);
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
    sout << ": " << d << "\n";
    PrintLocation(d.getLocation(), sout);
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

std::string CompilerModuleBundle::ToAsm(bool enableDebugInfo, bool prettyForm,
                                        int64_t largeElementLimit) {
  // Print to asm.
  std::string asm_output;
  raw_string_ostream sout(asm_output);
  OpPrintingFlags print_flags;
  if (enableDebugInfo) {
    print_flags.enableDebugInfo(prettyForm);
  }
  if (largeElementLimit >= 0) {
    print_flags.elideLargeElementsAttrs(largeElementLimit);
  }
  module_op().print(sout, print_flags);
  return sout.str();
}

std::shared_ptr<OpaqueBlob> CompilerModuleBundle::Compile(
    BytecodeTargetOptions options, std::vector<std::string> target_backends) {
  mlir::PassManager pass_manager(context_->mlir_context());
  mlir::applyPassManagerCLOptions(pass_manager);
  auto crash_reproducer_path = context_->crash_reproducer_path();
  if (crash_reproducer_path) {
    pass_manager.enableCrashReproducerGeneration(*crash_reproducer_path, true);
  }

  mlir::iree_compiler::IREE::HAL::TargetOptions hal_target_options;
  if (target_backends.empty()) {
    hal_target_options.targets =
        mlir::iree_compiler::IREE::HAL::getRegisteredTargetBackends();
  } else {
    hal_target_options.targets = std::move(target_backends);
  }

  auto vm_target_options =
      mlir::iree_compiler::IREE::VM::getTargetOptionsFromFlags();

  mlir::iree_compiler::IREE::Flow::buildFlowTransformPassPipeline(pass_manager);
  mlir::iree_compiler::IREE::HAL::buildHALTransformPassPipeline(
      pass_manager, hal_target_options);
  mlir::iree_compiler::IREE::VM::buildVMTransformPassPipeline(
      pass_manager, vm_target_options);

  // Run primary passes.
  auto diag_capture = context_->CaptureDiagnostics();
  if (failed(pass_manager.run(module_op_))) {
    throw RaisePyError(
        PyExc_RuntimeError,
        diag_capture.ConsumeDiagnosticsAsString("Error compiling IREE module:")
            .c_str());
  }

  // Run serialization.
  std::string contents;
  raw_string_ostream out(contents);
  if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
          module_op_, options, out))) {
    throw RaisePyError(
        PyExc_RuntimeError,
        diag_capture
            .ConsumeDiagnosticsAsString("Error serializing to flatbuffer:")
            .c_str());
  }

  out.flush();
  return std::make_shared<OpaqueStringBlob>(std::move(out.str()));
}

void CompilerModuleBundle::RunPassPipeline(
    const std::vector<std::string>& pipelines) {
  mlir::PassManager pm(context_->mlir_context());
  mlir::applyPassManagerCLOptions(pm);
  auto crash_reproducer_path = context_->crash_reproducer_path();
  if (crash_reproducer_path) {
    pm.enableCrashReproducerGeneration(*crash_reproducer_path);
  }

  // Parse the pass pipelines.
  std::string error;
  raw_string_ostream error_stream(error);
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

void SetupCommonCompilerBindings(pybind11::module m) {
  // Guard the once init to happen once per process (vs module, which in
  // mondo builds can happen multiple times).
  static bool llvm_init_baton = ([]() { return LLVMOnceInit(); })();
  (void)(llvm_init_baton);

  // llvm module
  auto llvm_m = m.def_submodule("llvm", "Global LLVM configuration");
  SetupLLVMModule(llvm_m);

  // OpaqueBlob class
  py::class_<OpaqueBlob, std::shared_ptr<OpaqueBlob>>(m, "OpaqueBlob",
                                                      py::buffer_protocol())
      .def_buffer([](OpaqueBlob* self) -> py::buffer_info {
        return py::buffer_info(
            self->data(),                           // Pointer to buffer
            sizeof(uint8_t),                        // Size of one scalar
            py::format_descriptor<uint8_t>::value,  // Python struct-style
                                                    // format
            1,                                      // Number of dimensions
            {self->size()},                         // Buffer dimensions
            {self->size()}                          // Strides
        );
      })
      .def_property_readonly("bytes",
                             [](OpaqueBlob* self) -> py::bytes {
                               return py::bytes(
                                   static_cast<const char*>(self->data()),
                                   self->size());
                             })
      .def_property_readonly("text", [](OpaqueBlob* self) -> py::str {
        return py::str(static_cast<const char*>(self->data()), self->size());
      });

  // CompilerContext class
  py::class_<CompilerContextBundle, std::shared_ptr<CompilerContextBundle>>(
      m, "CompilerContext")
      .def(py::init<>([]() {
        // Need explicit make_shared to avoid UB with enable_shared_from_this.
        return std::make_shared<CompilerContextBundle>();
      }))
      .def("parse_asm", &CompilerContextBundle::ParseAsm)
      .def("get_diagnostics",
           &CompilerContextBundle::ConsumeDiagnosticsAsString)
      .def("clear_diagnostics", &CompilerContextBundle::ClearDiagnostics)
      .def_property_static(
          "default_crash_reproducer_path",
          [](py::object /* cls */) {
            return CompilerContextBundle::default_crash_reproducer_path();
          },
          [](py::object /* cls */, absl::optional<std::string> p) {
            CompilerContextBundle::set_default_crash_reproducer_path(
                std::move(p));
          })
      .def_property("crash_reproducer_path",
                    &CompilerContextBundle::crash_reproducer_path,
                    &CompilerContextBundle::set_crash_reproducer_path);

  // OutputFormat enum
  py::enum_<BytecodeOutputFormat>(m, "OutputFormat")
      .value("FLATBUFFER_BINARY", BytecodeOutputFormat::kFlatBufferBinary)
      .value("FLATBUFFER_TEXT", BytecodeOutputFormat::kFlatBufferText)
      .value("MLIR_TEXT", BytecodeOutputFormat::kMlirText)
      .export_values();

  // CompileOptions class
  py::class_<BytecodeTargetOptions>(m, "CompileOptions")
      .def(py::init<>())
      .def_readwrite("output_format", &BytecodeTargetOptions::outputFormat)
      .def_readwrite("optimize", &BytecodeTargetOptions::optimize)
      .def_readwrite("strip_debug_ops", &BytecodeTargetOptions::stripDebugOps)
      .def_readwrite("strip_source_map", &BytecodeTargetOptions::stripSourceMap)
      .def_readwrite("strip_symbols", &BytecodeTargetOptions::stripSymbols);

  // CompilerModule class
  py::class_<CompilerModuleBundle>(m, "CompilerModule")
      .def("to_asm", &CompilerModuleBundle::ToAsm,
           py::arg("debug_info") = false, py::arg("pretty") = false,
           py::arg("large_element_limit") = -1)
      .def("compile", &CompilerModuleBundle::Compile,
           py::arg("options") = BytecodeTargetOptions{},
           py::arg("target_backends") = std::vector<std::string>())
      .def("run_pass_pipeline", &CompilerModuleBundle::RunPassPipeline,
           py::arg("pipelines") = std::vector<std::string>());
}

}  // namespace python
}  // namespace iree
