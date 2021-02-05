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

// IREE source.mlir -> execution output test runner.
// This is meant to be called from LIT for FileCheck tests, and tries to match
// the interface of mlir-opt (featuring -split-input-file, etc) so it's easier
// to work with there. If you want a more generalized runner for standalone
// precompiled IREE modules use iree-run-module.
//
// By default all exported functions in the module will be run in order.
// All input values, provided via -function-inputs, will be passed to the
// functions (this means all input signatures must match). Results from the
// executed functions will be printed to stdout for checking.
//
// Example input:
// // RUN: iree-run-mlir %s | IreeFileCheck %s
// // CHECK-LABEL: @foo
// // CHECK: 1xf32: 2
// func @foo() -> tensor<f32> attributes {iree.module.export} {
//   %0 = constant dense<2.0> : tensor<f32>
//   return %0 : tensor<f32>
// }
//
// Command line arguments are handled by LLVM's parser by default but -- can be
// used to separate the compiler flags from the runtime flags, such as:
//   iree-run-mlir -iree-hal-target-backends=vulkan-spirv -- --logtostderr

#include <iostream>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Translation/IREEVM.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/init_dialects.h"
#include "iree/tools/init_targets.h"
#include "iree/tools/utils/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

static llvm::cl::opt<std::string> input_file_flag{
    llvm::cl::Positional,
    llvm::cl::desc("<input .mlir file>"),
    llvm::cl::init("-"),
};

static llvm::cl::opt<bool> split_input_file_flag{
    "split-input-file",
    llvm::cl::desc("Split the input file into multiple modules"),
    llvm::cl::init(true),
};

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> export_all_flag{
    "export-all",
    llvm::cl::desc("Adds iree.module.export to all functions"),
    llvm::cl::init(false),
};

static llvm::cl::opt<bool> print_mlir_flag{
    "print-mlir",
    llvm::cl::desc("Prints MLIR IR after translation"),
    llvm::cl::init(false),
};

static llvm::cl::opt<bool> print_annotated_mlir_flag{
    "print-annotated-mlir",
    llvm::cl::desc("Prints MLIR IR with final serialization annotations"),
    llvm::cl::init(false),
};

static llvm::cl::opt<bool> print_flatbuffer_flag{
    "print-flatbuffer",
    llvm::cl::desc("Prints Flatbuffer text after serialization"),
    llvm::cl::init(false),
};

static llvm::cl::list<std::string> function_inputs_flag{
    "function-input",
    llvm::cl::desc("Input shapes and optional values"),
    llvm::cl::ZeroOrMore,
};

static llvm::cl::opt<std::string> function_inputs_file_flag{
    "function-input-file",
    llvm::cl::desc("Provides a file for input shapes and optional values (see "
                   "ParseToVariantListFromFile in vm_util.h for details)"),
    llvm::cl::init(""),
};

static llvm::cl::opt<bool> run_flag{
    "run",
    llvm::cl::desc("Runs the module (vs. just compiling and verifing)"),
    llvm::cl::init(true),
};

static llvm::cl::list<std::string> run_args_flag{
    "run-arg",
    llvm::cl::desc("Argument passed to the execution flag parser"),
    llvm::cl::ZeroOrMore,
};

namespace iree {
namespace {

// Returns a driver name capable of handling input from the given backend.
std::string BackendToDriverName(std::string backend) {
  size_t dash = backend.find('-');
  if (dash == std::string::npos) {
    return backend;
  } else {
    return backend.substr(0, dash);
  }
}

// Returns a list of target compiler backends to use for file evaluation.
Status GetTargetBackends(std::vector<std::string>* out_target_backends) {
  IREE_TRACE_SCOPE();
  out_target_backends->clear();
  auto target_backends =
      mlir::iree_compiler::IREE::HAL::getTargetOptionsFromFlags().targets;
  if (target_backends.empty()) {
    iree_hal_driver_info_t* driver_infos = NULL;
    iree_host_size_t driver_info_count = 0;
    IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
        iree_hal_driver_registry_default(), iree_allocator_system(),
        &driver_infos, &driver_info_count));
    for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
      target_backends.push_back(std::string(driver_infos[i].driver_name.data,
                                            driver_infos[i].driver_name.size));
    }
    iree_allocator_system_free(NULL, driver_infos);
  }
  *out_target_backends = std::move(target_backends);
  return OkStatus();
}

// Prepares a module for evaluation by running MLIR import and IREE translation.
// Returns the serialized flatbuffer data.
Status PrepareModule(std::string target_backend,
                     std::unique_ptr<llvm::MemoryBuffer> file_buffer,
                     mlir::DialectRegistry& registry, std::string* out_module) {
  IREE_TRACE_SCOPE();
  out_module->clear();

  mlir::MLIRContext context;
  registry.appendTo(context.getDialectRegistry());

  // Parse input MLIR module.
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file_buffer), llvm::SMLoc());
  mlir::OwningModuleRef mlir_module =
      mlir::parseSourceFile(source_mgr, &context);
  if (!mlir_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "could not parse MLIR file");
  }

  if (export_all_flag) {
    for (auto function : mlir_module->getOps<mlir::FuncOp>()) {
      function->setAttr("iree.module.export", mlir::UnitAttr::get(&context));
    }
  }

  // Translate from MLIR to IREE bytecode.
  IREE_LOG(INFO) << "Compiling for target backend '" << target_backend
                 << "'...";
  auto hal_target_options =
      mlir::iree_compiler::IREE::HAL::getTargetOptionsFromFlags();
  hal_target_options.targets = {std::move(target_backend)};
  auto vm_target_options =
      mlir::iree_compiler::IREE::VM::getTargetOptionsFromFlags();
  mlir::PassManager pass_manager(mlir_module->getContext());
  pass_manager.enableVerifier(verifyPasses);
  mlir::applyPassManagerCLOptions(pass_manager);
  mlir::iree_compiler::IREE::Flow::buildFlowTransformPassPipeline(pass_manager);
  mlir::iree_compiler::IREE::HAL::buildHALTransformPassPipeline(
      pass_manager, hal_target_options);
  mlir::iree_compiler::IREE::VM::buildVMTransformPassPipeline(
      pass_manager, vm_target_options);
  pass_manager.addPass(
      mlir::iree_compiler::IREE::createDropCompilerHintsPass());
  if (failed(pass_manager.run(mlir_module.get()))) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "conversion from source -> vm failed");
  }

  if (print_mlir_flag) {
    mlir_module->dump();
  }

  auto bytecode_options =
      mlir::iree_compiler::IREE::VM::getBytecodeTargetOptionsFromFlags();
  std::string binary_contents;
  llvm::raw_string_ostream binary_output(binary_contents);
  if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
          mlir_module.get(), bytecode_options, binary_output))) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "serialization to flatbuffer bytecode (binary) failed");
  }
  binary_output.flush();

  // Print the annotated MLIR and flatbuffer; easiest way right now is to just
  // do it all again.
  if (print_annotated_mlir_flag) {
    bytecode_options.outputFormat =
        mlir::iree_compiler::IREE::VM::BytecodeOutputFormat::kAnnotatedMlirText;
    std::string text_contents;
    llvm::raw_string_ostream text_output(text_contents);
    if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
            mlir_module.get(), bytecode_options, text_output))) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "serialization to annotated MLIR (text) failed");
    }
    text_output.flush();
    std::cerr << text_contents << std::endl;
  }
  if (print_flatbuffer_flag) {
    bytecode_options.outputFormat =
        mlir::iree_compiler::IREE::VM::BytecodeOutputFormat::kFlatBufferText;
    std::string text_contents;
    llvm::raw_string_ostream text_output(text_contents);
    if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
            mlir_module.get(), bytecode_options, text_output))) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "serialization to flatbuffer bytecode (text) failed");
    }
    text_output.flush();
    std::cerr << text_contents << std::endl;
  }

  *out_module = std::move(binary_contents);
  return OkStatus();
}

// Evaluates a single function in its own fiber, printing the results to stdout.
Status EvaluateFunction(iree_vm_context_t* context,
                        iree_hal_allocator_t* allocator,
                        iree_vm_function_t function,
                        absl::string_view export_name) {
  IREE_TRACE_SCOPE();

  std::cout << "EXEC @" << export_name << std::endl;
  std::vector<RawSignatureParser::Description> input_descs;
  IREE_RETURN_IF_ERROR(ParseInputSignature(function, &input_descs));
  vm::ref<iree_vm_list_t> inputs;
  if (!function_inputs_file_flag.empty()) {
    if (!function_inputs_flag.empty()) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected only one of function_inputs and "
                              "function_inputs_file to be set");
    }
    IREE_RETURN_IF_ERROR(ParseToVariantListFromFile(
        input_descs, allocator, function_inputs_file_flag, &inputs));
  } else {
    auto function_inputs_list = absl::MakeConstSpan(
        function_inputs_flag.empty() ? nullptr : &function_inputs_flag.front(),
        function_inputs_flag.size());
    IREE_RETURN_IF_ERROR(ParseToVariantList(input_descs, allocator,
                                            function_inputs_list, &inputs));
  }

  std::vector<RawSignatureParser::Description> output_descs;
  IREE_RETURN_IF_ERROR(ParseOutputSignature(function, &output_descs));
  // Prepare outputs list to accept the results from the invocation.
  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr,
                                           output_descs.size(),
                                           iree_allocator_system(), &outputs));

  // Synchronously invoke the function.
  IREE_RETURN_IF_ERROR(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                      inputs.get(), outputs.get(),
                                      iree_allocator_system()));

  // Print outputs.
  IREE_RETURN_IF_ERROR(PrintVariantList(output_descs, outputs.get()));

  return OkStatus();
}

// Evaluates all exported functions within given module.
Status EvaluateFunctions(iree_vm_instance_t* instance,
                         absl::string_view driver_name,
                         const std::string& flatbuffer_data) {
  IREE_TRACE_SCOPE0("EvaluateFunctions");

  IREE_LOG(INFO) << "Evaluating all functions in module for driver '"
                 << driver_name << "'...";

  // Load the bytecode module from the flatbuffer data.
  // We do this first so that if we fail validation we know prior to dealing
  // with devices.
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_RETURN_IF_ERROR(LoadBytecodeModule(flatbuffer_data, &bytecode_module));

  if (!run_flag) {
    // Just wanted verification; return without running.
    iree_vm_module_release(bytecode_module);
    return OkStatus();
  }

  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(CreateDevice(driver_name, &device));
  iree_vm_module_t* hal_module = nullptr;
  IREE_RETURN_IF_ERROR(CreateHalModule(device, &hal_module));

  // Evaluate all exported functions.
  auto run_function = [&](int ordinal) -> Status {
    iree_vm_function_t function;
    iree_string_view_t export_name_isv;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
                             bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                             ordinal, &function, &export_name_isv),
                         "Looking up function export %d", ordinal);
    absl::string_view export_name(export_name_isv.data, export_name_isv.size);
    if (absl::StartsWith(export_name, "__") ||
        export_name.find('$') != absl::string_view::npos) {
      // Skip internal or special functions.
      return OkStatus();
    }
    IREE_RETURN_IF_ERROR(ValidateFunctionAbi(function));

    // Create the context we'll use for this (ensuring that we can't interfere
    // with other running evaluations, such as when in a multithreaded test
    // runner).
    iree_vm_context_t* context = nullptr;
    std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
    IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
                             instance, modules.data(), modules.size(),
                             iree_allocator_system(), &context),
                         "Creating context");

    // Invoke the function and print results.
    IREE_RETURN_IF_ERROR(
        EvaluateFunction(context, iree_hal_device_allocator(device), function,
                         export_name),
        "Evaluating export function %d", ordinal);

    iree_vm_context_release(context);
    return OkStatus();
  };

  Status evaluate_status = OkStatus();
  auto module_signature = iree_vm_module_signature(bytecode_module);
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    evaluate_status = run_function(i);
    if (!evaluate_status.ok()) {
      break;
    }
  }

  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);
  iree_hal_device_release(device);

  return evaluate_status;
}

// Translates and runs a single LLVM file buffer.
Status EvaluateFile(std::unique_ptr<llvm::MemoryBuffer> file_buffer,
                    mlir::DialectRegistry& registry) {
  IREE_TRACE_SCOPE0("EvaluateFile");

  // TODO(benvanik): move to instance-based registration.
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types(),
                       "Registering HAL types");

  iree_vm_instance_t* instance = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance),
      "Creating instance");

  std::vector<std::string> target_backends;
  IREE_RETURN_IF_ERROR(GetTargetBackends(&target_backends));
  for (const auto& target_backend : target_backends) {
    // Prepare the module for execution and evaluate it.
    IREE_TRACE_FRAME_MARK();
    auto cloned_file_buffer = llvm::MemoryBuffer::getMemBufferCopy(
        file_buffer->getBuffer(), file_buffer->getBufferIdentifier());
    std::string flatbuffer_data;
    IREE_RETURN_IF_ERROR(
        PrepareModule(target_backend + '*', std::move(cloned_file_buffer),
                      registry, &flatbuffer_data),
        "Translating module");
    IREE_TRACE_FRAME_MARK();
    IREE_RETURN_IF_ERROR(
        EvaluateFunctions(instance, BackendToDriverName(target_backend),
                          flatbuffer_data),
        "Evaluating functions");
  }

  iree_vm_instance_release(instance);
  return OkStatus();
}

// Runs the given .mlir file based on the current flags.
Status RunFile(const std::string& mlir_filename,
               mlir::DialectRegistry& registry) {
  IREE_TRACE_SCOPE0("RunFile");

  // Load input file/from stdin.
  std::string error_message;
  auto file = mlir::openInputFile(mlir_filename, &error_message);
  if (!file) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "unable to open input file %.*s: %s",
        (int)mlir_filename.size(), mlir_filename.data(), error_message.c_str());
  }

  if (!split_input_file_flag) {
    // Use entire buffer as a single module.
    return EvaluateFile(std::move(file), registry);
  }

  // Split the buffer into separate modules and evaluate independently.
  // This matches the -split-input-file arg to mlir-opt.
  const char kSplitMarker[] = "// -----";
  auto* full_buffer = file.get();
  llvm::SmallVector<llvm::StringRef, 8> source_buffers;
  full_buffer->getBuffer().split(source_buffers, kSplitMarker);

  // Add the original buffer to the source manager.
  llvm::SourceMgr file_source_mgr;
  file_source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Process each chunk in turn. Only return the first error (but log all).
  Status any_failure;
  for (auto& sub_source_buffer : source_buffers) {
    auto split_loc = llvm::SMLoc::getFromPointer(sub_source_buffer.data());
    unsigned split_line = file_source_mgr.getLineAndColumn(split_loc).first;
    auto sub_buffer = llvm::MemoryBuffer::getMemBufferCopy(
        sub_source_buffer, full_buffer->getBufferIdentifier() +
                               llvm::Twine(" split at line #") +
                               llvm::Twine(split_line));
    auto sub_failure = EvaluateFile(std::move(sub_buffer), registry);
    if (!sub_failure.ok()) {
      IREE_LOG(ERROR) << "Failure for split at line #" << split_line << ": "
                      << sub_failure;
      if (any_failure.ok()) {
        any_failure = std::move(sub_failure);
      }
    }
  }

  return any_failure;
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  IREE_TRACE_SCOPE0("iree-run-mlir");

  int argc_llvm = argc;
  char** argv_llvm = argv;
  int argc_absl = 1;
  std::vector<char*> argv_absl = {argv[0]};
  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--") == 0) {
      argc_llvm = i;
      argc_absl = argc - i;
      for (int j = i + 1; j < argc; ++j) {
        argv_absl.push_back(argv[i + 1]);
      }
      break;
    }
  }

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerVMTargets();

  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register assembly printer command-line options like
  // -mlir-print-op-generic.
  mlir::registerAsmPrinterCLOptions();
  // Register pass manager command-line options like -print-ir-*.
  mlir::registerPassManagerCLOptions();

  llvm::InitLLVM init_llvm(argc_llvm, argv_llvm);
  llvm::cl::ParseCommandLineOptions(argc_llvm, argv_llvm);

  for (auto& run_arg : run_args_flag) {
    argv_absl.push_back(const_cast<char*>(run_arg.c_str()));
  }
  argc_absl += run_args_flag.size();
  char** argv_absl_ptr = argv_absl.data();
  iree_flags_parse_checked(&argc_absl, &argv_absl_ptr);
  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  auto status = RunFile(input_file_flag, registry);
  if (!status.ok()) {
    std::cerr << "ERROR running file (" << input_file_flag << "): " << status
              << "\n";
    return 1;
  }
  return 0;
}

}  // namespace iree
