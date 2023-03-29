// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
// // RUN: iree-run-mlir %s | FileCheck %s
// // CHECK-LABEL: @foo
// // CHECK: 1xf32: 2
// func.func @foo() -> tensor<f32> {
//   %0 = arith.constant dense<2.0> : tensor<f32>
//   return %0 : tensor<f32>
// }
//
// Command line arguments are handled by LLVM's parser by default but -- can be
// used to separate the compiler flags from the runtime flags, such as:
//   iree-run-mlir --iree-hal-target-backends=vulkan-spirv -- --logtostderr

#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/init_targets.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_targets.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

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

static llvm::cl::opt<bool> verify_passes_flag(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

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

static llvm::cl::opt<std::string> output_file_flag{
    "o",
    llvm::cl::desc("File path in which to write the compiled module file"),
    llvm::cl::init(""),
};

static llvm::cl::opt<bool> run_flag{
    "run",
    llvm::cl::desc("Runs the module (vs. just compiling and verifying)"),
    llvm::cl::init(true),
};

static llvm::cl::list<std::string> run_args_flag{
    "run-arg",
    llvm::cl::desc("Argument passed to the execution flag parser"),
    llvm::cl::ConsumeAfter,
};

IREE_FLAG_LIST(
    string, input,
    "An input (a) value or (b) buffer of the format:\n"
    "  (a) scalar value\n"
    "     value\n"
    "     e.g.: --input=\"3.14\"\n"
    "  (b) buffer:\n"
    "     [shape]xtype=[value]\n"
    "     e.g.: --input=\"2x2xi32=1 2 3 4\"\n"
    "Optionally, brackets may be used to separate the element values:\n"
    "  2x2xi32=[[1 2][3 4]]\n"
    "Raw binary files can be read to provide buffer contents:\n"
    "  2x2xi32=@some/file.bin\n"
    "\n"
    "Numpy npy files from numpy.save can be read to provide 1+ values:\n"
    "  @some.npy\n"
    "\n"
    "Each occurrence of the flag indicates an input in the order they were\n"
    "specified on the command line.");

IREE_FLAG_LIST(
    string, output,
    "Specifies how to handle an output from the invocation:\n"
    "  `` (empty): ignore output\n"
    "     e.g.: --output=\n"
    "  `-`: print textual form to stdout\n"
    "     e.g.: --output=-\n"
    "  `@file.npy`: create/overwrite a numpy npy file and write buffer view\n"
    "     e.g.: --output=@file.npy\n"
    "  `+file.npy`: create/append a numpy npy file and write buffer view\n"
    "     e.g.: --output=+file.npy\n"
    "\n"
    "Numpy npy files can be read in Python using numpy.load, for example an\n"
    "invocation producing two outputs can be concatenated as:\n"
    "    --output=@file.npy --output=+file.npy\n"
    "And then loaded in Python by reading from the same file:\n"
    "  with open('file.npy', 'rb') as f:\n"
    "    print(numpy.load(f))\n"
    "    print(numpy.load(f))\n"
    "\n"
    "Each occurrence of the flag indicates an output in the order they were\n"
    "specified on the command line.");

IREE_FLAG(int32_t, output_max_element_count, 1024,
          "Prints up to the maximum number of elements of output tensors, "
          "eliding the remainder.");

namespace iree {
namespace {

// Tries to guess a default device name from the backend, where possible.
// Users are still able to override this by passing in --device= flags.
std::string InferDefaultDeviceFromBackend(const std::string& backend) {
  if (backend == "vmvx" || backend == "llvm-cpu") {
    return "local-task";
  } else if (backend == "vmvx-inline") {
    return "";
  }
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
      mlir::iree_compiler::IREE::HAL::TargetOptions::FromFlags::get().targets;
  if (target_backends.empty()) {
    iree_allocator_t host_allocator = iree_allocator_system();
    iree_host_size_t driver_info_count = 0;
    iree_hal_driver_info_t* driver_infos = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
        iree_hal_available_driver_registry(), host_allocator,
        &driver_info_count, &driver_infos));
    for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
      target_backends.push_back(std::string(driver_infos[i].driver_name.data,
                                            driver_infos[i].driver_name.size));
    }
    iree_allocator_free(host_allocator, driver_infos);
  }
  *out_target_backends = std::move(target_backends);
  return OkStatus();
}

void BuildDefaultIREEVMTransformPassPipeline(mlir::OpPassManager& passManager) {
  static mlir::iree_compiler::IREEVMPipelineHooks defaultHooks = {
      // buildConstEvalPassPipelineCallback =
      [](mlir::OpPassManager& pm) {
        pm.addPass(mlir::iree_compiler::ConstEval::createJitGlobalsPass());
      }};

  buildIREEVMTransformPassPipeline(
      mlir::iree_compiler::BindingOptions::FromFlags::get(),
      mlir::iree_compiler::InputDialectOptions::FromFlags::get(),
      mlir::iree_compiler::PreprocessingOptions::FromFlags::get(),
      mlir::iree_compiler::HighLevelOptimizationOptions::FromFlags::get(),
      mlir::iree_compiler::SchedulingOptions::FromFlags::get(),
      mlir::iree_compiler::IREE::HAL::TargetOptions::FromFlags::get(),
      mlir::iree_compiler::IREE::VM::TargetOptions::FromFlags::get(),
      defaultHooks, passManager);
}

// Prepares a module for evaluation by running MLIR import and IREE translation.
// Returns the serialized flatbuffer data.
Status PrepareModule(std::string target_backend,
                     std::unique_ptr<llvm::MemoryBuffer> file_buffer,
                     mlir::DialectRegistry& registry, std::string* out_module) {
  IREE_TRACE_SCOPE();
  out_module->clear();

  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);

  // Parse input MLIR module.
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file_buffer), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context);
  if (!mlir_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "could not parse MLIR file");
  }

  // Translate from MLIR to IREE bytecode.
  printf("Compiling for target backend '%s'...\n", target_backend.c_str());
  mlir::PassManager pass_manager(mlir_module->getContext());
  pass_manager.enableVerifier(verify_passes_flag);
  mlir::applyPassManagerCLOptions(pass_manager);
  mlir::applyDefaultTimingPassManagerCLOptions(pass_manager);
  BuildDefaultIREEVMTransformPassPipeline(pass_manager);
  if (failed(pass_manager.run(mlir_module.get()))) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "conversion from source -> vm failed");
  }

  if (print_mlir_flag) {
    mlir_module->dump();
  }

  // NOTE: if we have an output file specified then we could compile into that
  // for greater efficiency. Today we assume that users aren't passing multi-GB
  // models through this tool (or if they are they have the memory to run them).
  auto vm_options =
      mlir::iree_compiler::IREE::VM::TargetOptions::FromFlags::get();
  auto bytecode_options =
      mlir::iree_compiler::IREE::VM::BytecodeTargetOptions::FromFlags::get();
  std::string binary_contents;
  llvm::raw_string_ostream binary_output(binary_contents);
  if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
          mlir_module.get(), vm_options, bytecode_options, binary_output))) {
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
            mlir_module.get(), vm_options, bytecode_options, text_output))) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "serialization to annotated MLIR (text) failed");
    }
    text_output.flush();
    fprintf(stderr, "%s\n", text_contents.c_str());
  }
  if (print_flatbuffer_flag) {
    bytecode_options.outputFormat =
        mlir::iree_compiler::IREE::VM::BytecodeOutputFormat::kFlatBufferText;
    std::string text_contents;
    llvm::raw_string_ostream text_output(text_contents);
    if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
            mlir_module.get(), vm_options, bytecode_options, text_output))) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "serialization to flatbuffer bytecode (text) failed");
    }
    text_output.flush();
    fprintf(stderr, "%s\n", text_contents.c_str());
  }
  if (!output_file_flag.empty()) {
    if (llvm::writeToOutput(
            output_file_flag, [&](llvm::raw_ostream& os) -> llvm::Error {
              os.write(binary_contents.data(), binary_contents.size());
              return llvm::Error::success();
            })) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "unable to write module output to %s",
                              output_file_flag.c_str());
    }
  }

  *out_module = std::move(binary_contents);
  return OkStatus();
}

// Evaluates a single function in its own fiber, printing the results to stdout.
Status EvaluateFunction(iree_vm_context_t* context, iree_hal_device_t* device,
                        iree_hal_allocator_t* device_allocator,
                        iree_vm_function_t function,
                        iree_string_view_t function_name) {
  IREE_TRACE_SCOPE();
  iree_allocator_t host_allocator = iree_allocator_system();

  printf("EXEC @%.*s\n", (int)function_name.size, function_name.data);

  // Parse input values from the flags.
  vm::ref<iree_vm_list_t> inputs;
  IREE_RETURN_IF_ERROR(iree_tooling_parse_to_variant_list(
      device_allocator, FLAG_input_list().values, FLAG_input_list().count,
      host_allocator, &inputs));

  // If the function is async add fences so we can invoke it synchronously.
  vm::ref<iree_hal_fence_t> finish_fence;
  IREE_RETURN_IF_ERROR(iree_tooling_append_async_fence_inputs(
      inputs.get(), &function, device, /*wait_fence=*/NULL, &finish_fence));

  // Prepare outputs list to accept the results from the invocation.
  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr, 16,
                                           host_allocator, &outputs));

  // Synchronously invoke the function.
  IREE_RETURN_IF_ERROR(iree_vm_invoke(
      context, function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/nullptr, inputs.get(), outputs.get(), host_allocator));

  // If the function is async we need to wait for it to complete.
  if (finish_fence) {
    IREE_RETURN_IF_ERROR(
        iree_hal_fence_wait(finish_fence.get(), iree_infinite_timeout()));
  }

  // Print outputs.
  if (FLAG_output_list().count == 0) {
    IREE_RETURN_IF_ERROR(
        iree_tooling_variant_list_fprint(
            IREE_SV("result"), outputs.get(),
            (iree_host_size_t)FLAG_output_max_element_count, stdout),
        "printing results");
  } else {
    IREE_RETURN_IF_ERROR(
        iree_tooling_output_variant_list(
            outputs.get(), FLAG_output_list().values, FLAG_output_list().count,
            (iree_host_size_t)FLAG_output_max_element_count, stdout),
        "outputting results");
  }

  return OkStatus();
}

// Evaluates all exported functions within given module.
Status EvaluateFunctions(iree_vm_instance_t* instance,
                         const std::string& default_device_uri,
                         const std::string& flatbuffer_data) {
  IREE_TRACE_SCOPE0("EvaluateFunctions");

  // Load the bytecode module from the flatbuffer data.
  // We do this first so that if we fail validation we know prior to dealing
  // with devices.
  vm::ref<iree_vm_module_t> main_module;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance,
      iree_make_const_byte_span((void*)flatbuffer_data.data(),
                                flatbuffer_data.size()),
      iree_allocator_null(), iree_allocator_system(), &main_module));

  if (!run_flag) {
    // Just wanted verification; return without running.
    main_module.reset();
    return OkStatus();
  }

  // Evaluate all exported functions.
  auto run_function = [&](int ordinal) -> Status {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
                             main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
                             ordinal, &function),
                         "looking up function export %d", ordinal);
    iree_string_view_t function_name = iree_vm_function_name(&function);
    if (iree_string_view_starts_with(function_name,
                                     iree_make_cstring_view("__")) ||
        iree_string_view_find_char(function_name, '$', 0) !=
            IREE_STRING_VIEW_NPOS) {
      // Skip internal or special functions.
      return OkStatus();
    }

    // Create the context we'll use for this (ensuring that we can't interfere
    // with other running evaluations, such as when in a multithreaded test
    // runner).
    vm::ref<iree_vm_context_t> context;
    vm::ref<iree_hal_device_t> device;
    vm::ref<iree_hal_allocator_t> device_allocator;
    IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
        instance, /*user_module_count=*/1, /*user_modules=*/&main_module,
        iree_make_string_view(default_device_uri.data(),
                              default_device_uri.size()),
        iree_allocator_system(), &context, &device, &device_allocator));

    IREE_RETURN_IF_ERROR(iree_hal_begin_profiling_from_flags(device.get()));

    // Invoke the function and print results.
    IREE_RETURN_IF_ERROR(
        EvaluateFunction(context.get(), device.get(), device_allocator.get(),
                         function, function_name),
        "evaluating export function %d", ordinal);

    IREE_RETURN_IF_ERROR(iree_hal_end_profiling_from_flags(device.get()));

    context.reset();
    device_allocator.reset();
    device.reset();
    return OkStatus();
  };

  Status evaluate_status = OkStatus();
  auto module_signature = iree_vm_module_signature(main_module.get());
  for (iree_host_size_t i = 0; i < module_signature.export_function_count;
       ++i) {
    evaluate_status = run_function(i);
    if (!evaluate_status.ok()) {
      break;
    }
  }

  main_module.reset();

  return evaluate_status;
}

// Translates and runs a single LLVM file buffer.
Status EvaluateFile(std::unique_ptr<llvm::MemoryBuffer> file_buffer,
                    mlir::DialectRegistry& registry) {
  IREE_TRACE_SCOPE0("EvaluateFile");

  vm::ref<iree_vm_instance_t> instance;
  IREE_RETURN_IF_ERROR(
      iree_tooling_create_instance(iree_allocator_system(), &instance),
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
    std::string default_device_uri =
        InferDefaultDeviceFromBackend(target_backend);
    IREE_RETURN_IF_ERROR(
        EvaluateFunctions(instance.get(), default_device_uri, flatbuffer_data),
        "Evaluating functions");
  }

  instance.reset();
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
  // This matches the --split-input-file arg to mlir-opt.
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
      fprintf(stderr, "Failure for split at line #%u: %s\n", split_line,
              sub_failure.ToString().c_str());
      if (any_failure.ok()) {
        any_failure = std::move(sub_failure);
      }
    }
  }

  return any_failure;
}

}  // namespace

extern "C" int main(int argc_llvm, char** argv_llvm) {
  IREE_TRACE_SCOPE0("iree-run-mlir");

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerHALTargetBackends();
  mlir::iree_compiler::registerVMTargets();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  // Make sure command line options are registered.
  // Flag options structs (must resolve prior to CLI parsing).
  (void)mlir::iree_compiler::BindingOptions::FromFlags::get();
  (void)mlir::iree_compiler::InputDialectOptions::FromFlags::get();
  (void)mlir::iree_compiler::HighLevelOptimizationOptions::FromFlags::get();
  (void)mlir::iree_compiler::SchedulingOptions::FromFlags::get();
  (void)mlir::iree_compiler::IREE::HAL::TargetOptions::FromFlags::get();
  (void)mlir::iree_compiler::IREE::VM::TargetOptions::FromFlags::get();
  (void)mlir::iree_compiler::IREE::VM::BytecodeTargetOptions::FromFlags::get();

  // Register MLIRContext command-line options like
  // -mlir-print-op-on-diagnostic.
  mlir::registerMLIRContextCLOptions();
  // Register assembly printer command-line options like
  // -mlir-print-op-generic.
  mlir::registerAsmPrinterCLOptions();
  // Register pass manager command-line options like -mlir-print-ir-*.
  mlir::registerPassManagerCLOptions();

  // On Windows InitLLVM re-queries the command line from Windows directly and
  // totally messes up the array.
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc_llvm, argv_llvm);
  llvm::cl::ParseCommandLineOptions(argc_llvm, argv_llvm);

  // Consume all options after the positional filename and pass them to the IREE
  // flag parser.
  std::vector<char*> argv_iree = {argv_llvm[0]};
  for (auto& run_arg : run_args_flag) {
    if (run_arg == "--") continue;
    argv_iree.push_back(const_cast<char*>(run_arg.c_str()));
  }
  int argc_iree = static_cast<int>(argv_iree.size());
  char** argv_iree_ptr = argv_iree.data();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc_iree,
                           &argv_iree_ptr);

  auto status = RunFile(input_file_flag, registry);
  if (!status.ok()) {
    fprintf(stderr, "ERROR running file (%s):\n%s\n", input_file_flag.c_str(),
            status.ToString().c_str());
    return 1;
  }
  return 0;
}

}  // namespace iree
