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
// precompiled IREE modules use //third_party/iree/tools:iree-run-module.
//
// By default all exported functions in the module will be run in order.
// All input values, provided via -input-values, will be passed to the
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
#include "absl/strings/numbers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/init.h"
#include "iree/base/shaped_buffer.h"
#include "iree/base/shaped_buffer_string_util.h"
#include "iree/base/signature_mangle.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Translation/IREEVM.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm2/api.h"
#include "iree/vm2/bytecode_module.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
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

static llvm::cl::opt<bool> export_all_flag{
    "export-all",
    llvm::cl::desc("Adds iree.module.export to all functions"),
    llvm::cl::init(true),
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

static llvm::cl::list<std::string> input_values_flag{
    "input-value",
    llvm::cl::desc("Input shapes and optional values"),
    llvm::cl::ZeroOrMore,
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
StatusOr<std::vector<std::string>> GetTargetBackends() {
  auto target_backends =
      mlir::iree_compiler::IREE::HAL::getExecutableTargetOptionsFromFlags()
          .targets;
  if (target_backends.empty()) {
    iree_string_view_t* driver_names = nullptr;
    iree_host_size_t driver_count = 0;
    RETURN_IF_ERROR(
        FromApiStatus(iree_hal_driver_registry_query_available_drivers(
                          IREE_ALLOCATOR_SYSTEM, &driver_names, &driver_count),
                      IREE_LOC));
    for (int i = 0; i < driver_count; ++i) {
      target_backends.push_back(
          std::string(driver_names[i].data, driver_names[i].size));
    }
    iree_allocator_free(IREE_ALLOCATOR_SYSTEM, driver_names);
  }
  return target_backends;
}

// Prepares a module for evaluation by running MLIR import and IREE translation.
// Returns the serialized flatbuffer data.
StatusOr<std::string> PrepareModule(
    std::string target_backend,
    std::unique_ptr<llvm::MemoryBuffer> file_buffer) {
  mlir::MLIRContext context;

  // Parse input MLIR module.
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file_buffer), llvm::SMLoc());
  mlir::OwningModuleRef mlir_module =
      mlir::parseSourceFile(source_mgr, &context);

  if (export_all_flag) {
    for (auto function : mlir_module->getOps<mlir::FuncOp>()) {
      function.setAttr("iree.module.export", mlir::UnitAttr::get(&context));
    }
  }

  // Translate from MLIR to IREE bytecode.
  LOG(INFO) << "Compiling for target backend '" << target_backend << "'...";
  auto executable_options =
      mlir::iree_compiler::IREE::HAL::getExecutableTargetOptionsFromFlags();
  executable_options.targets = {std::move(target_backend)};
  mlir::PassManager pass_manager(mlir_module->getContext());
  mlir::iree_compiler::IREE::Flow::buildFlowTransformPassPipeline(pass_manager);
  mlir::iree_compiler::IREE::HAL::buildHALTransformPassPipeline(
      pass_manager, executable_options);
  mlir::iree_compiler::IREE::VM::buildVMTransformPassPipeline(pass_manager);
  if (failed(pass_manager.run(mlir_module.get()))) {
    return InternalErrorBuilder(IREE_LOC)
           << "Conversion from source -> vm failed";
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
    return InternalErrorBuilder(IREE_LOC)
           << "Serialization to flatbuffer bytecode (binary) failed";
  }
  binary_output.flush();

  // Print the annotated MLIR and flatbuffer; easiest way right now is to just
  // do it all again.
  if (print_annotated_mlir_flag) {
    bytecode_options.outputFormat =
        mlir::iree_compiler::IREE::VM::BytecodeOutputFormat::kMlirText;
    std::string text_contents;
    llvm::raw_string_ostream text_output(text_contents);
    if (failed(mlir::iree_compiler::IREE::VM::translateModuleToBytecode(
            mlir_module.get(), bytecode_options, text_output))) {
      return InternalErrorBuilder(IREE_LOC)
             << "Serialization to annotated MLIR (text) failed";
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
      return InternalErrorBuilder(IREE_LOC)
             << "Serialization to flatbuffer bytecode (text) failed";
    }
    text_output.flush();
    std::cerr << text_contents << std::endl;
  }

  return binary_contents;
}

// Parses a list of input shapes and values from a string of newline-separated
// inputs. Expects the contents to have one value per line with each value
// listed as
//   [shape]xtype=[value]
// Example:
//   4x4xi8=0,1,2,3
StatusOr<iree_vm_variant_list_t*> ParseInputsFromFlags(
    iree_vm_function_t function, iree_hal_allocator_t* allocator) {
  iree_vm_variant_list_t* inputs = nullptr;
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_variant_list_alloc(input_values_flag.size(),
                                               IREE_ALLOCATOR_SYSTEM, &inputs),
                    IREE_LOC));
  for (const auto& input_value : input_values_flag) {
    ASSIGN_OR_RETURN(auto shaped_buffer,
                     ParseShapedBufferFromString(input_value),
                     _ << "Parsing input value '" << input_value << "'");
    iree_hal_buffer_t* input_buffer = nullptr;
    // TODO(benvanik): combined function for linear to optimal upload.
    iree_device_size_t allocation_size =
        shaped_buffer.shape().element_count() * shaped_buffer.element_size();
    RETURN_IF_ERROR(FromApiStatus(
        iree_hal_allocator_allocate_buffer(
            allocator,
            static_cast<iree_hal_memory_type_t>(
                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
            static_cast<iree_hal_buffer_usage_t>(
                IREE_HAL_BUFFER_USAGE_ALL | IREE_HAL_BUFFER_USAGE_CONSTANT),
            allocation_size, &input_buffer),
        IREE_LOC))
        << "Allocating input buffer";
    RETURN_IF_ERROR(FromApiStatus(
        iree_hal_buffer_write_data(input_buffer, 0,
                                   shaped_buffer.contents().data(),
                                   shaped_buffer.contents().size()),
        IREE_LOC))
        << "Populating input buffer contents";
    auto input_buffer_ref = iree_hal_buffer_move_ref(input_buffer);
    RETURN_IF_ERROR(FromApiStatus(
        iree_vm_variant_list_append_ref_move(inputs, &input_buffer_ref),
        IREE_LOC));
  }
  return inputs;
}

// Outputs all results from the function to stdout in IREE BufferView format.
Status OutputFunctionResults(iree_vm_function_t function,
                             iree_vm_variant_list_t* outputs) {
  iree_string_view_t sig_fv =
      iree_vm_function_reflection_attr(&function, iree_make_cstring_view("fv"));
  if (iree_string_view_compare(sig_fv, iree_make_cstring_view("1")) != 0) {
    return UnimplementedErrorBuilder(IREE_LOC) << "Unsupported function ABI";
  }

  iree_string_view_t sig_f =
      iree_vm_function_reflection_attr(&function, iree_make_cstring_view("f"));
  RawSignatureParser sig_parser;
  absl::InlinedVector<RawSignatureParser::Description, 4> output_descs;
  sig_parser.VisitResults(absl::string_view{sig_f.data, sig_f.size},
                          [&](const RawSignatureParser::Description& desc) {
                            output_descs.push_back(desc);
                          });
  if (output_descs.size() != iree_vm_variant_list_size(outputs)) {
    return FailedPreconditionErrorBuilder(IREE_LOC)
           << "Result signature mismatch; expected " << output_descs.size()
           << " results but VM returned " << iree_vm_variant_list_size(outputs);
  }

  for (int i = 0; i < iree_vm_variant_list_size(outputs); ++i) {
    iree_vm_variant_t* variant = iree_vm_variant_list_get(outputs, i);
    auto* buffer = iree_hal_buffer_deref(&variant->ref);

    const auto& desc = output_descs[i];
    std::string desc_str;
    desc.ToString(desc_str);
    auto print_mode = BufferDataPrintMode::kFloatingPoint;
    int8_t element_size = 4;
    Shape shape;
    switch (desc.type) {
      case RawSignatureParser::Type::kBuffer:
        switch (desc.buffer.scalar_type) {
          case AbiConstants::ScalarType::kIeeeFloat16:
          case AbiConstants::ScalarType::kIeeeFloat32:
          case AbiConstants::ScalarType::kIeeeFloat64:
            print_mode = BufferDataPrintMode::kFloatingPoint;
            break;
          case AbiConstants::ScalarType::kSint8:
          case AbiConstants::ScalarType::kSint16:
          case AbiConstants::ScalarType::kSint32:
          case AbiConstants::ScalarType::kSint64:
            print_mode = BufferDataPrintMode::kSignedInteger;
            break;
          case AbiConstants::ScalarType::kUint8:
          case AbiConstants::ScalarType::kUint16:
          case AbiConstants::ScalarType::kUint32:
          case AbiConstants::ScalarType::kUint64:
            print_mode = BufferDataPrintMode::kUnsignedInteger;
            break;
          default:
            print_mode = BufferDataPrintMode::kBinary;
            break;
        }
        element_size = AbiConstants::kScalarTypeSize[static_cast<unsigned>(
            desc.buffer.scalar_type)];
        shape = Shape{desc.dims};
        break;
      default:
        return UnimplementedErrorBuilder(IREE_LOC)
               << "Unsupported signature type: " << desc_str;
    }

    // TODO(benvanik): debug string C API: buffer->DebugString();
    LOG(INFO) << "result[" << i << "]: " << desc_str;
    iree_hal_mapped_memory_t mapped_memory;
    RETURN_IF_ERROR(
        FromApiStatus(iree_hal_buffer_map(buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                          0, IREE_WHOLE_BUFFER, &mapped_memory),
                      IREE_LOC));
    auto contents = absl::MakeConstSpan(mapped_memory.contents.data,
                                        mapped_memory.contents.data_length);
    ShapedBuffer shaped_buffer(
        element_size, shape,
        std::vector<uint8_t>(contents.begin(), contents.end()));
    ASSIGN_OR_RETURN(auto result_str, PrintShapedBufferToString(
                                          shaped_buffer, print_mode, 1024));
    iree_hal_buffer_unmap(buffer, &mapped_memory);
    std::cout << result_str << "\n";
  }

  return OkStatus();
}

// Evaluates a single function in its own fiber, printing the results to stdout.
Status EvaluateFunction(iree_vm_context_t* context,
                        iree_hal_allocator_t* allocator,
                        iree_vm_function_t function) {
  auto function_name = iree_vm_function_name(&function);
  std::cout << "EXEC @"
            << absl::string_view(function_name.data, function_name.size)
            << std::endl;

  // Parse inputs and create the required input buffers.
  ASSIGN_OR_RETURN(auto* input_list, ParseInputsFromFlags(function, allocator));

  // Prepare outputs list to accept the results from the invocation.
  iree_vm_variant_list_t* output_list = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_variant_list_alloc(16, IREE_ALLOCATOR_SYSTEM, &output_list),
      IREE_LOC));

  // Synchronously invoke the function.
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_invoke(context, function, /*policy=*/nullptr, input_list,
                     output_list, IREE_ALLOCATOR_SYSTEM),
      IREE_LOC));

  iree_vm_variant_list_free(input_list);

  // Print outputs.
  RETURN_IF_ERROR(OutputFunctionResults(function, output_list));
  iree_vm_variant_list_free(output_list);

  return OkStatus();
}

// Evaluates all exported functions within given module.
Status EvaluateFunctions(iree_vm_instance_t* instance,
                         absl::string_view target_backend,
                         const std::string& flatbuffer_data) {
  LOG(INFO) << "Evaluating all functions in module for backend '"
            << target_backend << "'...";

  // Load the bytecode module from the flatbuffer data.
  // We do this first so that if we fail validation we know prior to dealing
  // with devices.
  iree_vm_module_t* bytecode_module = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_bytecode_module_create(
          iree_const_byte_span_t{
              reinterpret_cast<const uint8_t*>(flatbuffer_data.c_str()),
              flatbuffer_data.size()},
          IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &bytecode_module),
      IREE_LOC))
      << "Deserializing flatbuffer module";

  if (!run_flag) {
    // Just wanted verification; return without running.
    iree_vm_module_release(bytecode_module);
    return OkStatus();
  }

  // Create the driver/device as defined by the test and setup the HAL module.
  LOG(INFO) << "Creating target backend driver '" << target_backend << "'...";
  iree_hal_driver_t* driver = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_driver_registry_create_driver(
          iree_string_view_t{target_backend.data(), target_backend.size()},
          IREE_ALLOCATOR_SYSTEM, &driver),
      IREE_LOC))
      << "Creating driver for " << target_backend;
  iree_hal_device_t* device = nullptr;
  RETURN_IF_ERROR(FromApiStatus(iree_hal_driver_create_default_device(
                                    driver, IREE_ALLOCATOR_SYSTEM, &device),
                                IREE_LOC))
      << "Creating default device for " << target_backend;
  iree_vm_module_t* hal_module = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_hal_module_create(device, IREE_ALLOCATOR_SYSTEM, &hal_module),
      IREE_LOC))
      << "Creating HAL module";
  iree_hal_driver_release(driver);

  // Evaluate all exported functions.
  auto run_function = [&](int ordinal) -> Status {
    // Create the context we'll use for this (ensuring that we can't interfere
    // with other running evaluations, such as when in a multithreaded test
    // runner).
    iree_vm_context_t* context = nullptr;
    std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
    RETURN_IF_ERROR(FromApiStatus(iree_vm_context_create_with_modules(
                                      instance, modules.data(), modules.size(),
                                      IREE_ALLOCATOR_SYSTEM, &context),
                                  IREE_LOC))
        << "Creating context";

    iree_vm_function_t function;
    RETURN_IF_ERROR(
        FromApiStatus(iree_vm_module_lookup_function_by_ordinal(
                          bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                          ordinal, &function),
                      IREE_LOC))
        << "Looking up function export " << ordinal;

    // Invoke the function and print results.
    RETURN_IF_ERROR(
        EvaluateFunction(context, iree_hal_device_allocator(device), function))
        << "Evaluating export function " << ordinal;

    iree_vm_context_release(context);
    return OkStatus();
  };

  Status evaluate_status = OkStatus();
  auto module_signature = iree_vm_module_signature(bytecode_module);
  for (int i = 0; i < module_signature.export_function_count; ++i) {
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
Status EvaluateFile(std::unique_ptr<llvm::MemoryBuffer> file_buffer) {
  // TODO(benvanik): move to instance-based registration.
  RETURN_IF_ERROR(FromApiStatus(iree_hal_module_register_types(), IREE_LOC))
      << "Registering HAL types";

  iree_vm_instance_t* instance = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance), IREE_LOC))
      << "Create instance";

  ASSIGN_OR_RETURN(auto target_backends, GetTargetBackends());
  for (const auto& target_backend : target_backends) {
    // Prepare the module for execution and evaluate it.
    auto cloned_file_buffer = llvm::MemoryBuffer::getMemBufferCopy(
        file_buffer->getBuffer(), file_buffer->getBufferIdentifier());
    ASSIGN_OR_RETURN(
        auto flatbuffer_data,
        PrepareModule(target_backend + '*', std::move(cloned_file_buffer)),
        _ << "Translating module");
    RETURN_IF_ERROR(EvaluateFunctions(
        instance, BackendToDriverName(target_backend), flatbuffer_data))
        << "Evaluating functions";
  }

  iree_vm_instance_release(instance);
  return OkStatus();
}

// Runs the given .mlir file based on the current flags.
Status RunFile(const std::string& mlir_filename) {
  // Load input file/from stdin.
  std::string error_message;
  auto file = mlir::openInputFile(mlir_filename, &error_message);
  if (!file) {
    return NotFoundErrorBuilder(IREE_LOC)
           << "Unable to open input file " << mlir_filename << ": "
           << error_message;
  }

  if (!split_input_file_flag) {
    // Use entire buffer as a single module.
    return EvaluateFile(std::move(file));
  }

  // Split the buffer into separate modules and evaluate independently.
  // This matches the -split-input-file arg to mlir-opt.
  const char kSplitMarker[] = "// -----\n";
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
    auto sub_failure = EvaluateFile(std::move(sub_buffer));
    if (!sub_failure.ok()) {
      LOG(ERROR) << sub_failure;
      if (any_failure.ok()) {
        any_failure = std::move(sub_failure);
      }
    }
  }

  return any_failure;
}

}  // namespace

extern "C" int main(int argc, char** argv) {
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

  mlir::registerPassManagerCLOptions();
  llvm::InitLLVM init_llvm(argc_llvm, argv_llvm);
  llvm::cl::ParseCommandLineOptions(argc_llvm, argv_llvm);

  for (auto& run_arg : run_args_flag) {
    argv_absl.push_back(const_cast<char*>(run_arg.c_str()));
  }
  argc_absl += run_args_flag.size();
  char** argv_absl_ptr = argv_absl.data();
  InitializeEnvironment(&argc_absl, &argv_absl_ptr);

  auto status = RunFile(input_file_flag);
  if (!status.ok()) {
    std::cerr << "ERROR running file (" << input_file_flag << "): " << status
              << "\n";
    return 1;
  }
  return 0;
}

}  // namespace iree
