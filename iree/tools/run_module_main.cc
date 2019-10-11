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

#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "iree/base/file_io.h"
#include "iree/base/file_path.h"
#include "iree/base/init.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view_string_util.h"
#include "iree/hal/driver_registry.h"
#include "iree/schemas/module_def_generated.h"
#include "iree/vm/bytecode_printer.h"
#include "iree/vm/bytecode_tables_sequencer.h"
#include "iree/vm/debug/debug_server_flags.h"
#include "iree/vm/fiber_state.h"
#include "iree/vm/function.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"
#include "iree/vm/module_printer.h"
#include "iree/vm/sequencer_context.h"

ABSL_FLAG(std::string, main_module, "", "Main module with entry point.");
ABSL_FLAG(std::string, main_function, "",
          "Function within the main module to execute.");

ABSL_FLAG(bool, print_source_info, false,
          "Prints source map information in bytecode output.");

ABSL_FLAG(std::string, input_values, "", "Input shapes and optional values.");
ABSL_FLAG(std::string, input_file, "",
          "Input shapes and optional values serialized in a file.");

ABSL_FLAG(std::string, output_types, "",
          "Output data types (comma delimited list of b/i/u/f for "
          "binary/signed int/unsigned int/float).");

namespace iree {
namespace vm {
namespace {

using ::iree::hal::BufferView;

// Parses a list of input shapes and values from a string of newline-separated
// inputs. Expects the contents to have one value per line with each value
// listed as
//   [shape]xtype=[value]
// Example:
//   4x4xi8=0,1,2,3
StatusOr<std::vector<BufferView>> ParseInputsFromFlags(
    hal::Allocator* allocator) {
  std::string file_contents;
  if (!absl::GetFlag(FLAGS_input_values).empty()) {
    file_contents =
        absl::StrReplaceAll(absl::GetFlag(FLAGS_input_values), {{"\\n", "\n"}});
  } else if (!absl::GetFlag(FLAGS_input_file).empty()) {
    ASSIGN_OR_RETURN(file_contents,
                     file_io::GetFileContents(absl::GetFlag(FLAGS_input_file)));
  }
  std::vector<BufferView> inputs;
  for (const auto& line :
       absl::StrSplit(file_contents, '\n', absl::SkipWhitespace())) {
    ASSIGN_OR_RETURN(auto input,
                     hal::ParseBufferViewFromString(line, allocator));
    inputs.push_back(input);
  }
  return inputs;
}

}  // namespace

Status Run() {
  ASSIGN_OR_RETURN(auto debug_server, debug::CreateDebugServerFromFlags());
  auto instance = std::make_shared<Instance>(std::move(debug_server));
  ASSIGN_OR_RETURN(auto driver, hal::DriverRegistry::shared_registry()->Create(
                                    "interpreter"));
  ASSIGN_OR_RETURN(auto device, driver->CreateDefaultDevice());
  RETURN_IF_ERROR(instance->device_manager()->RegisterDevice(device));
  SequencerContext context(instance);

  // Load main module.
  ASSIGN_OR_RETURN(
      auto main_module_file,
      ModuleFile::LoadFile(ModuleDefIdentifier(),
                           absl::GetFlag(FLAGS_main_module)),
      _ << "while loading module file " << absl::GetFlag(FLAGS_main_module));
  ASSIGN_OR_RETURN(auto main_module,
                   Module::FromFile(std::move(main_module_file)));

  // Add native functions for use by the module.
  RETURN_IF_ERROR(context.RegisterNativeFunction(
      "fabsf",
      [](Stack* stack, absl::Span<const BufferView> args,
         absl::Span<BufferView> results) -> Status {
        // TODO(benvanik): example native functions.
        LOG(INFO) << "fabsf";
        return OkStatus();
      }));

  // Register the main module with the context.
  // We could add additional modules (specializations, shared libraries, etc).
  // ModuleFioles are stateless so we could have the same module_file used by
  // multiple contexts simultaneously.
  auto* main_module_ptr = main_module.get();
  RETURN_IF_ERROR(context.RegisterModule(std::move(main_module)));

  // Dump the registered modules.
  PrintModuleFlagBitfield print_flags = PrintModuleFlag::kNone;
  if (absl::GetFlag(FLAGS_print_source_info)) {
    print_flags |= PrintModuleFlag::kIncludeSourceMapping;
  }
  for (const auto& module : context.modules()) {
    RETURN_IF_ERROR(PrintModuleToStream(sequencer_opcode_table(), *module,
                                        print_flags, &std::cout));
  }

  // Setup a new fiber.
  FiberState fiber_state(instance);

  // Setup arguments and storage for results.
  Function main_function;
  if (!absl::GetFlag(FLAGS_main_function).empty()) {
    // User-specified main function.
    ASSIGN_OR_RETURN(main_function,
                     context.LookupExport(absl::GetFlag(FLAGS_main_function)));
  } else {
    // No main function specified; to prevent non-deterministic behavior we
    // require one unless there's exactly one exported function in the module.
    auto* exports = main_module_ptr->function_table().def().exports();
    if (exports && exports->size() == 1) {
      ASSIGN_OR_RETURN(
          main_function,
          main_module_ptr->function_table().LookupFunction(exports->Get(0)));
    } else {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "--main_function= must be specified to disambiguate the "
                "function to run";
    }
  }
  ASSIGN_OR_RETURN(std::vector<BufferView> args,
                   ParseInputsFromFlags(device->allocator()));
  std::vector<BufferView> results;
  results.resize(main_function.result_count());

  // Call into the main function.
  RETURN_IF_ERROR(context.Invoke(&fiber_state, main_function,
                                 absl::MakeSpan(args),
                                 absl::MakeSpan(results)));

  // Dump all results to stdout.
  std::vector<std::string> output_types =
      absl::StrSplit(absl::GetFlag(FLAGS_output_types), absl::ByAnyChar(", "),
                     absl::SkipWhitespace());
  if (!output_types.empty() && output_types.size() != results.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "--output_types= specified but has " << output_types.size()
           << " types when the function returns " << results.size();
  }
  for (int i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    auto print_mode = hal::BufferViewPrintMode::kFloatingPoint;
    if (!output_types.empty()) {
      ASSIGN_OR_RETURN(print_mode,
                       hal::ParseBufferViewPrintMode(output_types[i]));
    }
    ASSIGN_OR_RETURN(auto result_str,
                     PrintBufferViewToString(result, print_mode, 1024));
    const auto& buffer = result.buffer;
    if (!buffer) {
      return InternalErrorBuilder(IREE_LOC)
             << "result[" << i << "] unexpectedly has no buffer";
    }
    LOG(INFO) << "result[" << i << "]: " << buffer->DebugString();
    std::cout << result_str << "\n";
  }

  return OkStatus();
}

extern "C" int main(int argc, char** argv) {
  InitializeEnvironment(&argc, &argv);
  CHECK_OK(Run());
  return 0;
}

}  // namespace vm
}  // namespace iree
