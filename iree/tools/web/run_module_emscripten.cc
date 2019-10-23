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

#include <emscripten.h>
#include <emscripten/bind.h>

#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/init.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/buffer_view_string_util.h"
#include "iree/hal/driver_registry.h"
#include "iree/rt/context.h"
#include "iree/rt/instance.h"
#include "iree/schemas/module_def_generated.h"
#include "iree/vm/sequencer_module.h"

namespace iree {

// Parses a list of input shapes and values from a string of newline-separated
// inputs. Expects the contents to have one value per line with each value
// listed as
//   [shape]xtype=[value]
// Example:
//   4x4xi8=0,1,2,3
StatusOr<std::vector<hal::BufferView>> ParseInputs(
    absl::string_view inputs_string, hal::Allocator* allocator) {
  std::string input_lines = absl::StrReplaceAll(inputs_string, {{"\\n", "\n"}});
  std::vector<hal::BufferView> input_buffer_views;
  for (const auto& input_line :
       absl::StrSplit(input_lines, '\n', absl::SkipWhitespace())) {
    ASSIGN_OR_RETURN(auto input_buffer_view,
                     hal::ParseBufferViewFromString(input_line, allocator));
    input_buffer_views.push_back(input_buffer_view);
  }
  return input_buffer_views;
}

// Runs an IREE module with the provided inputs and returns its outputs.
StatusOr<std::string> RunIreeModule(std::string module_file_data,
                                    absl::string_view inputs_string) {
  auto instance = make_ref<rt::Instance>();

  // Create driver and device.
  ASSIGN_OR_RETURN(auto driver, hal::DriverRegistry::shared_registry()->Create(
                                    "interpreter"));
  ASSIGN_OR_RETURN(auto device, driver->CreateDefaultDevice());
  RETURN_IF_ERROR(instance->device_manager()->RegisterDevice(device));

  auto policy = make_ref<rt::Policy>();
  auto context = make_ref<rt::Context>(add_ref(instance), std::move(policy));

  // Load main module FlatBuffer.
  ASSIGN_OR_RETURN(auto main_module_file,
                   FlatBufferFile<ModuleDef>::FromString(ModuleDefIdentifier(),
                                                         module_file_data));
  ASSIGN_OR_RETURN(auto main_module,
                   vm::SequencerModule::FromFile(std::move(main_module_file)));

  // Register the main module with the context.
  RETURN_IF_ERROR(context->RegisterModule(add_ref(main_module)));

  // Setup arguments and storage for results.
  // TODO(scotttodd): Receive main function name from JS.
  ASSIGN_OR_RETURN(auto main_function,
                   main_module->LookupFunctionByName(
                       rt::Function::Linkage::kExport, "main"));

  ASSIGN_OR_RETURN(auto arguments,
                   ParseInputs(inputs_string, device->allocator()));

  // Call into the main function.
  ASSIGN_OR_RETURN(auto invocation,
                   rt::Invocation::Create(add_ref(context), main_function,
                                          make_ref<rt::Policy>(), {},
                                          absl::MakeConstSpan(arguments)));

  // Wait until invocation completes.
  // TODO(scotttodd): make this an async callback.
  RETURN_IF_ERROR(invocation->Await(absl::InfiniteFuture()));
  ASSIGN_OR_RETURN(auto results, invocation->ConsumeResults());

  // Dump all results to stdout.
  // TODO(scotttodd): Receive output types / print mode from JS.
  // TODO(scotttodd): Return list of outputs instead of just the first (proto?)
  for (int i = 0; i < results.size(); ++i) {
    const auto& result = results[i];
    auto print_mode = hal::BufferViewPrintMode::kFloatingPoint;
    ASSIGN_OR_RETURN(auto result_str,
                     PrintBufferViewToString(result, print_mode, 1024));
    const auto& buffer = result.buffer;
    if (!buffer) {
      return InternalErrorBuilder(IREE_LOC)
             << "result[" << i << "] unexpectedly has no buffer";
    }

    return result_str;
  }

  return InternalErrorBuilder(IREE_LOC) << "Received no results";
}

std::string RunIreeModuleEntry(std::string module_file_data,
                               std::string inputs_string) {
  // TODO(scotttodd): optimize, minimize copies
  // https://groups.google.com/d/msg/emscripten-discuss/CMfYljLWMvY/Di52WB2QAgAJ
  auto result_or = RunIreeModule(std::move(module_file_data), inputs_string);
  if (!result_or.ok()) {
    return "Error: " + result_or.status().ToString();
  } else {
    return result_or.ValueOrDie();
  }
}

EMSCRIPTEN_BINDINGS(iree) {
  emscripten::function("runIreeModule", &RunIreeModuleEntry);
}

extern "C" int main(int argc, char** argv) {
  InitializeEnvironment(&argc, &argv);
  return 0;
}

}  // namespace iree
