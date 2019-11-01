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

#include "iree/testing/benchmark/benchmark_module.h"

#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/driver_registry.h"
#include "iree/rt/context.h"
#include "iree/rt/debug/debug_server_flags.h"
#include "iree/rt/instance.h"
#include "iree/vm/sequencer_module.h"

namespace iree {
namespace {

StatusOr<hal::BufferView> BufferViewFromConstant(const ShapedBuffer& buf_data,
                                                 hal::Allocator* allocator) {
  hal::BufferView view;
  view.element_size = buf_data.element_size();
  view.shape = buf_data.shape();

  size_t allocation_size = view.shape.element_count() * view.element_size;
  ASSIGN_OR_RETURN(
      view.buffer,
      allocator->Allocate(
          hal::MemoryType::kHostLocal | hal::MemoryType::kDeviceVisible,
          hal::BufferUsage::kAll | hal::BufferUsage::kConstant,
          allocation_size));

  RETURN_IF_ERROR(view.buffer->WriteData(0, buf_data.contents().data(),
                                         buf_data.contents().size()));

  return view;
}

StatusOr<std::vector<hal::BufferView>> BufferViewsFromConstants(
    absl::Span<const ShapedBuffer> constants, hal::Allocator* allocator) {
  std::vector<hal::BufferView> views;
  for (const auto& buf_data : constants) {
    ASSIGN_OR_RETURN(auto view, BufferViewFromConstant(buf_data, allocator));
    views.push_back(view);
  }
  return views;
}
}  // namespace

Status RunModuleBenchmark(benchmark::State& state,
                          std::unique_ptr<vm::ModuleFile> main_module_file,
                          absl::string_view main_function_name,
                          absl::string_view driver_name,
                          absl::Span<const ShapedBuffer> arguments_data) {
  ASSIGN_OR_RETURN(auto debug_server, rt::debug::CreateDebugServerFromFlags());
  auto instance = make_ref<rt::Instance>(std::move(debug_server));
  ASSIGN_OR_RETURN(auto driver,
                   hal::DriverRegistry::shared_registry()->Create(driver_name));
  ASSIGN_OR_RETURN(auto device, driver->CreateDefaultDevice());
  RETURN_IF_ERROR(instance->device_manager()->RegisterDevice(device));
  auto policy = make_ref<rt::Policy>();
  auto context = make_ref<rt::Context>(add_ref(instance), std::move(policy));

  ASSIGN_OR_RETURN(auto main_module,
                   vm::SequencerModule::FromFile(std::move(main_module_file)));

  // Register the main module with the context.
  // We could add additional modules (specializations, shared libraries, etc).
  // ModuleFiles are stateless so we could have the same module_file used by
  // multiple contexts simultaneously.
  RETURN_IF_ERROR(context->RegisterModule(add_ref(main_module)));

  rt::Function main_function;
  if (!main_function_name.empty()) {
    // User-specified main function.
    ASSIGN_OR_RETURN(main_function,
                     main_module->LookupFunctionByName(
                         rt::Function::Linkage::kExport, main_function_name));
    // No main function specified; to prevent non-deterministic behavior we
    // require one unless there's exactly one exported function in the module.
  } else if (main_module->signature().export_function_count() == 1) {
    ASSIGN_OR_RETURN(main_function, main_module->LookupFunctionByOrdinal(
                                        rt::Function::Linkage::kExport, 0));
  } else {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "main_function must be specified to disambiguate the function "
              "to run";
  }

  // Call into the main function.
  ASSIGN_OR_RETURN(auto arguments, BufferViewsFromConstants(
                                       arguments_data, device->allocator()));

  for (auto _ : state) {
    ASSIGN_OR_RETURN(auto invocation,
                     rt::Invocation::Create(add_ref(context), main_function,
                                            make_ref<rt::Policy>(), {},
                                            absl::MakeConstSpan(arguments)));
    RETURN_IF_ERROR(invocation->Await(absl::InfiniteFuture()));
  }

  return OkStatus();
}

}  // namespace iree
