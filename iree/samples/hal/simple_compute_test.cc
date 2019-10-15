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

// A simple backend-agnostic compute test for the HAL API.
// This will load an IREE module containing one or more executables and attempt
// to run them against all registered driver backends.
//
// The input file, simple_compute_test.mlir, is as generic as possible to ensure
// we don't need too many variants. This means that it does not use any FFI
// imports requiring runtime support, uses floats exclusively (as that's assumed
// available everywhere), etc.
//
// The `iree_bytecode_module` build rule is used to translate the MLIR to the
// module flatbuffer. Additional target support can be defined there.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_replace.h"
#include "absl/time/time.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/driver_registry.h"
#include "iree/samples/hal/simple_compute_test_module.h"
#include "iree/schemas/module_def_generated.h"

namespace iree {
namespace hal {
namespace samples {
namespace {

using ModuleFile = FlatBufferFile<ModuleDef>;

struct TestParams {
  // HAL driver to use for the test.
  std::string driver_name;
  // Ordinal within the module to execute.
  int executable_ordinal;
  // Name of the executable (just for prettier logging).
  std::string executable_name;
};

std::ostream& operator<<(std::ostream& os, const TestParams& params) {
  return os << absl::StrReplaceAll(params.driver_name, {{":", "_"}}) << "_ex"
            << params.executable_ordinal << "_" << params.executable_name;
}

// Loads the precompiled module file (from simple_compute_test.mlir).
std::unique_ptr<ModuleFile> LoadModuleFile() {
  const auto* file_toc = simple_compute_test_module_create();
  return ModuleFile::WrapBuffer(
             ModuleDefIdentifier(),
             absl::MakeSpan(reinterpret_cast<const uint8_t*>(file_toc->data),
                            file_toc->size))
      .ValueOrDie();
}

// Builds a list of tests to run for each [driver x available executable].
std::vector<TestParams> GetAvailableDriverTestParams() {
  auto module_file = LoadModuleFile();
  auto& executable_table = *module_file->root()->executable_table();
  std::vector<TestParams> all_test_params;
  for (const auto& driver_name :
       DriverRegistry::shared_registry()->EnumerateAvailableDrivers()) {
    int executable_ordinal = 0;
    for (const auto* multi_arch_executable_def :
         *executable_table.multi_arch_executables()) {
      TestParams test_params;
      test_params.driver_name = driver_name;
      test_params.executable_ordinal = executable_ordinal--;
      test_params.executable_name =
          std::string(WrapString(multi_arch_executable_def->name()));
      all_test_params.push_back(std::move(test_params));
    }
  }
  return all_test_params;
}

class SimpleComputeTest : public ::testing::Test,
                          public ::testing::WithParamInterface<TestParams> {
 protected:
  virtual void SetUp() { module_file_ = LoadModuleFile(); }

  std::unique_ptr<ModuleFile> module_file_;
};

TEST_P(SimpleComputeTest, RunOnce) {
  const auto& test_params = GetParam();

  // Create driver for this test (based on params) and then get a default
  // device.
  LOG(INFO) << "Creating driver '" << test_params.driver_name << "'...";
  auto driver_or =
      DriverRegistry::shared_registry()->Create(test_params.driver_name);
  if (IsUnavailable(driver_or.status())) {
    LOG(WARNING) << "Skipping test as driver is unavailable: "
                 << driver_or.status();
    GTEST_SKIP();
    return;
  }
  ASSERT_OK_AND_ASSIGN(auto driver, driver_or);
  ASSERT_OK_AND_ASSIGN(auto available_devices,
                       driver->EnumerateAvailableDevices());
  for (const auto& device_info : available_devices) {
    LOG(INFO) << "  Device: " << device_info.name();
  }
  LOG(INFO) << "Creating default device...";
  ASSERT_OK_AND_ASSIGN(auto device, driver->CreateDefaultDevice());
  LOG(INFO) << "Successfully created device '" << device->info().name() << "'";

  // Attempt to compile the appropriate executable. This may fail if there's no
  // executable available in the input file that the driver can load.
  auto executable_cache = device->CreateExecutableCache();
  auto& executable_table = *module_file_->root()->executable_table();
  auto multi_arch_executable_def =
      executable_table.multi_arch_executables()->Get(
          test_params.executable_ordinal);
  ref_ptr<Executable> executable;
  for (auto executable_def : *multi_arch_executable_def->executables()) {
    if (!executable_cache->CanPrepareFormat(executable_def->format())) {
      continue;
    }
    ExecutableSpec spec;
    spec.format = executable_def->format();
    spec.executable_data = *executable_def->contents();
    ASSERT_OK_AND_ASSIGN(executable,
                         executable_cache->PrepareExecutable(
                             ExecutableCachingMode::kDefault, spec));
    break;
  }
  ASSERT_NE(executable, nullptr)
      << "No executable found that has a supported format for driver "
      << test_params.driver_name;

  // Create I/O buffers.
  ASSERT_OK_AND_ASSIGN(auto arg0_buffer,
                       device->allocator()->Allocate(
                           MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                           BufferUsage::kAll, 4 * sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto arg1_buffer,
                       device->allocator()->Allocate(
                           MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                           BufferUsage::kAll, 4 * sizeof(float)));
  ASSERT_OK_AND_ASSIGN(auto ret0_buffer,
                       device->allocator()->Allocate(
                           MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                           BufferUsage::kAll, 4 * sizeof(float)));

  // Populate initial values for 4 * 2 = 8.
  // We scribble into the result buffer so that it's easy to ensure it's
  // overwritten.
  ASSERT_OK(arg0_buffer->Fill32(4.0f));
  ASSERT_OK(arg1_buffer->Fill32(2.0f));
  ASSERT_OK(ret0_buffer->Fill32(99999.0f));

  // Record the command buffer that dispatches the executable.
  ASSERT_OK_AND_ASSIGN(
      auto cmd, device->CreateCommandBuffer(
                    CommandBufferMode::kOneShot,
                    CommandCategory::kTransfer | CommandCategory::kDispatch));
  ASSERT_OK(cmd->Begin());
  DispatchRequest dispatch_request;
  dispatch_request.executable = executable.get();
  dispatch_request.entry_point = 0;
  dispatch_request.workload[0] = 4;
  dispatch_request.workload[1] = 1;
  dispatch_request.workload[2] = 1;
  BufferBinding bindings[3];
  bindings[0].buffer = arg0_buffer.get();
  bindings[0].access = MemoryAccess::kRead;
  bindings[0].element_size = sizeof(float);
  bindings[0].shape = {4};
  bindings[1].buffer = arg1_buffer.get();
  bindings[1].access = MemoryAccess::kRead;
  bindings[1].element_size = sizeof(float);
  bindings[1].shape = {4};
  bindings[2].buffer = ret0_buffer.get();
  bindings[2].access = MemoryAccess::kDiscardWrite;
  bindings[2].element_size = sizeof(float);
  bindings[2].shape = {4};
  dispatch_request.bindings = bindings;
  ASSERT_OK(cmd->Dispatch(dispatch_request));
  ASSERT_OK(cmd->End());

  // Schedule and wait for completion.
  ASSERT_FALSE(device->dispatch_queues().empty());
  CommandQueue* queue = device->dispatch_queues().front();
  ASSERT_OK_AND_ASSIGN(auto fence, device->CreateFence(0u));
  ASSERT_OK(
      queue->Submit(SubmissionBatch{{}, {cmd.get()}, {}}, {fence.get(), 1u}));
  ASSERT_OK(device->WaitAllFences({{fence.get(), 1u}}, absl::InfiniteFuture()));

  // Read back the results.
  ASSERT_OK_AND_ASSIGN(auto ret0_mapping,
                       ret0_buffer->MapMemory<float>(MemoryAccess::kRead));
  EXPECT_THAT(ret0_mapping.contents(),
              ::testing::ElementsAreArray({8.0f, 8.0f, 8.0f, 8.0f}));
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, SimpleComputeTest,
                         ::testing::ValuesIn(GetAvailableDriverTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
}  // namespace samples
}  // namespace hal
}  // namespace iree
