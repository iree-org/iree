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

// A simple sample demonstrating simple synchronous module loading and VM use.
// This will load an IREE module containing a @simple_mul method that performs
// an element-wise multiplication. It will invoke @simple_mul in the VM, once
// for each available HAL driver linked into the binary.
//
// The synchronous invocation method (Context::Invoke) used here waits until all
// asynchronous HAL work completes before returning. It's still possible get
// overlapped execution by invoking methods from other threads with their own
// FiberState, though it's best to use the asynchronous API instead.
//
// The `iree_module` build rule is used to translate the MLIR to the module
// flatbuffer. Additional HAL backend target support can be defined there.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_replace.h"
#include "iree/base/flatbuffer_util.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/driver_registry.h"
#include "iree/samples/vm/simple_module_test_module.h"
#include "iree/schemas/module_def_generated.h"
#include "iree/vm/fiber_state.h"
#include "iree/vm/instance.h"
#include "iree/vm/sequencer_context.h"

namespace iree {
namespace vm {
namespace samples {
namespace {

using iree::hal::BufferView;

using ModuleFile = FlatBufferFile<ModuleDef>;

struct TestParams {
  // HAL driver to use for the test.
  std::string driver_name;
};

std::ostream& operator<<(std::ostream& os, const TestParams& params) {
  return os << absl::StrReplaceAll(params.driver_name, {{":", "_"}});
}

// Builds a list of tests to run based on the linked in driver modules.
std::vector<TestParams> GetAvailableDriverTestParams() {
  std::vector<TestParams> all_test_params;
  for (const auto& driver_name :
       hal::DriverRegistry::shared_registry()->EnumerateAvailableDrivers()) {
    TestParams test_params;
    test_params.driver_name = driver_name;
    all_test_params.push_back(std::move(test_params));
  }
  return all_test_params;
}

class SimpleModuleTest : public ::testing::Test,
                         public ::testing::WithParamInterface<TestParams> {
 protected:
};

TEST_P(SimpleModuleTest, RunOnce) {
  auto instance = std::make_shared<Instance>();

  // Create driver for this test (based on params) and then get a default
  // device.
  const auto& test_params = GetParam();
  LOG(INFO) << "Creating driver '" << test_params.driver_name << "'...";
  auto driver_or =
      hal::DriverRegistry::shared_registry()->Create(test_params.driver_name);
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
  ASSERT_OK(instance->device_manager()->RegisterDevice(device));
  LOG(INFO) << "Successfully created device '" << device->info().name() << "'";

  // Make a new context and load the precompiled module file (from
  // simple_module_test.mlir) into it.
  LOG(INFO) << "Loading simple_module_test.mlir...";
  SequencerContext context(instance);
  const auto* module_file_toc = simple_module_test_module_create();
  ASSERT_OK_AND_ASSIGN(
      auto module_file,
      ModuleFile::WrapBuffer(ModuleDefIdentifier(),
                             absl::MakeSpan(reinterpret_cast<const uint8_t*>(
                                                module_file_toc->data),
                                            module_file_toc->size)));
  ASSERT_OK_AND_ASSIGN(auto main_module,
                       Module::FromFile(std::move(module_file)));
  ASSERT_OK(context.RegisterModule(std::move(main_module)));
  LOG(INFO) << "Module loaded and context is ready for use";

  // In this sample we just have a single fiber.
  FiberState fiber_state(instance);

  // Allocate buffers that can be mapped on the CPU and that can also be used
  // on the device. Not all devices support this, but the ones we have now do.
  LOG(INFO) << "Creating I/O buffers...";
  constexpr int kElementCount = 4;
  ASSERT_OK_AND_ASSIGN(
      auto arg0_buffer,
      instance->device_manager()->AllocateDeviceVisibleBuffer(
          hal::BufferUsage::kAll, sizeof(float) * kElementCount, {{device}}));
  ASSERT_OK_AND_ASSIGN(
      auto arg1_buffer,
      instance->device_manager()->AllocateDeviceVisibleBuffer(
          hal::BufferUsage::kAll, sizeof(float) * kElementCount, {{device}}));

  // Populate initial values for 4 * 2 = 8.
  ASSERT_OK(arg0_buffer->Fill32(4.0f));
  ASSERT_OK(arg1_buffer->Fill32(2.0f));

  // Call into the @simple_mul function.
  LOG(INFO) << "Calling @simple_mul...";
  std::vector<BufferView> args{
      BufferView{add_ref(arg0_buffer), {kElementCount}, sizeof(float)},
      BufferView{add_ref(arg1_buffer), {kElementCount}, sizeof(float)},
  };
  std::vector<BufferView> results{1};
  ASSERT_OK_AND_ASSIGN(auto simple_mul, context.LookupExport("simple_mul"));
  ASSERT_OK(context.Invoke(&fiber_state, simple_mul, absl::MakeSpan(args),
                           absl::MakeSpan(results)));

  // Read back the results and ensure we got the right values.
  LOG(INFO) << "Reading back results...";
  auto& ret_buffer_view = results[0];
  ASSERT_OK_AND_ASSIGN(
      auto ret_mapping,
      ret_buffer_view.buffer->MapMemory<float>(hal::MemoryAccess::kRead));
  ASSERT_THAT(ret_mapping.contents(),
              ::testing::ElementsAreArray({8.0f, 8.0f, 8.0f, 8.0f}));
  LOG(INFO) << "Results match!";
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, SimpleModuleTest,
                         ::testing::ValuesIn(GetAvailableDriverTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
}  // namespace samples
}  // namespace vm
}  // namespace iree
