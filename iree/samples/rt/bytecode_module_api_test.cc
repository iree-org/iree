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

#include "absl/strings/str_replace.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

// C API:
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/rt/api.h"
#include "iree/vm/api.h"

// Only temporary, used for test device registration:
#include "iree/hal/driver_registry.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/rt/simple_module_test_bytecode_module.h"

namespace iree {
namespace rt {
namespace samples {
namespace {

#define ASSERT_API_OK(expr) ASSERT_EQ(IREE_STATUS_OK, (expr))

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

class BytecodeModuleApiTest : public ::testing::Test,
                              public ::testing::WithParamInterface<TestParams> {
 protected:
};

TEST_P(BytecodeModuleApiTest, RunOnce) {
  iree_rt_instance_t* instance = nullptr;
  ASSERT_API_OK(iree_rt_instance_create(IREE_ALLOCATOR_DEFAULT, &instance));

  // TEMPORARY: until policies and placement are performed with manually
  // register drivers via a magic function.
  const auto& driver_name = GetParam().driver_name;
  LOG(INFO) << "Creating driver '" << driver_name << "'...";
  ASSERT_API_OK(iree_rt_instance_register_driver_ex(
      instance, iree_string_view_t{driver_name.data(), driver_name.size()}));

  // Allocate a context that will hold the module state across invocations.
  iree_rt_policy_t* dummy_policy = nullptr;
  ASSERT_API_OK(iree_rt_policy_create(IREE_ALLOCATOR_DEFAULT, &dummy_policy));
  iree_rt_context_t* context = nullptr;
  ASSERT_API_OK(iree_rt_context_create(instance, dummy_policy,
                                       IREE_ALLOCATOR_DEFAULT, &context));
  iree_rt_policy_release(dummy_policy);

  // Load bytecode module from the embedded data.
  LOG(INFO) << "Loading simple_module_test.mlir...";
  const auto* module_file_toc = simple_module_test_bytecode_module_create();
  iree_rt_module_t* bytecode_module = nullptr;
  ASSERT_API_OK(iree_vm_bytecode_module_create_from_buffer(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      nullptr, nullptr, IREE_ALLOCATOR_DEFAULT, &bytecode_module));

  // Register modules that we want to be able to use in the context.
  std::vector<iree_rt_module_t*> modules;
  modules.push_back(bytecode_module);
  ASSERT_API_OK(
      iree_rt_context_register_modules(context, &modules[0], modules.size()));
  iree_rt_module_release(bytecode_module);
  LOG(INFO) << "Module loaded and context is ready for use";

  // Lookup the entry point function.
  iree_rt_function_t main_function;
  const char kMainFunctionName[] = "module.simple_mul";
  ASSERT_API_OK(iree_rt_context_resolve_function(
      context,
      iree_string_view_t{kMainFunctionName, sizeof(kMainFunctionName) - 1},
      &main_function));

  // Allocate buffers that can be mapped on the CPU and that can also be used
  // on the device. Not all devices support this, but the ones we have now do.
  LOG(INFO) << "Creating I/O buffers...";
  constexpr int kElementCount = 4;
  iree_hal_buffer_t* arg0_buffer = nullptr;
  iree_hal_buffer_t* arg1_buffer = nullptr;
  ASSERT_API_OK(iree_rt_context_allocate_device_visible_buffer(
      context, IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount,
      IREE_ALLOCATOR_DEFAULT, &arg0_buffer));
  ASSERT_API_OK(iree_rt_context_allocate_device_visible_buffer(
      context, IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount,
      IREE_ALLOCATOR_DEFAULT, &arg1_buffer));

  // Populate initial values for 4 * 2 = 8.
  float kFloat4 = 4.0f;
  float kFloat2 = 2.0f;
  ASSERT_API_OK(iree_hal_buffer_fill(arg0_buffer, 0, IREE_WHOLE_BUFFER,
                                     &kFloat4, sizeof(float)));
  ASSERT_API_OK(iree_hal_buffer_fill(arg1_buffer, 0, IREE_WHOLE_BUFFER,
                                     &kFloat2, sizeof(float)));

  // Wrap buffers in buffer views to provide shape information.
  std::array<iree_hal_buffer_view_t*, 2> arg_buffer_views;
  ASSERT_API_OK(iree_hal_buffer_view_create(
      arg0_buffer, iree_shape_t{1, {kElementCount}}, sizeof(float),
      IREE_ALLOCATOR_DEFAULT, &arg_buffer_views[0]));
  ASSERT_API_OK(iree_hal_buffer_view_create(
      arg1_buffer, iree_shape_t{1, {kElementCount}}, sizeof(float),
      IREE_ALLOCATOR_DEFAULT, &arg_buffer_views[1]));
  iree_hal_buffer_release(arg0_buffer);
  iree_hal_buffer_release(arg1_buffer);

  // Call into the @simple_mul function.
  LOG(INFO) << "Calling @simple_mul...";
  iree_rt_invocation_t* invocation = nullptr;
  ASSERT_API_OK(iree_rt_invocation_create(
      context, &main_function, nullptr, nullptr, arg_buffer_views.data(), 2,
      nullptr, 0, IREE_ALLOCATOR_DEFAULT, &invocation));
  ASSERT_API_OK(iree_hal_buffer_view_release(arg_buffer_views[0]));
  ASSERT_API_OK(iree_hal_buffer_view_release(arg_buffer_views[1]));
  ASSERT_API_OK(
      iree_rt_invocation_await(invocation, IREE_TIME_INFINITE_FUTURE));

  // Get the result buffers from the invocation.
  LOG(INFO) << "Retreiving results...";
  std::array<iree_hal_buffer_view_t*, 2> result_buffer_views;
  iree_host_size_t result_count;
  ASSERT_API_OK(iree_rt_invocation_consume_results(
      invocation, result_buffer_views.size(), IREE_ALLOCATOR_DEFAULT,
      result_buffer_views.data(), &result_count));
  iree_rt_invocation_release(invocation);

  // Read back the results and ensure we got the right values.
  LOG(INFO) << "Reading back results...";
  iree_hal_buffer_t* result_buffer =
      iree_hal_buffer_view_buffer(result_buffer_views[0]);
  iree_hal_mapped_memory_t mapped_memory;
  ASSERT_API_OK(iree_hal_buffer_map(result_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                    0, IREE_WHOLE_BUFFER, &mapped_memory));
  ASSERT_THAT(absl::Span<const float>(
                  reinterpret_cast<const float*>(mapped_memory.contents.data),
                  mapped_memory.contents.data_length / sizeof(float)),
              ::testing::ElementsAreArray({8.0f, 8.0f, 8.0f, 8.0f}));
  ASSERT_API_OK(iree_hal_buffer_unmap(result_buffer, &mapped_memory));
  LOG(INFO) << "Results match!";

  iree_hal_buffer_view_release(result_buffer_views[0]);

  iree_rt_context_release(context);
  iree_rt_instance_release(instance);
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, BytecodeModuleApiTest,
                         ::testing::ValuesIn(GetAvailableDriverTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
}  // namespace samples
}  // namespace rt
}  // namespace iree
