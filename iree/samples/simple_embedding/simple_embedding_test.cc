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

#include "absl/base/macros.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm2/api.h"
#include "iree/vm2/bytecode_module.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/simple_embedding/simple_embedding_test_bytecode_module.h"

namespace iree {
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
  iree_string_view_t* driver_names = nullptr;
  iree_host_size_t driver_count = 0;
  CHECK_EQ(IREE_STATUS_OK,
           iree_hal_driver_registry_query_available_drivers(
               IREE_ALLOCATOR_SYSTEM, &driver_names, &driver_count));
  for (int i = 0; i < driver_count; ++i) {
    TestParams test_params;
    test_params.driver_name =
        std::string(driver_names[i].data, driver_names[i].size);
    all_test_params.push_back(std::move(test_params));
  }
  iree_allocator_free(IREE_ALLOCATOR_SYSTEM, driver_names);
  return all_test_params;
}

class SimpleEmbeddingTest : public ::testing::Test,
                            public ::testing::WithParamInterface<TestParams> {
 protected:
};

TEST_P(SimpleEmbeddingTest, RunOnce) {
  // TODO(benvanik): move to instance-based registration.
  ASSERT_EQ(IREE_STATUS_OK, iree_hal_module_register_types());

  iree_vm_instance_t* instance = nullptr;
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance));

  // Create the driver/device as defined by the test and setup the HAL module.
  const auto& driver_name = GetParam().driver_name;
  LOG(INFO) << "Creating driver '" << driver_name << "'...";
  iree_hal_driver_t* driver = nullptr;
  ASSERT_API_OK(iree_hal_driver_registry_create_driver(
      iree_string_view_t{driver_name.data(), driver_name.size()},
      IREE_ALLOCATOR_SYSTEM, &driver));
  iree_hal_device_t* device = nullptr;
  ASSERT_API_OK(iree_hal_driver_create_default_device(
      driver, IREE_ALLOCATOR_SYSTEM, &device));
  iree_vm_module_t* hal_module = nullptr;
  ASSERT_API_OK(
      iree_hal_module_create(device, IREE_ALLOCATOR_SYSTEM, &hal_module));
  iree_hal_driver_release(driver);

  // Load bytecode module from the embedded data.
  LOG(INFO) << "Loading simple_module_test.mlir...";
  const auto* module_file_toc = simple_embedding_test_bytecode_module_create();
  iree_vm_module_t* bytecode_module = nullptr;
  ASSERT_API_OK(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = nullptr;
  std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
  ASSERT_EQ(IREE_STATUS_OK, iree_vm_context_create_with_modules(
                                instance, modules.data(), modules.size(),
                                IREE_ALLOCATOR_SYSTEM, &context));
  LOG(INFO) << "Module loaded and context is ready for use";
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  const char kMainFunctionName[] = "module.simple_mul";
  iree_vm_function_t main_function;
  CHECK_EQ(
      IREE_STATUS_OK,
      iree_vm_context_resolve_function(
          context, iree_make_cstring_view(kMainFunctionName), &main_function))
      << "Exported function '" << kMainFunctionName << "' not found";

  // Allocate buffers that can be mapped on the CPU and that can also be used
  // on the device. Not all devices support this, but the ones we have now do.
  LOG(INFO) << "Creating I/O buffers...";
  constexpr int kElementCount = 4;
  iree_hal_buffer_t* arg0_buffer = nullptr;
  iree_hal_buffer_t* arg1_buffer = nullptr;
  ASSERT_API_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device),
      iree_hal_memory_type_t(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                             IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount, &arg0_buffer));
  ASSERT_API_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device),
      iree_hal_memory_type_t(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                             IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount, &arg1_buffer));

  // Populate initial values for 4 * 2 = 8.
  float kFloat4 = 4.0f;
  float kFloat2 = 2.0f;
  ASSERT_API_OK(iree_hal_buffer_fill(arg0_buffer, 0, IREE_WHOLE_BUFFER,
                                     &kFloat4, sizeof(float)));
  ASSERT_API_OK(iree_hal_buffer_fill(arg1_buffer, 0, IREE_WHOLE_BUFFER,
                                     &kFloat2, sizeof(float)));

  // Setup call inputs with our buffers.
  // TODO(benvanik): make a macro/magic.
  iree_vm_variant_list_t* inputs = nullptr;
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_variant_list_alloc(2, IREE_ALLOCATOR_SYSTEM, &inputs));
  auto arg0_buffer_ref = iree_hal_buffer_move_ref(arg0_buffer);
  auto arg1_buffer_ref = iree_hal_buffer_move_ref(arg1_buffer);
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_variant_list_append_ref_move(inputs, &arg0_buffer_ref));
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_variant_list_append_ref_move(inputs, &arg1_buffer_ref));

  // Prepare outputs list to accept the results from the invocation.
  iree_vm_variant_list_t* outputs = nullptr;
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &outputs));

  // Synchronously invoke the function.
  LOG(INFO) << "Calling " << kMainFunctionName << "...";
  ASSERT_EQ(IREE_STATUS_OK, iree_vm_invoke(context, main_function,
                                           /*policy=*/nullptr, inputs, outputs,
                                           IREE_ALLOCATOR_SYSTEM));
  iree_vm_variant_list_free(inputs);

  // Get the result buffers from the invocation.
  LOG(INFO) << "Retreiving results...";
  iree_hal_buffer_t* ret_buffer =
      iree_hal_buffer_deref(&iree_vm_variant_list_get(outputs, 0)->ref);
  ASSERT_NE(nullptr, ret_buffer);

  // Read back the results and ensure we got the right values.
  LOG(INFO) << "Reading back results...";
  iree_hal_mapped_memory_t mapped_memory;
  ASSERT_API_OK(iree_hal_buffer_map(ret_buffer, IREE_HAL_MEMORY_ACCESS_READ, 0,
                                    IREE_WHOLE_BUFFER, &mapped_memory));
  ASSERT_THAT(absl::Span<const float>(
                  reinterpret_cast<const float*>(mapped_memory.contents.data),
                  mapped_memory.contents.data_length / sizeof(float)),
              ::testing::ElementsAreArray({8.0f, 8.0f, 8.0f, 8.0f}));
  ASSERT_API_OK(iree_hal_buffer_unmap(ret_buffer, &mapped_memory));
  LOG(INFO) << "Results match!";

  iree_vm_variant_list_free(outputs);

  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
}

INSTANTIATE_TEST_SUITE_P(AllDrivers, SimpleEmbeddingTest,
                         ::testing::ValuesIn(GetAvailableDriverTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
}  // namespace samples
}  // namespace iree
