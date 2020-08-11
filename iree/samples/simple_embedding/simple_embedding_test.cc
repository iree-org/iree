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
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/simple_embedding/simple_embedding_test_bytecode_module.h"

namespace iree {
namespace samples {
namespace {

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
  IREE_CHECK_OK(iree_hal_driver_registry_query_available_drivers(
      iree_allocator_system(), &driver_names, &driver_count));
  for (int i = 0; i < driver_count; ++i) {
    TestParams test_params;
    test_params.driver_name =
        std::string(driver_names[i].data, driver_names[i].size);
    all_test_params.push_back(std::move(test_params));
  }
  iree_allocator_free(iree_allocator_system(), driver_names);
  return all_test_params;
}

class SimpleEmbeddingTest : public ::testing::Test,
                            public ::testing::WithParamInterface<TestParams> {
 protected:
};

TEST_P(SimpleEmbeddingTest, RunOnce) {
  // TODO(benvanik): move to instance-based registration.
  IREE_ASSERT_OK(iree_hal_module_register_types());

  iree_vm_instance_t* instance = nullptr;
  IREE_ASSERT_OK(iree_vm_instance_create(iree_allocator_system(), &instance));

  // Create the driver/device as defined by the test and setup the HAL module.
  const auto& driver_name = GetParam().driver_name;
  LOG(INFO) << "Creating driver '" << driver_name << "'...";
  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_create_driver(
      iree_string_view_t{driver_name.data(), driver_name.size()},
      iree_allocator_system(), &driver));
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), &device));
  iree_vm_module_t* hal_module = nullptr;
  IREE_ASSERT_OK(
      iree_hal_module_create(device, iree_allocator_system(), &hal_module));
  iree_hal_driver_release(driver);

  // Load bytecode module from the embedded data.
  LOG(INFO) << "Loading simple_module_test.mlir...";
  const auto* module_file_toc = simple_embedding_test_bytecode_module_create();
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      iree_allocator_null(), iree_allocator_system(), &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = nullptr;
  std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
  IREE_ASSERT_OK(iree_vm_context_create_with_modules(
      instance, modules.data(), modules.size(), iree_allocator_system(),
      &context));
  LOG(INFO) << "Module loaded and context is ready for use";
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  // Note that we use the "raw" variant which operates on pure type/shape
  // erased buffers.
  const char kMainFunctionName[] = "module.simple_mul$raw";
  iree_vm_function_t main_function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function))
      << "Exported function '" << kMainFunctionName << "' not found";

  // Allocate buffers that can be mapped on the CPU and that can also be used
  // on the device. Not all devices support this, but the ones we have now do.
  LOG(INFO) << "Creating I/O buffers...";
  constexpr int kElementCount = 4;
  iree_hal_buffer_t* arg0_buffer = nullptr;
  iree_hal_buffer_t* arg1_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device),
      iree_hal_memory_type_t(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                             IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount, &arg0_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device),
      iree_hal_memory_type_t(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                             IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount, &arg1_buffer));

  // Populate initial values for 4 * 2 = 8.
  float kFloat4 = 4.0f;
  float kFloat2 = 2.0f;
  IREE_ASSERT_OK(iree_hal_buffer_fill(arg0_buffer, 0, IREE_WHOLE_BUFFER,
                                      &kFloat4, sizeof(float)));
  IREE_ASSERT_OK(iree_hal_buffer_fill(arg1_buffer, 0, IREE_WHOLE_BUFFER,
                                      &kFloat2, sizeof(float)));

  // Setup call inputs with our buffers.
  // TODO(benvanik): make a macro/magic.
  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 2,
                                     iree_allocator_system(), &inputs));
  auto arg0_buffer_ref = iree_hal_buffer_move_ref(arg0_buffer);
  auto arg1_buffer_ref = iree_hal_buffer_move_ref(arg1_buffer);
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &arg0_buffer_ref));
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &arg1_buffer_ref));

  // Prepare outputs list to accept the results from the invocation.
  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                     iree_allocator_system(), &outputs));

  // Synchronously invoke the function.
  LOG(INFO) << "Calling " << kMainFunctionName << "...";
  IREE_ASSERT_OK(iree_vm_invoke(context, main_function,
                                /*policy=*/nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  // Get the result buffers from the invocation.
  LOG(INFO) << "Retreiving results...";
  auto* ret_buffer =
      reinterpret_cast<iree_hal_buffer_t*>(iree_vm_list_get_ref_deref(
          outputs.get(), 0, iree_hal_buffer_get_descriptor()));
  ASSERT_NE(nullptr, ret_buffer);

  // Read back the results and ensure we got the right values.
  LOG(INFO) << "Reading back results...";
  iree_hal_mapped_memory_t mapped_memory;
  IREE_ASSERT_OK(iree_hal_buffer_map(ret_buffer, IREE_HAL_MEMORY_ACCESS_READ, 0,
                                     IREE_WHOLE_BUFFER, &mapped_memory));
  ASSERT_THAT(absl::Span<const float>(
                  reinterpret_cast<const float*>(mapped_memory.contents.data),
                  mapped_memory.contents.data_length / sizeof(float)),
              ::testing::ElementsAreArray({8.0f, 8.0f, 8.0f, 8.0f}));
  IREE_ASSERT_OK(iree_hal_buffer_unmap(ret_buffer, &mapped_memory));
  LOG(INFO) << "Results match!";

  inputs.reset();
  outputs.reset();
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
