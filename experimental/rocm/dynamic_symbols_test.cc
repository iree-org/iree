// Copyright 2021 Google LLC
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

#include "experimental/rocm/dynamic_symbols.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace rocm {
namespace {

#define ROCM_CHECK_ERRORS(expr)    \
  {                                \
    hipError_t status = expr;      \
    ASSERT_EQ(hipSuccess, status); \
  }

TEST(DynamicSymbolsTest, CreateFromSystemLoader) {
  iree_hal_rocm_dynamic_symbols_t symbols;
  iree_status_t status = iree_hal_rocm_dynamic_symbols_initialize(
      iree_allocator_system(), &symbols);
  if (!iree_status_is_ok(status)) {
    IREE_LOG(WARNING) << "Symbols cannot be loaded, skipping test.";
    GTEST_SKIP();
  }

  int device_count = 0;
  ROCM_CHECK_ERRORS(symbols.hipInit(0));
  ROCM_CHECK_ERRORS(symbols.hipGetDeviceCount(&device_count));
  if (device_count > 0) {
    hipDevice_t device;
    ROCM_CHECK_ERRORS(symbols.hipDeviceGet(&device, /*ordinal=*/0));
  }

  iree_hal_rocm_dynamic_symbols_deinitialize(&symbols);
}

}  // namespace
}  // namespace rocm
}  // namespace hal
}  // namespace iree
