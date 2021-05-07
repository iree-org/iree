// Copyright 2020 Google LLC
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

#include "iree/tools/utils/vm_util.h"

#include <sstream>

#include "absl/strings/str_cat.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vmla/registration/driver_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

class VmUtilTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_CHECK_OK(iree_hal_vmla_driver_module_register(
        iree_hal_driver_registry_default()));
  }

  virtual void SetUp() {
    IREE_ASSERT_OK(iree_hal_module_register_types());
    IREE_ASSERT_OK(CreateDevice("vmla", &device_));
    allocator_ = iree_hal_device_allocator(device_);
  }

  virtual void TearDown() { iree_hal_device_release(device_); }

  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
};

TEST_F(VmUtilTest, ParsePrintBuffer) {
  absl::string_view buf_string = "2x2xi32=[42 43][44 45]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(allocator_, {buf_string}, &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(),
            absl::StrCat("result[0]: hal.buffer_view\n", buf_string, "\n"));
}

TEST_F(VmUtilTest, ParsePrintScalar) {
  absl::string_view input_string = "42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(allocator_, {input_string}, &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(), absl::StrCat("result[0]: i32=", input_string, "\n"));
}

TEST_F(VmUtilTest, ParsePrintRank0Buffer) {
  absl::string_view buf_string = "i32=42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(allocator_, {buf_string}, &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(),
            absl::StrCat("result[0]: hal.buffer_view\n", buf_string, "\n"));
}

TEST_F(VmUtilTest, ParsePrintMultipleBuffers) {
  absl::string_view buf_string1 = "2x2xi32=[42 43][44 45]";
  absl::string_view buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(allocator_, {buf_string1, buf_string2},
                                    &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(),
            absl::StrCat("result[0]: hal.buffer_view\n", buf_string1,
                         "\nresult[1]: hal.buffer_view\n", buf_string2, "\n"));
}

}  // namespace
}  // namespace iree
