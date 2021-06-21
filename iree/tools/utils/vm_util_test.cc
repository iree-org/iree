// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/utils/vm_util.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vmvx/registration/driver_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/ref_cc.h"

namespace iree {
namespace {

class VmUtilTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_CHECK_OK(iree_hal_vmvx_driver_module_register(
        iree_hal_driver_registry_default()));
  }

  virtual void SetUp() {
    IREE_ASSERT_OK(iree_hal_module_register_types());
    IREE_ASSERT_OK(CreateDevice("vmvx", &device_));
    allocator_ = iree_hal_device_allocator(device_);
  }

  virtual void TearDown() { iree_hal_device_release(device_); }

  iree_hal_device_t* device_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
};

TEST_F(VmUtilTest, ParsePrintBuffer) {
  std::string buf_string = "2x2xi32=[42 43][44 45]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      allocator_, std::vector<std::string>{buf_string}, &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(),
            std::string("result[0]: hal.buffer_view\n") + buf_string + "\n");
}

TEST_F(VmUtilTest, ParsePrintScalar) {
  std::string input_string = "42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      allocator_, std::vector<std::string>{input_string}, &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(), std::string("result[0]: i32=") + input_string + "\n");
}

TEST_F(VmUtilTest, ParsePrintRank0Buffer) {
  std::string buf_string = "i32=42";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      allocator_, std::vector<std::string>{buf_string}, &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(),
            std::string("result[0]: hal.buffer_view\n") + buf_string + "\n");
}

TEST_F(VmUtilTest, ParsePrintMultipleBuffers) {
  std::string buf_string1 = "2x2xi32=[42 43][44 45]";
  std::string buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  vm::ref<iree_vm_list_t> variant_list;
  IREE_ASSERT_OK(ParseToVariantList(
      allocator_, std::vector<std::string>{buf_string1, buf_string2},
      &variant_list));
  std::stringstream os;
  IREE_ASSERT_OK(PrintVariantList(variant_list.get(), &os));
  EXPECT_EQ(os.str(), std::string("result[0]: hal.buffer_view\n") +
                          buf_string1 + "\nresult[1]: hal.buffer_view\n" +
                          buf_string2 + "\n");
}

}  // namespace
}  // namespace iree
