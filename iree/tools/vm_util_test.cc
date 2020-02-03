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

#include "iree/tools/vm_util.h"

#include <sstream>

#include "iree/base/api.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/value.h"
#include "iree/vm/variant_list.h"

namespace iree {
namespace {

class VmUtilTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_ASSERT_OK(iree_hal_module_register_types());
    ASSERT_OK(CreateDevice("interpreter", &device_));
    allocator_ = iree_hal_device_allocator(device_);
  }

  virtual void TearDown() {
    IREE_ASSERT_OK(iree_hal_device_release(device_));
    if (outputs_ != nullptr) {
      IREE_ASSERT_OK(iree_vm_variant_list_free(outputs_));
    }
  }

  iree_hal_device_t* device_ = nullptr;
  iree_vm_variant_list_t* outputs_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
};

TEST_F(VmUtilTest, PrintVariantListScalar) {
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &outputs_));
  IREE_ASSERT_OK(
      iree_vm_variant_list_append_value(outputs_, IREE_VM_VALUE_MAKE_I32(42)));
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kScalar;
  desc.scalar.type = AbiConstants::ScalarType::kSint32;
  std::stringstream os;
  ASSERT_OK(PrintVariantList({desc}, outputs_, &os));
  EXPECT_EQ(os.str(), "i32=42\n");
}

TEST_F(VmUtilTest, PrintVariantListMultiple) {
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(2, IREE_ALLOCATOR_SYSTEM, &outputs_));
  IREE_ASSERT_OK(
      iree_vm_variant_list_append_value(outputs_, IREE_VM_VALUE_MAKE_I32(42)));
  IREE_ASSERT_OK(
      iree_vm_variant_list_append_value(outputs_, IREE_VM_VALUE_MAKE_I32(13)));
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kScalar;
  desc.scalar.type = AbiConstants::ScalarType::kSint32;
  std::stringstream os;
  ASSERT_OK(PrintVariantList({desc, desc}, outputs_, &os));
  EXPECT_EQ(os.str(),
            "i32=42\n"
            "i32=13\n");
}

TEST_F(VmUtilTest, PrintVariantListBuffer) {
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &outputs_));
  iree_hal_buffer_t* buf = nullptr;
  std::array<int32_t, 4> buf_data{42, -43, 44, 45};
  iree_device_size_t allocation_size = sizeof(buf_data);
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator_,
      static_cast<iree_hal_memory_type_t>(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      static_cast<iree_hal_buffer_usage_t>(IREE_HAL_BUFFER_USAGE_ALL |
                                           IREE_HAL_BUFFER_USAGE_CONSTANT),
      allocation_size, &buf));
  IREE_ASSERT_OK(iree_hal_buffer_write_data(
      buf, 0, reinterpret_cast<uint8_t*>(buf_data.data()), allocation_size));
  auto buf_ref = iree_hal_buffer_move_ref(buf);
  IREE_ASSERT_OK(iree_vm_variant_list_append_ref_move(outputs_, &buf_ref));
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kBuffer;
  desc.buffer.scalar_type = AbiConstants::ScalarType::kSint32;
  desc.dims = {2, 2};
  std::stringstream os;
  ASSERT_OK(PrintVariantList({desc}, outputs_, &os));
  EXPECT_EQ(os.str(), "2x2xi32=[42 -43][44 45]\n");
}

TEST_F(VmUtilTest, PrintVariantListMultiBuffer) {
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(2, IREE_ALLOCATOR_SYSTEM, &outputs_));

  // Buffer 1
  iree_hal_buffer_t* buf1 = nullptr;
  int element_count1 = 4;
  int element_size1 = sizeof(int32_t);
  iree_device_size_t allocation_size1 = element_count1 * element_size1;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator_,
      static_cast<iree_hal_memory_type_t>(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      static_cast<iree_hal_buffer_usage_t>(IREE_HAL_BUFFER_USAGE_ALL |
                                           IREE_HAL_BUFFER_USAGE_CONSTANT),
      allocation_size1, &buf1));
  std::array<int32_t, 4> buf1_data{42, 43, 44, 45};
  IREE_ASSERT_OK(iree_hal_buffer_write_data(
      buf1, 0, reinterpret_cast<uint8_t*>(buf1_data.data()), allocation_size1));
  auto buf1_ref = iree_hal_buffer_move_ref(buf1);
  IREE_ASSERT_OK(iree_vm_variant_list_append_ref_move(outputs_, &buf1_ref));
  RawSignatureParser::Description desc1;
  desc1.type = RawSignatureParser::Type::kBuffer;
  desc1.buffer.scalar_type = AbiConstants::ScalarType::kSint32;
  desc1.dims = {2, 2};

  // Buffer 2
  iree_hal_buffer_t* buf2 = nullptr;
  int element_count2 = 6;
  int element_size2 = sizeof(double);
  iree_device_size_t allocation_size2 = element_count2 * element_size2;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator_,
      static_cast<iree_hal_memory_type_t>(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      static_cast<iree_hal_buffer_usage_t>(IREE_HAL_BUFFER_USAGE_ALL |
                                           IREE_HAL_BUFFER_USAGE_CONSTANT),
      allocation_size2, &buf2));
  std::array<double, 6> buf2_data{1, 2, 3, 4, 5, 6};
  IREE_ASSERT_OK(iree_hal_buffer_write_data(
      buf2, 0, reinterpret_cast<uint8_t*>(buf2_data.data()), allocation_size2));
  auto buf2_ref = iree_hal_buffer_move_ref(buf2);
  IREE_ASSERT_OK(iree_vm_variant_list_append_ref_move(outputs_, &buf2_ref));
  RawSignatureParser::Description desc2;
  desc2.type = RawSignatureParser::Type::kBuffer;
  desc2.buffer.scalar_type = AbiConstants::ScalarType::kIeeeFloat64;
  desc2.dims = {2, 3};

  std::stringstream os;
  ASSERT_OK(PrintVariantList({desc1, desc2}, outputs_, &os));
  EXPECT_EQ(os.str(),
            "2x2xi32=[42 43][44 45]\n"
            "2x3xf64=[1 2 3][4 5 6]\n");
}

TEST_F(VmUtilTest, ParsePrint) {
  auto buf_string = "2x2xi32=[42 43][44 45]";
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kBuffer;
  desc.buffer.scalar_type = AbiConstants::ScalarType::kSint32;
  desc.dims = {2, 2};

  ASSERT_OK_AND_ASSIGN(auto* variant_list,
                       ParseToVariantList({desc}, allocator_, {buf_string}));
  std::stringstream os;
  ASSERT_OK(PrintVariantList({desc}, variant_list, &os));
  EXPECT_EQ(os.str(), absl::StrCat(buf_string, "\n"));

  IREE_ASSERT_OK(iree_vm_variant_list_free(variant_list));
}

TEST_F(VmUtilTest, ParsePrintRank0Buffer) {
  auto buf_string = "i32=42";
  RawSignatureParser::Description desc;
  desc.type = RawSignatureParser::Type::kBuffer;
  desc.buffer.scalar_type = AbiConstants::ScalarType::kSint32;

  ASSERT_OK_AND_ASSIGN(auto* variant_list,
                       ParseToVariantList({desc}, allocator_, {buf_string}));
  std::stringstream os;
  ASSERT_OK(PrintVariantList({desc}, variant_list, &os));
  EXPECT_EQ(os.str(), absl::StrCat(buf_string, "\n"));

  IREE_ASSERT_OK(iree_vm_variant_list_free(variant_list));
}

TEST_F(VmUtilTest, ParsePrintMultipleBuffers) {
  auto buf_string1 = "2x2xi32=[42 43][44 45]";
  RawSignatureParser::Description desc1;
  desc1.type = RawSignatureParser::Type::kBuffer;
  desc1.buffer.scalar_type = AbiConstants::ScalarType::kSint32;
  desc1.dims = {2, 2};

  auto buf_string2 = "2x3xf64=[1 2 3][4 5 6]";
  RawSignatureParser::Description desc2;
  desc2.type = RawSignatureParser::Type::kBuffer;
  desc2.buffer.scalar_type = AbiConstants::ScalarType::kIeeeFloat64;
  desc2.dims = {2, 3};

  ASSERT_OK_AND_ASSIGN(auto* variant_list,
                       ParseToVariantList({desc1, desc2}, allocator_,
                                          {buf_string1, buf_string2}));
  std::stringstream os;
  ASSERT_OK(PrintVariantList({desc1, desc2}, variant_list, &os));
  EXPECT_EQ(os.str(), absl::StrCat(buf_string1, "\n", buf_string2, "\n"));

  IREE_ASSERT_OK(iree_vm_variant_list_free(variant_list));
}

}  // namespace
}  // namespace iree
