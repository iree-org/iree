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

#include <vector>

#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/testing/driver_registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

namespace {

// TODO(scotttodd): ಠ_ಠ
std::vector<char> HexToBytes(const std::string& hex) {
  std::vector<char> bytes;
  for (unsigned int i = 0; i < hex.length(); i += 2) {
    std::string byteString = hex.substr(i, 2);
    char byte = (char)strtol(byteString.c_str(), NULL, 16);
    bytes.push_back(byte);
  }
  return bytes;
}

}  // namespace

class ExecutableCacheTest : public CtsTestBase {};

TEST_P(ExecutableCacheTest, Create) {
  iree_hal_executable_cache_t* executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"), &executable_cache));

  iree_hal_executable_cache_release(executable_cache);
}

TEST_P(ExecutableCacheTest, CantPrepareUnknownFormat) {
  iree_hal_executable_cache_t* executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"), &executable_cache));

  iree_hal_executable_format_t format = iree_hal_make_executable_format("FOO?");
  EXPECT_FALSE(
      iree_hal_executable_cache_can_prepare_format(executable_cache, format));

  iree_hal_executable_cache_release(executable_cache);
}

TEST_P(ExecutableCacheTest, PrepareExecutable) {
  iree_hal_executable_cache_t* executable_cache;
  IREE_ASSERT_OK(iree_hal_executable_cache_create(
      device_, iree_make_cstring_view("default"), &executable_cache));

  // TODO(scotttodd): also write standalone tests for each of these
  iree_hal_descriptor_set_layout_t* descriptor_set_layout;
  iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {
      {/*binding=*/0, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_READ},
      {/*binding=*/1, /*type=*/IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       /*access=*/IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE},
  };
  IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
      device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_IMMUTABLE,
      IREE_ARRAYSIZE(descriptor_set_layout_bindings),
      descriptor_set_layout_bindings, &descriptor_set_layout));
  iree_hal_executable_layout_t* executable_layout;
  IREE_ASSERT_OK(iree_hal_executable_layout_create(
      device_, /*set_layout_count=*/1, &descriptor_set_layout,
      /*push_constants=*/0, &executable_layout));

  // TODO(scotttodd): ಠ_ಠ
  std::vector<char> executable_bytes = HexToBytes(
      "080000005350564504FDFFFF08000000240000000100000004000000110000006162735F"
      "65785F64697370617463685F30000000B300000003022307000001001600000015000000"
      "0000000011000200010000000A000B005350565F4B48525F73746F726167655F62756666"
      "65725F73746F726167655F636C617373000000000E00030000000000010000000F000800"
      "050000000A0000006162735F65785F64697370617463685F30000000100006000A000000"
      "1100000001000000010000000100000005000A00060000005F5F7265736F757263655F76"
      "61725F313236313034303135373732305F5F000005000A00070000005F5F7265736F7572"
      "63655F7661725F313236313034303136333931325F5F0000050007000A0000006162735F"
      "65785F64697370617463685F300000004700040003000000060000000400000048000500"
      "020000000000000023000000000000004700030002000000020000004700040006000000"
      "210000000000000047000400060000002200000000000000470003000200000002000000"
      "470004000700000021000000010000004700040007000000220000000000000015000400"
      "0400000020000000000000002B0004000400000005000000010000001C00040003000000"
      "04000000050000001E000300020000000300000020000400010000000C00000002000000"
      "3B00040001000000060000000C0000003B00040001000000070000000C00000013000200"
      "090000002100030008000000090000002B000400040000000C0000000000000020000400"
      "0D0000000C00000004000000140002001000000036000500090000000A00000000000000"
      "08000000F80002000B000000410006000D0000000E000000060000000C0000000C000000"
      "3D000400040000000F0000000E000000AF00050010000000110000000F0000000C000000"
      "8200050004000000120000000C0000000F000000A9000600040000001300000011000000"
      "0F00000012000000410006000D00000014000000070000000C0000000C0000003E000300"
      "1400000013000000FD0001003800010008000C0004000800");
  iree_const_byte_span_t executable_data = iree_make_const_byte_span(
      executable_bytes.data(), executable_bytes.size());

  iree_hal_executable_t* executable;
  IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
      executable_cache, executable_layout,
      IREE_HAL_EXECUTABLE_CACHING_MODE_DEFAULT, executable_data, &executable));

  iree_hal_executable_release(executable);
  iree_hal_executable_layout_release(executable_layout);
  iree_hal_descriptor_set_layout_release(descriptor_set_layout);
  iree_hal_executable_cache_release(executable_cache);
}

INSTANTIATE_TEST_SUITE_P(
    AllDrivers, ExecutableCacheTest,
    ::testing::ValuesIn(testing::EnumerateAvailableDrivers()),
    GenerateTestName());

}  // namespace cts
}  // namespace hal
}  // namespace iree
