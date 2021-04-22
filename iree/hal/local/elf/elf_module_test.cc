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

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/hal/local/executable_library.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" {
#include "iree/hal/local/elf/elf_module.h"
}  // extern "C"

// ELF modules for various platforms embedded in the binary:
#include "iree/hal/local/elf/testdata/simple_mul_dispatch.h"

namespace {

class ELFModuleTest : public ::testing::Test {
 protected:
  const iree_const_byte_span_t GetCurrentPlatformFile() {
    iree_string_view_t pattern = iree_string_view_empty();
#if defined(IREE_ARCH_ARM_32)
    pattern = iree_make_cstring_view("*_arm_32.so");
#elif defined(IREE_ARCH_ARM_64)
    pattern = iree_make_cstring_view("*_arm_64.so");
#elif defined(IREE_ARCH_RISCV_32)
    pattern = iree_make_cstring_view("*_riscv_32.so");
#elif defined(IREE_ARCH_RISCV_64)
    pattern = iree_make_cstring_view("*_riscv_64.so");
#elif defined(IREE_ARCH_X86_32)
    pattern = iree_make_cstring_view("*_x86_32.so");
#elif defined(IREE_ARCH_X86_64)
    pattern = iree_make_cstring_view("*_x86_64.so");
#else
#warning "No architecture pattern specified; ELF linker will not be tested"
#endif  // IREE_ARCH_*

    if (!iree_string_view_is_empty(pattern)) {
      for (size_t i = 0; i < iree::elf::simple_mul_dispatch_size(); ++i) {
        const auto* file_toc = &iree::elf::simple_mul_dispatch_create()[i];
        if (iree_string_view_match_pattern(
                iree_make_cstring_view(file_toc->name), pattern)) {
          return iree_make_const_byte_span(file_toc->data, file_toc->size);
        }
      }
    }
    return iree_make_const_byte_span(nullptr, 0);
  }
};

TEST_F(ELFModuleTest, Check) {
  auto file_data = GetCurrentPlatformFile();
  if (!file_data.data_length) {
    GTEST_SKIP() << "No ELF file built for this platform";
    return;
  }

  iree_elf_import_table_t import_table;
  memset(&import_table, 0, sizeof(import_table));
  iree_elf_module_t module;
  IREE_ASSERT_OK(iree_elf_module_initialize_from_memory(
      file_data, &import_table, iree_allocator_system(), &module));

  void* query_fn_ptr = NULL;
  IREE_ASSERT_OK(iree_elf_module_lookup_export(
      &module, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME, &query_fn_ptr));

  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;
  library.header =
      (const iree_hal_executable_library_header_t**)iree_elf_call_p_ip(
          query_fn_ptr, IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION,
          /*reserved=*/NULL);
  ASSERT_TRUE(library.header != NULL);

  auto* header = *library.header;
  ASSERT_EQ(header->version, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0);
  ASSERT_STREQ(header->name, "simple_mul_dispatch_0");
  ASSERT_EQ(1, library.v0->entry_point_count);

  float arg0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float arg1[4] = {100.0f, 200.0f, 300.0f, 400.0f};
  float ret0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  size_t binding_lengths[3] = {
      sizeof(arg0),
      sizeof(arg1),
      sizeof(ret0),
  };
  void* binding_ptrs[3] = {
      arg0,
      arg1,
      ret0,
  };

  iree_hal_executable_dispatch_state_v0_t dispatch_state;
  memset(&dispatch_state, 0, sizeof(dispatch_state));
  dispatch_state.workgroup_count = {{1, 1, 1}};
  dispatch_state.workgroup_size = {{1, 1, 1}};
  dispatch_state.binding_count = 1;
  dispatch_state.binding_lengths = binding_lengths;
  dispatch_state.binding_ptrs = binding_ptrs;
  iree_hal_vec3_t workgroup_id = {{0, 0, 0}};
  int ret = iree_elf_call_i_pp((const void*)library.v0->entry_points[0],
                               (void*)&dispatch_state, (void*)&workgroup_id);
  EXPECT_EQ(0, ret);

  EXPECT_EQ(ret0[0], 100.0f);
  EXPECT_EQ(ret0[1], 400.0f);
  EXPECT_EQ(ret0[2], 900.0f);
  EXPECT_EQ(ret0[3], 1600.0f);

  iree_elf_module_deinitialize(&module);
}

}  // namespace
