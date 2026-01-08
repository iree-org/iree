// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/executable_library.h"

#include "nl_api.h"

// ELF modules for various platforms embedded in the binary:
#include "iree/hal/local/elf/testdata/elementwise_mul.h"

static int query_arch_test_file_data(
    void ** out_file_data, int *out_file_size) {

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

   printf("pattern %.*s\n", (int)pattern.size, pattern.data);

   if (!iree_string_view_is_empty(pattern)) {
    for (size_t i = 0; i < elementwise_mul_size(); ++i) {
      const struct iree_file_toc_t* file_toc = &elementwise_mul_create()[i];
      printf("file_toc name %s\n", file_toc->name);
      if (iree_string_view_match_pattern(iree_make_cstring_view(file_toc->name),
                                         pattern)) {
        *out_file_data = (void *)file_toc->data;
        *out_file_size = file_toc->size;
        return 0;
      }
    }
  }
  printf(
                          "no architecture-specific ELF binary embedded into "
                          "the application for the current target platform\n");
  return -1;
}

static int run_test() {
  void *file_data;
  int file_size;
  query_arch_test_file_data(&file_data, &file_size);
  void *module = nl_elf_executable_load(file_data, file_size);

  void *module_data = nl_elf_executable_init(module);

  // ret0 = arg0 * arg1
  float arg0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float arg1[4] = {100.0f, 200.0f, 300.0f, 400.0f};
  float ret0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float expected[4] = {100.0f, 400.0f, 900.0f, 1600.0f};

  size_t binding_lengths[3] = {
      sizeof(arg0),
      sizeof(arg1),
      sizeof(ret0),
  };
  void *arg0_device = nl_mem_alloc(sizeof(arg0));
  void *arg1_device = nl_mem_alloc(sizeof(arg1));
  void *ret0_device = nl_mem_alloc(sizeof(ret0));
  nl_mem_copy_in(arg0_device, arg0, sizeof(arg0));
  nl_mem_copy_in(arg1_device, arg1, sizeof(arg1));
  void* binding_ptrs[3] = {
      arg0_device,
      arg1_device,
      ret0_device,
  };
  const iree_hal_executable_dispatch_state_v0_t dispatch_state = {
      .workgroup_size_x = 1,
      .workgroup_size_y = 1,
      .workgroup_size_z = 1,
      .workgroup_count_x = 1,
      .workgroup_count_y = 1,
      .workgroup_count_z = 1,
      .max_concurrency = 1,
      .binding_count = 3,
      .binding_lengths = binding_lengths,
      .binding_ptrs = binding_ptrs,
  };
  const iree_hal_executable_workgroup_state_v0_t workgroup_state = {
      .workgroup_id_x = 0,
      .workgroup_id_y = 0,
      .workgroup_id_z = 0,
      .processor_id = iree_cpu_query_processor_id(),
  };
  int ret = nl_elf_executable_call(module_data, 0, (void*)&dispatch_state, (void*)&workgroup_state);

  nl_mem_copy_out(ret0, ret0_device, sizeof(ret0));

  for (int i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    if (ret0[i] != expected[i]) {
      ret = -1;
      printf("output mismatch: ret[%d] = %.1f, expected %.1f\n", i,
                           ret0[i], expected[i]);
    }
  }
  if(0 == ret) {
    printf("success.\n");
  }

  nl_mem_free(arg0_device);
  nl_mem_free(arg1_device);
  nl_mem_free(ret0_device);

  nl_elf_executable_destroy(module);
  return ret;
}

int main() {
  return run_test();
}
