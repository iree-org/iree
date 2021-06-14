// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_library.h"

// ELF modules for various platforms embedded in the binary:
#include "iree/hal/local/elf/testdata/simple_mul_dispatch.h"

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
    for (size_t i = 0; i < simple_mul_dispatch_size(); ++i) {
      const struct iree_file_toc_t* file_toc = &simple_mul_dispatch_create()[i];
      if (iree_string_view_match_pattern(iree_make_cstring_view(file_toc->name),
                                         pattern)) {
        return iree_make_const_byte_span(file_toc->data, file_toc->size);
      }
    }
  }
  return iree_make_const_byte_span(NULL, 0);
}

iree_status_t Run() {
  iree_status_t ret_status = iree_ok_status();
  const iree_const_byte_span_t file_data = GetCurrentPlatformFile();
  if (!file_data.data_length) {
    fprintf(stdout, "No ELF file built for this platform, skip");
    return ret_status;
  }

  iree_elf_import_table_t import_table;
  memset(&import_table, 0, sizeof(import_table));
  iree_elf_module_t module;
  IREE_RETURN_IF_ERROR(iree_elf_module_initialize_from_memory(
      file_data, &import_table, iree_allocator_system(), &module));

  void* query_fn_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_elf_module_lookup_export(
      &module, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME, &query_fn_ptr));

  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;
  library.header =
      (const iree_hal_executable_library_header_t**)iree_elf_call_p_ip(
          query_fn_ptr, IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION,
          /*reserved=*/NULL);
  if (library.header == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "library header is empty");
  }

  const iree_hal_executable_library_header_t* header = *library.header;
  if (header->version != IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "library version error");
  }

  if (strncmp(header->name, "simple_mul_dispatch_0", strlen(header->name)) !=
      0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "library name mismatches");
  }

  if (library.v0->entry_point_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point count mismatches");
  }

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
  iree_hal_vec3_t workgroup_count = {{1, 1, 1}};
  iree_hal_vec3_t workgroup_size = {{1, 1, 1}};
  iree_hal_executable_dispatch_state_v0_t dispatch_state;
  memset(&dispatch_state, 0, sizeof(dispatch_state));
  dispatch_state.workgroup_count = workgroup_count;
  dispatch_state.workgroup_size = workgroup_size;
  dispatch_state.binding_count = 1;
  dispatch_state.binding_lengths = binding_lengths;
  dispatch_state.binding_ptrs = binding_ptrs;
  iree_hal_vec3_t workgroup_id = {{0, 0, 0}};
  int ret = iree_elf_call_i_pp((const void*)library.v0->entry_points[0],
                               (void*)&dispatch_state, (void*)&workgroup_id);

  if (ret != 0) {
    return iree_make_status(IREE_STATUS_INTERNAL, "elf call fails");
  }
  float expected_ret[4] = {100.0f, 400.0f, 900.0f, 1600.0f};
  for (int i = 0; i < 4; ++i) {
    if (ret0[i] != expected_ret[i]) {
      fprintf(stderr, "ret[%d] is incorrect; expected: %.1f; actual: %.1f\n", i,
              expected_ret[i], ret0[i]);
      ret_status = iree_make_status(IREE_STATUS_INTERNAL, "output mismatches");
    }
  }

  iree_elf_module_deinitialize(&module);
  return ret_status;
}

int main() {
  const iree_status_t result = Run();
  int ret = (int)iree_status_code(result);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
  }
  return ret;
}
