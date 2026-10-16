// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bare-metal single-op runner using the inline HAL: no HAL device, no
// threads, no filesystem. The backend variant is selected by USE_LLVMCPU below;
// the matching iree-compile invocations are in CMakeLists.txt.
//
// NOTE: Error paths may leak resources. This is acceptable for this single-shot
// runner because main() reports the error and exits immediately. The successful
// path releases all resources.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/inline/module.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// USE_LLVMCPU: vmfb compiled with llvm-cpu (inline-dynamic); kernels are real
// RISC-V machine code in an embedded ELF, loaded via the hal_loader module.
// Default: vmfb compiled with vmvx-inline (inline-static); kernels are VM
// bytecode and only the hal_inline module is needed.
#if defined(USE_LLVMCPU)
#include "iree/hal/local/loaders/embedded_elf_loader.h"
#include "iree/modules/hal/loader/module.h"
#include "samples/baremetal_riscv64/simple_mul_module_llvmcpu_c.h"
#define simple_mul_module_create \
  iree_samples_baremetal_riscv64_module_llvmcpu_create
#else
#include "samples/baremetal_riscv64/simple_mul_module_vmvx_c.h"
#define simple_mul_module_create \
  iree_samples_baremetal_riscv64_module_vmvx_create
#endif  // USE_LLVMCPU

static iree_status_t make_input_buffer_view(
    iree_hal_allocator_t* device_allocator, const float* data,
    iree_host_size_t count, iree_hal_buffer_view_t** out_view) {
  iree_hal_buffer_params_t params = {
      .type =
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device_allocator, params, count * sizeof(float), &buffer));
  iree_status_t status =
      iree_hal_buffer_map_write(buffer, 0, data, count * sizeof(float));
  if (iree_status_is_ok(status)) {
    iree_hal_dim_t shape[1] = {(iree_hal_dim_t)count};
    status = iree_hal_buffer_view_create(buffer, 1, shape,
                                         IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                         IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                         iree_allocator_system(), out_view);
  }
  iree_hal_buffer_release(buffer);
  return status;
}

static iree_status_t run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator, &instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_inline_types(instance));

  iree_hal_allocator_t* device_allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
      iree_make_cstring_view("bare-metal"), host_allocator, host_allocator,
      &device_allocator));

  iree_vm_module_t* hal_inline_module = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_inline_module_create(
      instance, IREE_HAL_INLINE_MODULE_FLAG_NONE,
      iree_hal_module_debug_sink_stdio(stderr), device_allocator,
      host_allocator, &hal_inline_module));

#if defined(USE_LLVMCPU)
  IREE_RETURN_IF_ERROR(iree_hal_module_register_loader_types(instance));
  iree_hal_executable_loader_t* elf_loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
      /*plugin_manager=*/NULL, host_allocator, &elf_loader));
  iree_vm_module_t* hal_loader_module = NULL;
  iree_status_t loader_status = iree_hal_loader_module_create(
      instance, IREE_HAL_LOADER_MODULE_FLAG_NONE, /*loader_count=*/1,
      &elf_loader, host_allocator, &hal_loader_module);
  iree_hal_executable_loader_release(elf_loader);
  IREE_RETURN_IF_ERROR(loader_status);
#endif  // USE_LLVMCPU

  const struct iree_file_toc_t* module_toc = simple_mul_module_create();
  iree_vm_module_t* bytecode_module = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_make_const_byte_span(module_toc->data, module_toc->size),
      iree_allocator_null(), host_allocator, &bytecode_module));

  iree_vm_context_t* context = NULL;
#if defined(USE_LLVMCPU)
  iree_vm_module_t* modules[] = {hal_inline_module, hal_loader_module,
                                 bytecode_module};
#else
  iree_vm_module_t* modules[] = {hal_inline_module, bytecode_module};
#endif  // USE_LLVMCPU
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      host_allocator, &context));
  iree_vm_module_release(hal_inline_module);
#if defined(USE_LLVMCPU)
  iree_vm_module_release(hal_loader_module);
#endif  // USE_LLVMCPU
  iree_vm_module_release(bytecode_module);

  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module.simple_mul"), &main_function));

  const float kFloat4[4] = {4.0f, 4.0f, 4.0f, 4.0f};
  const float kFloat2[4] = {2.0f, 2.0f, 2.0f, 2.0f};
  iree_hal_buffer_view_t* arg0 = NULL;
  iree_hal_buffer_view_t* arg1 = NULL;
  IREE_RETURN_IF_ERROR(make_input_buffer_view(device_allocator, kFloat4,
                                              IREE_ARRAYSIZE(kFloat4), &arg0));
  IREE_RETURN_IF_ERROR(make_input_buffer_view(device_allocator, kFloat2,
                                              IREE_ARRAYSIZE(kFloat2), &arg1));

  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                           host_allocator, &inputs));
  iree_vm_ref_t arg0_ref = iree_hal_buffer_view_move_ref(arg0);
  iree_vm_ref_t arg1_ref = iree_hal_buffer_view_move_ref(arg1);
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg0_ref));
  IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg1_ref));

  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                           host_allocator, &outputs));

  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/NULL, inputs, outputs, host_allocator));

  iree_hal_buffer_view_t* ret_view =
      iree_vm_list_get_buffer_view_assign(outputs, 0);
  if (!ret_view) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "no result buffer view");
  }

  float results[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
      iree_hal_buffer_view_buffer(ret_view), 0, results, sizeof(results)));

  printf("simple_mul result: [%f %f %f %f]\n", results[0], results[1],
         results[2], results[3]);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(results); ++i) {
    if (results[i] != 8.0f) {
      return iree_make_status(IREE_STATUS_UNKNOWN,
                              "result mismatch at %zu: %f != 8.0", (size_t)i,
                              results[i]);
    }
  }

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_vm_context_release(context);
  iree_hal_allocator_release(device_allocator);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

int main(void) {
  printf("bare-metal iree runner starting\n");
  iree_status_t status = run();
  int ret = (int)iree_status_code(status);
  if (iree_status_is_ok(status)) {
    printf("PASS: single-op inference on bare-metal riscv64\n");
  } else {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    printf("FAIL\n");
  }
  return ret;
}
