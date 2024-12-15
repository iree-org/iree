#include <stdlib.h>

#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/executable_library.h"

#include "nl_api.h"

nl_mem_device_ptr_t nl_mem_alloc(size_t size) { return malloc(size); }
void nl_mem_free(nl_mem_device_ptr_t ptr) { free(ptr); }
nl_mem_device_ptr_t nl_mem_host_alloc(size_t size) { return malloc(size); }
void nl_mem_host_free(void *ptr) { free(ptr); }
void nl_mem_register(void *host_ptr, unsigned int size, unsigned int register_flags) {}
void nl_mem_host_unregister(void *host_ptr) {}
void nl_mem_prefectch_async(nl_mem_device_ptr_t ptr, unsigned int size) {}

// load elf data, return elf module
nl_elf_module_handle_t nl_elf_executable_load(const uint8_t *elf_data, int elf_data_length) {
  iree_elf_module_t *module = malloc(sizeof(iree_elf_module_t));

  iree_const_byte_span_t elf_span;
  elf_span.data = elf_data;
  elf_span.data_length = elf_data_length;

  if(!iree_status_is_ok(iree_elf_module_initialize_from_memory(
      elf_span, NULL, iree_allocator_system(), module)))
    return NULL;
  return (nl_elf_module_handle_t) module;
}

// init elf module, return module data
void *nl_elf_executable_init(nl_elf_module_handle_t module) {
  iree_hal_executable_environment_v0_t environment;
  iree_hal_executable_environment_initialize(iree_allocator_system(), &environment);

  void* query_fn_ptr = NULL;
  if(!iree_status_is_ok(iree_elf_module_lookup_export(
      module, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME, &query_fn_ptr)))
    return NULL;

  return iree_elf_call_p_ip(
          query_fn_ptr, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
          &environment);
}

// call function
int nl_elf_executable_call(void *module_data, int ordinal, void *dispatch_state, void *workgroup_state) {
  iree_hal_executable_environment_v0_t environment;
  iree_hal_executable_environment_initialize(iree_allocator_system(), &environment);

  if(ordinal >= ((iree_hal_executable_library_v0_t *)module_data)->exports.count) return -1;

  return iree_elf_call_i_ppp((const void*)(((iree_hal_executable_library_v0_t *)module_data)->exports.ptrs[ordinal]),
                                &environment, dispatch_state,
                                workgroup_state);
}

void nl_elf_executable_destroy(nl_elf_module_handle_t module) {
  iree_elf_module_deinitialize((iree_elf_module_t *)module);
  free(module);
}
