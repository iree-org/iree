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
