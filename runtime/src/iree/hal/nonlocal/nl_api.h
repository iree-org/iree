typedef void *nl_mem_device_ptr_t;
typedef void *nl_elf_module_handle_t;

nl_mem_device_ptr_t nl_mem_alloc(size_t size);
void nl_mem_free(nl_mem_device_ptr_t ptr);
nl_mem_device_ptr_t nl_mem_host_alloc(size_t size);
void nl_mem_host_free(void *ptr);
nl_elf_module_handle_t nl_elf_executable_load(const uint8_t *elf_data, int elf_data_length);
void *nl_elf_executable_init(nl_elf_module_handle_t module);
int nl_elf_executable_call(void *module_data, int ordinal, void *dispatch_state, void *workgroup_state);
void nl_elf_executable_destroy(nl_elf_module_handle_t module);
