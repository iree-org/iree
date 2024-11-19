typedef void *nl_mem_device_ptr_t;
typedef void *nl_elf_module_handle_t;

nl_mem_device_ptr_t nl_mem_alloc(size_t size);
void nl_mem_free(nl_mem_device_ptr_t ptr);
nl_mem_device_ptr_t nl_mem_host_alloc(size_t size);
void nl_mem_host_free(void *ptr);
void nl_mem_host_get_device_pointer(nl_mem_device_ptr_t *dptr, void *hptr, unsigned int flags);
void nl_mem_register(void *host_ptr, unsigned int size, unsigned int register_flags);
void nl_mem_host_unregister(void *host_ptr);
void nl_mem_prefectch_async(nl_mem_device_ptr_t ptr, unsigned int size);
