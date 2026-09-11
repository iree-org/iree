#include <stdlib.h>
#include <unistd.h>

#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_environment.h"
#include "iree/hal/local/executable_library.h"

#include "command_data.h"
#include "command_op.h"
#include "nl_api.h"
#include "debug.h"

static command_data_t *get_command_data() {
	static command_data_t command_data;
	static int set = 0;
	if(set) return &command_data;

	command_data_open(&command_data, "127.0.0.1", COMMAND_DATA_PORT);
	set = 1;
	return &command_data;
}

nl_mem_device_ptr_t nl_mem_alloc(size_t size) {
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_value_t size; } command;
	command_ptr_t p;
	command.op = OP_DEVICE_ALLOC;
	command.size = size;
	command_data_write(command_data, &command, sizeof(command));
	command_data_read(command_data, &p, sizeof(p));
	DEBUG_PRINTF("ALLOC %zu = %p\n", size, p);
	return p;
}

void nl_mem_free(nl_mem_device_ptr_t ptr) {
	DEBUG_PRINTF("FREE %p\n", ptr);
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_ptr_t ptr; } command;
	command.op = OP_DEVICE_FREE;
	command.ptr = ptr;
	command_data_write(command_data, &command, sizeof(command));
}

nl_mem_device_ptr_t nl_mem_host_alloc(size_t size) { return malloc(size); }
void nl_mem_host_free(void *ptr) { free(ptr); }

void nl_mem_copy_in(nl_mem_device_ptr_t dest, const void *src, size_t size) {
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_ptr_t dest; command_value_t size; } command;
	command.op = OP_COPY_IN;
	command.dest = dest;
	command.size = size;
	DEBUG_PRINTF("COPY IN %p %zu\n", dest, size);
	command_data_write(command_data, &command, sizeof(command));
	command_data_write(command_data, src, size);
}

void nl_mem_copy_out(void *dest, const nl_mem_device_ptr_t src, size_t size) {
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_ptr_t src; command_value_t size; } command;
	command.op = OP_COPY_OUT;
	command.src = src;
	command.size = size;
	DEBUG_PRINTF("COPY OUT %p %zu\n", src, size);
	command_data_write(command_data, &command, sizeof(command));
	command_data_read_full(command_data, dest, size);
}

void nl_mem_copy(nl_mem_device_ptr_t dest, const nl_mem_device_ptr_t src, size_t size) {
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_ptr_t src; command_ptr_t dest; command_value_t size; } command;
	command.op = OP_COPY;
	command.src = src;
	command.dest = dest;
	command.size = size;
	DEBUG_PRINTF("COPY %p %p %zu\n", src, dest, size);
	command_data_write(command_data, &command, sizeof(command));
}

// load elf data, return elf module
nl_elf_module_handle_t nl_elf_executable_load(const uint8_t *elf_data, int elf_data_length) {
	DEBUG_PRINTF("LOAD ELF %p %d\n", elf_data, elf_data_length);
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_value_t data_length; } command;
	command_ptr_t p;
	command.op = OP_ELF_LOAD;
	command.data_length = elf_data_length;
	command_data_write(command_data, &command, sizeof(command));
	command_data_write(command_data, (void *)elf_data, elf_data_length);
	command_data_read(command_data, &p, sizeof(p));
	DEBUG_PRINTF("LOADEd ELF %p\n", p);
	return p;
}

// init elf module, return module data
void *nl_elf_executable_init(nl_elf_module_handle_t module) {
	DEBUG_PRINTF("ELF INIT %p\n", module);
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_ptr_t module; } command;
	command_ptr_t p;
	command.op = OP_ELF_INIT;
	command.module = module;
	command_data_write(command_data, &command, sizeof(command));
	command_data_read(command_data, &p, sizeof(p));
	DEBUG_PRINTF("INITED ELF %p\n", p);
	return p;
}

void nl_elf_executable_get_attrs(void *module_data, void **attrs, int *count) {
	DEBUG_PRINTF("ELF GET ATTRS %p\n", module_data);
	command_data_t *command_data = get_command_data();
	command_value_t n;
	struct __attribute__((packed)) { command_op_t op; command_ptr_t module_data; } command;
	command.op = OP_ELF_GET_ATTRS;
	command.module_data = module_data;
	command_data_write(command_data, &command, sizeof(command));
	command_data_read(command_data, &n, sizeof(n));
	*attrs = malloc(n * sizeof(iree_hal_executable_dispatch_attrs_v0_t));
	command_data_read(command_data, *attrs, n * sizeof(iree_hal_executable_dispatch_attrs_v0_t));
	*count = n;
}

// call function
int nl_elf_executable_call(void *module_data, int ordinal, void *dispatch_state, void *workgroup_state) {
	DEBUG_PRINTF("ELF CALL %p %d\n", module_data, ordinal);
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_ptr_t module_data; command_value_t ordinal; } command;
	command_value_t n;
	iree_hal_executable_dispatch_state_v0_t *dispatch = (iree_hal_executable_dispatch_state_v0_t *)dispatch_state;
	command.op = OP_ELF_CALL;
	command.module_data = module_data;
	command.ordinal = ordinal;
	command_data_write(command_data, &command, sizeof(command));

	command_data_write(command_data, workgroup_state, sizeof(iree_hal_executable_workgroup_state_v0_t));
	command_data_write(command_data, dispatch, sizeof(*dispatch));
	if(dispatch->constant_count) command_data_write(command_data, dispatch->constants, dispatch->constant_count * sizeof(dispatch->constants[0]));
	if(dispatch->binding_count) {
		command_data_write(command_data, dispatch->binding_lengths, dispatch->binding_count * sizeof(dispatch->binding_lengths[0]));
		command_data_write(command_data, dispatch->binding_ptrs, dispatch->binding_count * sizeof(dispatch->binding_ptrs[0]));
	}

	command_data_read(command_data, &n, sizeof(n));
	DEBUG_PRINTF("CALLED ELF %llu\n", (long long unsigned int)n);
	return n;
}

void nl_elf_executable_destroy(nl_elf_module_handle_t module) {
	DEBUG_PRINTF("ELF FREE %p\n", module);
	command_data_t *command_data = get_command_data();
	struct __attribute__((packed)) { command_op_t op; command_ptr_t module; } command;
	command.op = OP_ELF_FREE;
	command.module = module;
	command_data_write(command_data, &command, sizeof(command));
}

