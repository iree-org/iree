#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef NO_IREE
#include "nl_api.h"
#include "iree/hal/local/executable_library.h"
#endif

#include "command_data.h"
#include "command_op.h"

int main() {
	command_data_t command_data;
	command_data_init(&command_data, COMMAND_DATA_PORT);
	command_op_t op;

	static unsigned char buffer[1048576];

	command_ptr_t ptr;
	command_ptr_t ptrs[4];
	command_value_t value;
	command_value_t size;

	iree_hal_executable_workgroup_state_v0_t workgroup_state;
	iree_hal_executable_dispatch_state_v0_t dispatch_state;

	while(0 <= command_data_process_connections(&command_data)) for(int c = 0; c < command_data_connection_count(&command_data); c++) while(sizeof(op) == command_data_connection_read(&command_data, c, &op, sizeof(op))) switch(op) {

		case OP_COPY_IN:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			command_data_connection_read_full(&command_data, c, ptr, size);
			break;
		case OP_COPY_OUT:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			command_data_connection_write(&command_data, c, ptr, size);
			break;
		case OP_COPY:
			command_data_connection_read(&command_data, c, ptrs, 2 * sizeof(ptrs[0]));
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			memcpy(ptrs[1], ptrs[0], size);
			break;
		case OP_DEVICE_ALLOC:
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			ptr = malloc(size);
			command_data_connection_write(&command_data, c, &ptr, sizeof(ptr));
			break;
		case OP_DEVICE_FREE:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			free(ptr);
			break;
#ifndef NO_IREE
		case OP_ELF_LOAD:
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			if(0 > command_data_connection_read_full(&command_data, c, buffer, size))
				break;

			ptr = nl_elf_executable_load(buffer, size);
			command_data_connection_write(&command_data, c, &ptr, sizeof(ptr));
			break;
		case OP_ELF_INIT:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			void *library = nl_elf_executable_init(ptr);
			ptr = (command_ptr_t)library;
			command_data_connection_write(&command_data, c, &ptr, sizeof(ptr));
			break;
		case OP_ELF_GET_ATTRS:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			printf("ELF GET ATTR %p\n", ptr);
			int n;
			nl_elf_executable_get_attrs(ptr, &ptr, &n);
			value = n;
			command_data_connection_write(&command_data, c, &value, sizeof(value));
			command_data_connection_write(&command_data, c, ptr, n * sizeof(iree_hal_executable_dispatch_attrs_v0_t));
			break;
		case OP_ELF_CALL:
			command_data_connection_read(&command_data, c, ptrs, 2 * sizeof(command_ptr_t));
			command_data_connection_read(&command_data, c, &workgroup_state, sizeof(workgroup_state));
			command_data_connection_read(&command_data, c, &dispatch_state, sizeof(dispatch_state));

			unsigned char *bptr = buffer;
			int data_len;

			// get constants
			if(dispatch_state.constant_count) {
				data_len = dispatch_state.constant_count * sizeof(dispatch_state.constants[0]);
				command_data_connection_read(&command_data, c, bptr, data_len);
				dispatch_state.constants = (void *)bptr;
				bptr += data_len;
			}

			// get binding lengths
			if(dispatch_state.binding_count) {
				data_len = dispatch_state.binding_count * sizeof(dispatch_state.binding_lengths[0]);
				command_data_connection_read(&command_data, c, bptr, data_len);
				dispatch_state.binding_lengths = (void *)bptr;
				bptr += data_len;

				data_len = dispatch_state.binding_count * sizeof(dispatch_state.binding_ptrs[0]);
				command_data_connection_read(&command_data, c, bptr, data_len);
				dispatch_state.binding_ptrs = (void *)bptr;
				bptr += data_len;
			}

			value = nl_elf_executable_call(ptrs[0], (command_value_t)(ptrs[1]), &dispatch_state, &workgroup_state);
			command_data_connection_write(&command_data, c, &value, sizeof(value));
			for(int i = 0; i < dispatch_state.binding_count; i++) {
				for(int j = 0; j < 4 && j < dispatch_state.binding_lengths[i] / sizeof(float); j++) {
					printf("BINDING[%d][%d] = %f\n", i, j, ((float *)dispatch_state.binding_ptrs[i])[j]);
				}
			}
			break;
		case OP_ELF_FREE:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			nl_elf_executable_destroy(ptr);
			break;
#endif
		default:
			command_data_connection_read(&command_data, c, buffer, sizeof(buffer));
			return -1;

	}

	command_data_close(&command_data);
}
