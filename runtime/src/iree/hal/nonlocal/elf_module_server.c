#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <inttypes.h>

#ifndef NO_IREE
#include "nl_api.h"
#include "iree/hal/local/executable_library.h"
#endif

#include "command_data.h"
#include "command_op.h"

#define BINDING_DEBUG_TYPE uint32_t
#define BINDING_DEBUG_FORMAT "u"

#include "debug.h"

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
			DEBUG_PRINTF("COPY IN %" PRIu64 " = %p ... %p \n", size, ptr, (uint8_t *)ptr + size);
			break;
		case OP_COPY_OUT:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			command_data_connection_write(&command_data, c, ptr, size);
			DEBUG_PRINTF("COPY OUT %" PRIu64 " = %p ... %p \n", size, ptr, (uint8_t *)ptr + size);
			break;
		case OP_COPY:
			command_data_connection_read(&command_data, c, ptrs, 2 * sizeof(ptrs[0]));
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			DEBUG_PRINTF("COPY OUT %p %p %" PRIu64 "\n", ptrs[0], ptrs[1], size);
			memcpy(ptrs[1], ptrs[0], size);
			break;
		case OP_DEVICE_ALLOC:
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			DEBUG_PRINTF("ALLOC %" PRIu64 "\n", size);
			posix_memalign(&ptr, 64, size);
			command_data_connection_write(&command_data, c, &ptr, sizeof(ptr));
			DEBUG_PRINTF("ALLOCED %" PRIu64 " = %p ... %p \n", size, ptr, (uint8_t *)ptr + size);
			break;
		case OP_DEVICE_FREE:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			DEBUG_PRINTF("FREE %p\n", ptr);
			free(ptr);
			DEBUG_PRINTF("FREED %p\n", ptr);
			break;
#ifndef NO_IREE
		case OP_ELF_LOAD:
			command_data_connection_read(&command_data, c, &size, sizeof(size));
			DEBUG_PRINTF("ELF LOAD READ %p %" PRIu64 "\n", buffer, size);
			if(0 > command_data_connection_read_full(&command_data, c, buffer, size))
				break;
			DEBUG_PRINTF("ELF LOAD %p %" PRIu64 "\n", buffer, size);

			ptr = nl_elf_executable_load(buffer, size);
			command_data_connection_write(&command_data, c, &ptr, sizeof(ptr));
			DEBUG_PRINTF("ELF LOADED %p ... %p = %p\n", buffer, buffer + size, ptr);
			break;
		case OP_ELF_INIT:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			DEBUG_PRINTF("ELF INIT %p\n", ptr);
			void *library = nl_elf_executable_init(ptr);
			ptr = (command_ptr_t)library;
			command_data_connection_write(&command_data, c, &ptr, sizeof(ptr));
			DEBUG_PRINTF("ELF INITED = %p\n", ptr);
			break;
		case OP_ELF_GET_ATTRS:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			DEBUG_PRINTF("ELF GET ATTR %p\n", ptr);
			int n;
			nl_elf_executable_get_attrs(ptr, &ptr, &n);
			value = n;
			command_data_connection_write(&command_data, c, &value, sizeof(value));
			command_data_connection_write(&command_data, c, ptr, n * sizeof(iree_hal_executable_dispatch_attrs_v0_t));
			break;
		case OP_ELF_CALL:
			command_data_connection_read(&command_data, c, ptrs, 2 * sizeof(command_ptr_t));
#if defined(DEBUG) && DEBUG > 1
			DEBUG_PRINTF("ELF CALL %p %llu\n", ptrs[0], (long long unsigned int)ptrs[1]);
#endif
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

#if defined(DEBUG) && DEBUG > 1
			for(int i = 0; i < dispatch_state.constant_count; i++) {
				DEBUG_PRINTF("%p CONSTANT[%d] = %d\n", dispatch_state.constants, i, dispatch_state.constants[i]);
			}
#endif

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
#if defined(DEBUG) && DEBUG > 1
			DEBUG_PRINTF("ELF CALLED = %llu\n", (long long unsigned int)value);
			for(int i = 0; i < dispatch_state.binding_count; i++) {
				DEBUG_PRINTF("BINDING[%d] = %p ... %p\n", i, dispatch_state.binding_ptrs[i], (uint8_t *)(dispatch_state.binding_ptrs[i]) + dispatch_state.binding_lengths[i]);
				for(int j = 0; j < dispatch_state.binding_lengths[i] / sizeof(BINDING_DEBUG_TYPE); j++) {
					DEBUG_PRINTF("BINDING[%d][%d/%zu] = %" BINDING_DEBUG_FORMAT "\n", i, j, dispatch_state.binding_lengths[i] / sizeof(BINDING_DEBUG_TYPE), ((BINDING_DEBUG_TYPE *)dispatch_state.binding_ptrs[i])[j]);
					if (j == 3 && dispatch_state.binding_lengths[i] / sizeof(BINDING_DEBUG_TYPE) > 8) j = dispatch_state.binding_lengths[i] / sizeof(BINDING_DEBUG_TYPE) - 5;
				}
			}
#endif
			break;

		case OP_ELF_FREE:
			command_data_connection_read(&command_data, c, &ptr, sizeof(ptr));
			DEBUG_PRINTF("ELF FREE %p\n", ptr);
			nl_elf_executable_destroy(ptr);
			DEBUG_PRINTF("ELF FREED %p\n", ptr);
			break;
#endif
		default:
			fprintf(stderr, "Invalid operation\n");
			command_data_connection_read(&command_data, c, buffer, sizeof(buffer));
			return -1;

	}

	command_data_close(&command_data);
}
