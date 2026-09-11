#include <stdio.h>
#include <inttypes.h>

#include "command_data.h"
#include "command_op.h"

int main() {
	command_data_t command_data;

	command_data_open(&command_data, "127.0.0.1", COMMAND_DATA_PORT);

	command_value_t command[4];
	command_value_t n;
	command_value_t n2;
	command_ptr_t p;
	command_ptr_t p2;

	command[0] = OP_DEVICE_ALLOC;
	command[1] = sizeof(n);
	command_data_write(&command_data, &command, 2 * sizeof(command_value_t));
	command_data_read(&command_data, &p, sizeof(p));

	command[0] = OP_DEVICE_ALLOC;
	command[1] = sizeof(n);
	command_data_write(&command_data, &command, 2 * sizeof(command_value_t));
	command_data_read(&command_data, &p2, sizeof(p2));

	command[0] = OP_COPY_IN;
	command[1] = (command_value_t)p;
	command[2] = sizeof(n);
	command[3] = 12345;
	command_data_write(&command_data, command, 4 * sizeof(command_value_t));

	command[0] = OP_COPY;
	command[1] = (command_value_t)p;
	command[2] = (command_value_t)p2;
	command[3] = sizeof(n);
	command_data_write(&command_data, command, 4 * sizeof(command_value_t));

	command[0] = OP_COPY_OUT;
	command[1] = (command_value_t)p;
	command[2] = sizeof(n);
	command_data_write(&command_data, command, 3 * sizeof(command_value_t));
	command_data_read(&command_data, &n, sizeof(n));

	command[0] = OP_COPY_OUT;
	command[1] = (command_value_t)p2;
	command[2] = sizeof(n2);
	command_data_write(&command_data, command, 3 * sizeof(command_value_t));
	command_data_read(&command_data, &n2, sizeof(n2));

	command[0] = OP_DEVICE_FREE;
	command[1] = (command_value_t)p;
	command_data_write(&command_data, &command, 2 * sizeof(command_value_t));

	command[0] = OP_DEVICE_FREE;
	command[1] = (command_value_t)p2;
	command_data_write(&command_data, &command, 2 * sizeof(command_value_t));

	printf("p = %p\n", p);
	printf("n = %" PRId64 "\n", n);
	printf("n2 = %" PRId64 "\n", n2);

	command_data_close(&command_data);

	return 0;
}

