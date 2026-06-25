#include <poll.h>

#define MAX_CLIENTS 4
#define COMMAND_DATA_PORT 9090

typedef struct {
	int fd;
	struct pollfd fds[MAX_CLIENTS + 1];
	int connection_count;
} command_data_t;

int command_data_init(command_data_t *command_data, int port);
int command_data_open(command_data_t *command_data, char *server, int port);
int command_data_read(command_data_t *command_data, void *buffer, int size);
int command_data_read_full(command_data_t *command_data, void *buffer, int size);
int command_data_write(command_data_t *command_data, const void *buffer, int size);
int command_data_connection_count(command_data_t *command_data);
int command_data_process_connections(command_data_t *command_data);
int command_data_connection_read(command_data_t *command_data, int i, void *buffer, int size);
int command_data_connection_read_full(command_data_t *command_data, int i, void *buffer, int size);
int command_data_connection_write(command_data_t *command_data, int i, const void *buffer, int size);
int command_data_close(command_data_t *command_data);

