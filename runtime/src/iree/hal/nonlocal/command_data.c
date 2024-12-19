#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <poll.h>
#include <netinet/tcp.h>

#include "command_data.h"

// Only debug in this file if debug is > 1
#if defined(DEBUG) && DEBUG < 2
#undef DEBUG
#endif

#include "debug.h"

int command_data_init(command_data_t *command_data, int port) {
	struct sockaddr_in address;
	int opt = 1;

	if ((command_data->fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
		perror("socket failed");
		return -1;
	}

	if (setsockopt(command_data->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
		perror("setsockopt");
		return -1;
	}

	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(COMMAND_DATA_PORT);

	if (bind(command_data->fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
		perror("bind failed");
		return -1;
	}

	if (listen(command_data->fd, 3) < 0) {
		perror("listen");
		return -1;
	}
	command_data->connection_count = 0;

	command_data->fds[0].fd = command_data->fd;
	command_data->fds[0].events = POLLIN;

	return 0;
}

int command_data_open(command_data_t *command_data, char *server, int port) {
	struct sockaddr_in serv_addr;

	if ((command_data->fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		fprintf(stderr, "\n Socket creation error \n");
		return -1;
	}

	int one = 1;
	if(setsockopt(command_data->fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
		perror("setsockopt NODELAY");
		return -1;
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);

	if (inet_pton(AF_INET, server, &serv_addr.sin_addr) <= 0) {
		fprintf(stderr, "\nInvalid address/ Address not supported \n");
		return -1;
	}

	if (connect(command_data->fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
		fprintf(stderr, "\nConnection Failed \n");
		return -1;
	}
	command_data->connection_count = 0;

	return 0;
}

int command_data_read(command_data_t *command_data, void *buffer, int size) {
	DEBUG_PRINTF("read %d\n", size);
	return recv(command_data->fd, buffer, size, 0);
}

int command_data_read_full(command_data_t *command_data, void *buffer, int size) {
	int remain = size;
	int n;
	do {
		n = recv(command_data->fd, buffer, remain, 0);
		if ( n <= 0 ) {
			perror("recv");
			close(command_data->fd);
			command_data->fd = -1;
			return -1;
		}
		remain -= n;
		buffer = (unsigned char *) buffer + n;
	} while (remain);
	return size;
}

int command_data_write(command_data_t *command_data, const void *buffer, int size) {
	DEBUG_PRINTF("write %d\n", size);
	return send(command_data->fd, buffer, size, 0);
}

int command_data_connection_count(command_data_t *command_data) {
	return command_data->connection_count;
}

int command_data_process_connections(command_data_t *command_data) {
	struct sockaddr_in address;
	int new_socket;

	// cleanup closed fd's
	int connection_count = command_data->connection_count;
	for (int i = 0, j = 0; i < connection_count; i++) {
		if(command_data->fds[i + 1].fd < 0) {
			command_data->connection_count--;
			continue;
		}
		if(j != i) command_data->fds[j+1] = command_data->fds[i+1];
		j++;
	}

	int poll_count = poll(command_data->fds, command_data->connection_count + 1, -1);
	if (poll_count < 0) {
		perror("poll error");
		return -1;
	}

	if (command_data->fds[0].revents & POLLIN) {
		socklen_t addrlen = sizeof(address);
		if ((new_socket = accept(command_data->fd, (struct sockaddr *)&address, &addrlen)) < 0) {
			perror("accept");
			return -1;
		}

		DEBUG_PRINTF("New connection: socket fd is %d, ip is : %s, port : %d\n", new_socket, inet_ntoa(address.sin_addr), ntohs(address.sin_port));

		if (command_data->connection_count < MAX_CLIENTS) {
			command_data->fds[command_data->connection_count + 1].fd = new_socket;
			command_data->fds[command_data->connection_count + 1].events = POLLIN;
			command_data->connection_count++;
		} else {
			fprintf(stderr, "Max clients reached. Connection rejected.\n");
			close(new_socket);
			return -1;
		}
	}
	return 0;
}

int command_data_connection_read(command_data_t *command_data, int i, void *buffer, int size) {
	i++;

	if(!(command_data->fds[i].revents & POLLIN)) return 0;
	DEBUG_PRINTF("connection %d read %d\n", i, size);

	int n = recv(command_data->fds[i].fd, buffer, size, 0);
	if ( n <= 0 ) {
		close(command_data->fds[i].fd);
		command_data->fds[i].fd = -1;
	}
	return n;
}

int command_data_connection_read_full(command_data_t *command_data, int i, void *buffer, int size) {
	i++;

	if(!(command_data->fds[i].revents & POLLIN)) return 0;
	DEBUG_PRINTF("connection %d read %d\n", i, size);

	int remain = size;
	int n;
	do {
		n = recv(command_data->fds[i].fd, buffer, remain, 0);
		if ( n <= 0 ) {
			perror("recv");
			close(command_data->fds[i].fd);
			command_data->fds[i].fd = -1;
			return -1;
		}
		remain -= n;
		buffer = (unsigned char *)buffer + n;
	} while (remain);
	return size;
}

int command_data_connection_write(command_data_t *command_data, int i, const void *buffer, int size) {
	i++;

	DEBUG_PRINTF("connection %d write %d\n", i, size);
	int n = send(command_data->fds[i].fd, buffer, size, 0);

	if ( n <= 0 ) {
		close(command_data->fds[i].fd);
		command_data->fds[i].fd = -1;
	}
	return n;
}

int command_data_close(command_data_t *command_data) {
	for (int i = 0; i < command_data->connection_count; i++) {
		close(command_data->fds[i].fd);
	}
	return close(command_data->fd);
}

