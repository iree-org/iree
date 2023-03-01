// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KVS_H
#define KVS_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  KVS_STATUS_OK = 0,
  KVS_STATUS_DEADLINE_EXCEEDED = 1,
  KVS_STATUS_INVALID_ARGUMENT = 2,
  KVS_STATUS_INTERNAL_ERROR = 3,
  KVS_STATUS_IN_PROGRESS = 4,
  KVS_STATUS_INVALID_USAGE = 5,
  KVS_STATUS_SYSTEM_ERROR = 6,
  KVS_STATUS_SERVER_ERROR = 7,
  KVS_STATUS_CONNECTION_ERROR = 8,
  KVS_STATUS_NUM = 9,
} kvs_status_t;

typedef struct {
  long long connection_timeout_ms; /* timeout for connection */
} kvs_client_config_t;

/* These are a C wrapper for the KVS client and server. */
typedef struct kvs_client_t kvs_client_t;

kvs_status_t kvs_client_create(kvs_client_t** kvs_client, const char* addr,
                               kvs_client_config_t* config);

kvs_status_t kvs_client_destroy(kvs_client_t** kvs_client);

kvs_status_t kvs_client_get(kvs_client_t* kvs_client, const char* key,
                            int key_len, char* value, int value_len);

kvs_status_t kvs_client_set(kvs_client_t* kvs_client, const char* key,
                            int key_len, const char* value, int value_len);

typedef struct {
  long long timeout_ms; /* timeout for kvs_get and kvs_set */
} kvs_server_config_t;

typedef struct kvs_server_t kvs_server_t;

kvs_status_t kvs_server_create(kvs_server_t** kvs_server, const char* addr,
                               kvs_server_config_t* config);

kvs_status_t kvs_server_wait(kvs_server_t* kvs_server);

kvs_status_t kvs_server_destroy(kvs_server_t** kvs_server);

#ifdef __cplusplus
}  // end extern "C"
#endif  // __cplusplus

#endif  // KVS_H
