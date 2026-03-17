// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// librdmacm symbol table for dynamic loading.
// This file is included multiple times with different macro definitions.
//
// See: https://github.com/linux-rdma/rdma-core/tree/master/librdmacm

//===----------------------------------------------------------------------===//
// Event Channels
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(struct rdma_event_channel*, rdma_create_event_channel,
                       DECL(void), ARGS())
IREE_NET_LIBRDMACM_PFN(void, rdma_destroy_event_channel,
                       DECL(struct rdma_event_channel* channel), ARGS(channel))

//===----------------------------------------------------------------------===//
// Connection IDs
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(int, rdma_create_id,
                       DECL(struct rdma_event_channel* channel,
                            struct rdma_cm_id** id, void* context,
                            enum rdma_port_space ps),
                       ARGS(channel, id, context, ps))
IREE_NET_LIBRDMACM_PFN(int, rdma_destroy_id, DECL(struct rdma_cm_id* id),
                       ARGS(id))

//===----------------------------------------------------------------------===//
// Address/Route Resolution
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(int, rdma_resolve_addr,
                       DECL(struct rdma_cm_id* id, struct sockaddr* src_addr,
                            struct sockaddr* dst_addr, int timeout_ms),
                       ARGS(id, src_addr, dst_addr, timeout_ms))
IREE_NET_LIBRDMACM_PFN(int, rdma_resolve_route,
                       DECL(struct rdma_cm_id* id, int timeout_ms),
                       ARGS(id, timeout_ms))

//===----------------------------------------------------------------------===//
// Connection Management
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(int, rdma_connect,
                       DECL(struct rdma_cm_id* id,
                            struct rdma_conn_param* conn_param),
                       ARGS(id, conn_param))
IREE_NET_LIBRDMACM_PFN(int, rdma_accept,
                       DECL(struct rdma_cm_id* id,
                            struct rdma_conn_param* conn_param),
                       ARGS(id, conn_param))
IREE_NET_LIBRDMACM_PFN(int, rdma_disconnect, DECL(struct rdma_cm_id* id),
                       ARGS(id))
IREE_NET_LIBRDMACM_PFN(int, rdma_listen,
                       DECL(struct rdma_cm_id* id, int backlog),
                       ARGS(id, backlog))
IREE_NET_LIBRDMACM_PFN(int, rdma_bind_addr,
                       DECL(struct rdma_cm_id* id, struct sockaddr* addr),
                       ARGS(id, addr))
IREE_NET_LIBRDMACM_PFN(int, rdma_reject,
                       DECL(struct rdma_cm_id* id, const void* private_data,
                            uint8_t private_data_len),
                       ARGS(id, private_data, private_data_len))

//===----------------------------------------------------------------------===//
// Event Handling
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(int, rdma_get_cm_event,
                       DECL(struct rdma_event_channel* channel,
                            struct rdma_cm_event** event),
                       ARGS(channel, event))
IREE_NET_LIBRDMACM_PFN(int, rdma_ack_cm_event,
                       DECL(struct rdma_cm_event* event), ARGS(event))

//===----------------------------------------------------------------------===//
// Address Info
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(int, rdma_getaddrinfo,
                       DECL(const char* node, const char* service,
                            const struct rdma_addrinfo* hints,
                            struct rdma_addrinfo** res),
                       ARGS(node, service, hints, res))
IREE_NET_LIBRDMACM_PFN(void, rdma_freeaddrinfo, DECL(struct rdma_addrinfo* res),
                       ARGS(res))

//===----------------------------------------------------------------------===//
// Queue Pair Creation (convenience wrappers)
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(int, rdma_create_qp,
                       DECL(struct rdma_cm_id* id, struct ibv_pd* pd,
                            struct ibv_qp_init_attr* qp_init_attr),
                       ARGS(id, pd, qp_init_attr))
IREE_NET_LIBRDMACM_PFN(void, rdma_destroy_qp, DECL(struct rdma_cm_id* id),
                       ARGS(id))

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

IREE_NET_LIBRDMACM_PFN(int, rdma_set_option,
                       DECL(struct rdma_cm_id* id, int level, int optname,
                            void* optval, size_t optlen),
                       ARGS(id, level, optname, optval, optlen))
IREE_NET_LIBRDMACM_PFN(int, rdma_migrate_id,
                       DECL(struct rdma_cm_id* id,
                            struct rdma_event_channel* channel),
                       ARGS(id, channel))

#undef IREE_NET_LIBRDMACM_PFN
#undef DECL
#undef ARGS
