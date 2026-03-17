// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// libibverbs symbol table for dynamic loading.
// This file is included multiple times with different macro definitions.
//
// See: https://github.com/linux-rdma/rdma-core/tree/master/libibverbs

//===----------------------------------------------------------------------===//
// Device Management
//===----------------------------------------------------------------------===//

IREE_NET_LIBVERBS_PFN(struct ibv_device**, ibv_get_device_list,
                      DECL(int* num_devices), ARGS(num_devices))
IREE_NET_LIBVERBS_PFN(void, ibv_free_device_list,
                      DECL(struct ibv_device** list), ARGS(list))
IREE_NET_LIBVERBS_PFN(struct ibv_context*, ibv_open_device,
                      DECL(struct ibv_device* device), ARGS(device))
IREE_NET_LIBVERBS_PFN(int, ibv_close_device, DECL(struct ibv_context* context),
                      ARGS(context))
IREE_NET_LIBVERBS_PFN(const char*, ibv_get_device_name,
                      DECL(struct ibv_device* device), ARGS(device))
IREE_NET_LIBVERBS_PFN(int, ibv_query_device,
                      DECL(struct ibv_context* context,
                           struct ibv_device_attr* device_attr),
                      ARGS(context, device_attr))
IREE_NET_LIBVERBS_PFN(int, ibv_query_port,
                      DECL(struct ibv_context* context, uint8_t port_number,
                           struct ibv_port_attr* port_attr),
                      ARGS(context, port_number, port_attr))
IREE_NET_LIBVERBS_PFN(int, ibv_query_gid,
                      DECL(struct ibv_context* context, uint8_t port_number,
                           int index, union ibv_gid* gid),
                      ARGS(context, port_number, index, gid))

//===----------------------------------------------------------------------===//
// Protection Domains
//===----------------------------------------------------------------------===//

IREE_NET_LIBVERBS_PFN(struct ibv_pd*, ibv_alloc_pd,
                      DECL(struct ibv_context* context), ARGS(context))
IREE_NET_LIBVERBS_PFN(int, ibv_dealloc_pd, DECL(struct ibv_pd* pd), ARGS(pd))

//===----------------------------------------------------------------------===//
// Memory Registration
//===----------------------------------------------------------------------===//

IREE_NET_LIBVERBS_PFN(struct ibv_mr*, ibv_reg_mr,
                      DECL(struct ibv_pd* pd, void* addr, size_t length,
                           int access),
                      ARGS(pd, addr, length, access))
IREE_NET_LIBVERBS_PFN(int, ibv_dereg_mr, DECL(struct ibv_mr* mr), ARGS(mr))

//===----------------------------------------------------------------------===//
// Completion Queues
//===----------------------------------------------------------------------===//

// NOTE: ibv_poll_cq and ibv_req_notify_cq are inline functions that dispatch
// through device-specific ops (cq->context->ops->poll_cq). Use them directly
// from the header rather than trying to load them dynamically.

IREE_NET_LIBVERBS_PFN(struct ibv_comp_channel*, ibv_create_comp_channel,
                      DECL(struct ibv_context* context), ARGS(context))
IREE_NET_LIBVERBS_PFN(int, ibv_destroy_comp_channel,
                      DECL(struct ibv_comp_channel* channel), ARGS(channel))
IREE_NET_LIBVERBS_PFN(struct ibv_cq*, ibv_create_cq,
                      DECL(struct ibv_context* context, int cqe,
                           void* cq_context, struct ibv_comp_channel* channel,
                           int comp_vector),
                      ARGS(context, cqe, cq_context, channel, comp_vector))
IREE_NET_LIBVERBS_PFN(int, ibv_destroy_cq, DECL(struct ibv_cq* cq), ARGS(cq))
IREE_NET_LIBVERBS_PFN(int, ibv_get_cq_event,
                      DECL(struct ibv_comp_channel* channel, struct ibv_cq** cq,
                           void** cq_context),
                      ARGS(channel, cq, cq_context))
IREE_NET_LIBVERBS_PFN(void, ibv_ack_cq_events,
                      DECL(struct ibv_cq* cq, unsigned int nevents),
                      ARGS(cq, nevents))

//===----------------------------------------------------------------------===//
// Queue Pairs
//===----------------------------------------------------------------------===//

IREE_NET_LIBVERBS_PFN(struct ibv_qp*, ibv_create_qp,
                      DECL(struct ibv_pd* pd,
                           struct ibv_qp_init_attr* qp_init_attr),
                      ARGS(pd, qp_init_attr))
IREE_NET_LIBVERBS_PFN(int, ibv_destroy_qp, DECL(struct ibv_qp* qp), ARGS(qp))
IREE_NET_LIBVERBS_PFN(int, ibv_modify_qp,
                      DECL(struct ibv_qp* qp, struct ibv_qp_attr* attr,
                           int attr_mask),
                      ARGS(qp, attr, attr_mask))
IREE_NET_LIBVERBS_PFN(int, ibv_query_qp,
                      DECL(struct ibv_qp* qp, struct ibv_qp_attr* attr,
                           int attr_mask, struct ibv_qp_init_attr* init_attr),
                      ARGS(qp, attr, attr_mask, init_attr))

//===----------------------------------------------------------------------===//
// Data Transfer
//===----------------------------------------------------------------------===//

// NOTE: ibv_post_send and ibv_post_recv are inline functions that dispatch
// through device-specific ops (qp->context->ops->post_send). Use them directly
// from the header rather than trying to load them dynamically.

#undef IREE_NET_LIBVERBS_PFN
#undef DECL
#undef ARGS
