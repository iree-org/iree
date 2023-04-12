/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015-2016 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 */
#ifndef MCA_SPML_BASE_REQUEST_H
#define MCA_SPML_BASE_REQUEST_H

#include "oshmem_config.h"
#include "oshmem/request/request.h" /* TODO: define */

#include "opal/datatype/opal_convertor.h"

#include "opal/class/opal_free_list.h"

BEGIN_C_DECLS

/**
 * External list for the requests. They are declared as lists of
 * the basic request type, which will allow all SPML to overload
 * the list. Beware these free lists have to be initialized
 * directly by the SPML who win the SPML election.
 */
OSHMEM_DECLSPEC extern opal_free_list_t mca_spml_base_put_requests;
OSHMEM_DECLSPEC extern opal_free_list_t mca_spml_base_get_requests;
OSHMEM_DECLSPEC extern opal_free_list_t mca_spml_base_send_requests;
OSHMEM_DECLSPEC extern opal_free_list_t mca_spml_base_recv_requests;
OSHMEM_DECLSPEC extern opal_free_list_t mca_spml_base_atomic_requests;

/* TODO: Consider to add requests lists
 * 1. List of Non blocking requests with NULL handle.
 * 2. List of Non blocking request with Non-NULL handle.
 * 3. List of non completed puts (for small msgs).
 */

/**
 * Types of one sided requests.
 */
typedef enum {
    MCA_SPML_REQUEST_NULL,
    MCA_SPML_REQUEST_PUT, /* Put request */
    MCA_SPML_REQUEST_GET, /* Get Request */
    MCA_SPML_REQUEST_SEND, /* Send Request */
    MCA_SPML_REQUEST_RECV, /* Receive Request */
    MCA_SPML_REQUEST_ATOMIC_CAS, /* Atomic Compare-And-Swap request */
    MCA_SPML_REQUEST_ATOMIC_FAAD /* Atomic Fatch-And-Add request */
} mca_spml_base_request_type_t;

/**
 *  Base type for SPML one sided requests
 */
struct mca_spml_base_request_t {

    oshmem_request_t req_oshmem; /**< base request */
    volatile bool req_spml_complete; /**< flag indicating if the one sided layer is done with this request */
    mca_spml_base_request_type_t req_type; /**< SHMEM request type */
    volatile bool req_free_called; /**< flag indicating if the user has freed this request */
    opal_convertor_t req_convertor; /**< always need the convertor */

    void *req_addr; /**< pointer to application buffer */
    size_t req_count; /**< count of user datatype elements *//* TODO: Need to remove since we are going to remove datatype*/
    int32_t req_peer; /**< peer process - rank of process executing the parallel program */
    ompi_proc_t* req_proc; /**< peer process */
    uint64_t req_sequence; /**< sequence number for shmem one sided ordering */
};
typedef struct mca_spml_base_request_t mca_spml_base_request_t;

OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(mca_spml_base_request_t);

END_C_DECLS

#endif

