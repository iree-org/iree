/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 */
#ifndef MCA_SPML_BASE_ATOMIC_REQUEST_H
#define MCA_SPML_BASE_ATOMIC_REQUEST_H

#include "oshmem_config.h"
#include "oshmem/mca/spml/base/spml_base_request.h"
#include "ompi/peruse/peruse-internal.h"

BEGIN_C_DECLS

/**
 * Base type for atomic requests.
 */
struct mca_spml_base_atomic_request_t {
    mca_spml_base_request_t req_base; /**< base request */
    size_t req_bytes_packed; /**< size of virtual heap memory variable operated on */
};
typedef struct mca_spml_base_atomic_request_t mca_spml_base_atomic_request_t;

OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(mca_spml_base_atomic_request_t);

/**
 * Initialize an atomic request with call parameters.
 *
 * @param request (IN)       Atomic request.
 * @param addr (IN)          User buffer.
 * @param count (IN)         Number of bytes.
 * @param src (IN)           Source rank w/in the communicator.
 * @param comm (IN)          Communicator.
 * @param persistent (IN)    Is this a persistent request.
 */

#define MCA_SPML_BASE_ATOMIC_REQUEST_INIT(                                  \
    request,                                                             \
    addr,                                                                \
    count,                                                               \
    src,                                                                 \
    comm,                                                                \
    persistent)                                                          \
{                                                                        \
    /* increment reference count on communicator */                      \
    OBJ_RETAIN(comm);                                                    \
                                                                         \
    OSHMEM_REQUEST_INIT(&(request)->req_base.req_oshmem, persistent);    \
    (request)->req_base.req_oshmem.req_shmem_object.comm = comm;         \
    (request)->req_bytes_packed = 0;                                     \
    (request)->req_base.req_addr = addr;                                 \
    (request)->req_base.req_count = count;                               \
    (request)->req_base.req_peer = src;                                  \
    (request)->req_base.req_comm = comm;                                 \
    (request)->req_base.req_proc = NULL;                                 \
    (request)->req_base.req_sequence = 0;                                \
    /* What about req_type ? */                                          \
    (request)->req_base.req_spml_complete = OPAL_INT_TO_BOOL(persistent); \
    (request)->req_base.req_free_called = false;                         \
}
/**
 *
 *
 */
#define MCA_SPML_BASE_ATOMIC_START( request )                                      \
    do {                                                                        \
        (request)->req_spml_complete = false;                                    \
                                                                                \
        (request)->req_oshmem.req_status.SHMEM_SOURCE = SHMEM_ANY_SOURCE;            \
        (request)->req_oshmem.req_status.SHMEM_ERROR = OSHMEM_SUCCESS;                \
        (request)->req_oshmem.req_status._count = 0;                              \
        (request)->req_oshmem.req_status._cancelled = 0;                          \
                                                                                \
        (request)->req_oshmem.req_complete = false;                               \
        (request)->req_oshmem.req_state = OSHMEM_REQUEST_ACTIVE;                    \
    } while (0)

/**
 *  Return a atomic request. Handle the release of the communicator and the
 *  attached datatype.
 *
 *  @param request (IN)     Get  request.
 */
#define MCA_SPML_BASE_ATOMIC_REQUEST_FINI( request )                       \
    do {                                                                \
        OSHMEM_REQUEST_FINI(&(request)->req_base.req_oshmem);               \
        OBJ_RELEASE( (request)->req_base.req_comm);                     \
        opal_convertor_cleanup( &((request)->req_base.req_convertor) ); \
    } while (0)

END_C_DECLS

#endif

