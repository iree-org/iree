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
#ifndef MCA_SPML_BASE_GET_REQUEST_H
#define MCA_SPML_BASE_GET_REQUEST_H

#include "oshmem_config.h"
#include "oshmem/mca/spml/base/spml_base_request.h"
#include "ompi/peruse/peruse-internal.h"

BEGIN_C_DECLS

/**
 * Base type for get requests.
 */
struct mca_spml_base_get_request_t {
    mca_spml_base_request_t req_base; /**< base request */
    void *req_addr; /**< pointer to recv buffer on the local PE - not necessarily an application buffer */
    size_t req_bytes_packed; /**< size of message being read */
};
typedef struct mca_spml_base_get_request_t mca_spml_base_get_request_t;
OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(mca_spml_base_get_request_t);

/**
 * Initialize a get request.
 *
 * @param request (IN)         Pointer to the Get request.
 * @param addr (IN)            User buffer.
 * @param count (IN)           Number of bytes.
 * @param peer (IN)            rank w/in the communicator where the data is read from.
 * @param mode (IN)            Get Mode.
 * @param persistent (IN)      Is this a persistent request.
 * @param convertor_flags(IN)
 */
#define MCA_SPML_BASE_GET_REQUEST_INIT( request,                          \
          addr,                             \
          count,                            \
          peer,                             \
          persistent)                  \
 {                                                                         \
     OSHMEM_REQUEST_INIT(&(request)->req_base.req_oshmem, persistent);     \
     (request)->req_addr = addr;                                           \
     (request)->req_base.req_addr = addr;                                  \
     (request)->req_base.req_count = count;                                \
     (request)->req_base.req_peer = (int32_t)peer;                         \
     (request)->req_base.req_spml_complete = OPAL_INT_TO_BOOL(persistent); \
     (request)->req_base.req_free_called = false;                          \
     (request)->req_base.req_oshmem.req_status._cancelled = 0;             \
     (request)->req_bytes_packed = 0;                                      \
}

/**
 *
 *
 */
#define MCA_SPML_BASE_GET_START( request )                                      \
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
 *  Return a Get request. Handle the release of the communicator and the
 *  attached datatype.
 *
 *  @param request (IN)     Get  request.
 */
#define MCA_SPML_BASE_GET_REQUEST_FINI( request )                       \
    do {                                                                \
        OSHMEM_REQUEST_FINI(&(request)->req_base.req_oshmem);               \
        opal_convertor_cleanup( &((request)->req_base.req_convertor) ); \
    } while (0)

END_C_DECLS

#endif

