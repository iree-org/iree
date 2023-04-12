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
#ifndef MCA_SPML_BASE_PUT_REQUEST_H
#define MCA_SPML_BASE_PUT_REQUEST_H

#include "oshmem_config.h"
#include "oshmem/mca/spml/spml.h"
#include "oshmem/mca/spml/base/spml_base_request.h"
#include "ompi/peruse/peruse-internal.h"

BEGIN_C_DECLS

/**
 * Base type for send requests
 */
struct mca_spml_base_put_request_t {
    mca_spml_base_request_t req_base; /**< base request type - common data structure for use by wait/test */
    void *req_addr; /**< pointer to send buffer - may not be application buffer */
    size_t req_bytes_packed; /**< packed size of a message given the datatype and count */
};
typedef struct mca_spml_base_put_request_t mca_spml_base_put_request_t;

OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION( mca_spml_base_put_request_t);

/**
 * Initialize a send request with call parameters.
 *
 * @param request (IN)         Send request
 * @param addr (IN)            User buffer
 * @param count (IN)           Number of bytes.
 * @param peer (IN)            Destination rank
 * @param comm (IN)            Communicator
 * @param mode (IN)            Send mode (STANDARD,BUFFERED,SYNCHRONOUS,READY)
 * @param persistent (IN)      Is request persistent.
 * @param convertor_flags (IN) Flags to pass to convertor
 *
 * Perform a any one-time initialization. Note that per-use initialization
 * is done in the send request start routine.
 */

#define MCA_SPML_BASE_PUT_REQUEST_INIT( request,                          \
                                        addr,                             \
                                        count,                            \
                                        peer,                             \
                                        persistent)                  \
   {                                                                      \
      OSHMEM_REQUEST_INIT(&(request)->req_base.req_oshmem, persistent);       \
      (request)->req_addr = addr;                                         \
      (request)->req_base.req_addr = addr;                                \
      (request)->req_base.req_count = count;                              \
      (request)->req_base.req_peer = (int32_t)peer;                       \
      (request)->req_base.req_spml_complete = OPAL_INT_TO_BOOL(persistent); \
      (request)->req_base.req_free_called = false;                        \
      (request)->req_base.req_oshmem.req_status._cancelled = 0;             \
      (request)->req_bytes_packed = 0;                                    \
                                                                          \
   }

/**
 * Mark the request as started from the SPML base point of view.
 *
 *  @param request (IN)    The put request.
 */

#define MCA_SPML_BASE_PUT_START( request )                    \
    do {                                                      \
        (request)->req_spml_complete = false;                  \
        (request)->req_oshmem.req_complete = false;             \
        (request)->req_oshmem.req_state = OSHMEM_REQUEST_ACTIVE;  \
        (request)->req_oshmem.req_status._cancelled = 0;        \
    } while (0)

/**
 *  Release the ref counts on the communicator and datatype.
 *
 *  @param request (IN)    The put request.
 */

#define MCA_SPML_BASE_PUT_REQUEST_FINI( request )                         \
    do {                                                                  \
        OSHMEM_REQUEST_FINI(&(request)->req_base.req_oshmem);                 \
        opal_convertor_cleanup( &((request)->req_base.req_convertor) );   \
    } while (0)

END_C_DECLS

#endif

