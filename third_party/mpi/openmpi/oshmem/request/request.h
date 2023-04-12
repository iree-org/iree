/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 *
 * Top-level description of requests
 */

#ifndef OSHMEM_REQUEST_H
#define OSHMEM_REQUEST_H

#include "oshmem_config.h"
#include "oshmem/constants.h"

#include "opal/class/opal_free_list.h"
#include "opal/class/opal_pointer_array.h"
#include "opal/threads/condition.h"

BEGIN_C_DECLS

/**
 * Request class
 */
/*OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(oshmem_request_t);*/
OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(oshmem_request_t);

/*
 * The following include pulls in shared typedefs with debugger plugins.
 * For more information on why we do this see the Notice to developers
 * comment at the top of the oshmem_msgq_dll.c file.
 */

#include "request_dbg.h"

struct oshmem_request_t;

typedef struct oshmem_request_t *SHMEM_Request;
typedef struct oshmem_status_public_t SHMEM_Status;

/* This constants are used to check status of  request->req_status.SHMEM_ERROR */
#define SHMEM_SUCCESS                   0
#define SHMEM_ERR_IN_STATUS             18

/*
 * SHMEM_Status
 */
struct oshmem_status_public_t {
    int SHMEM_SOURCE;
    /*int MPI_TAG;*/
    int SHMEM_ERROR;
    int _count;
    int _cancelled;
};
typedef struct oshmem_status_public_t oshmem_status_public_t;

typedef int (SHMEM_Grequest_query_function)(void *, SHMEM_Status *);
typedef int (SHMEM_Grequest_free_function)(void *);
typedef int (SHMEM_Grequest_cancel_function)(void *, int);

#define SHMEM_REQUEST_NULL OSHMEM_PREDEFINED_GLOBAL(SHMEM_Request, oshmem_request_null)

/*
 * Required function to free the request and any associated resources.
 */
typedef int (*oshmem_request_free_fn_t)(struct oshmem_request_t** rptr);

/*
 * Optional function to cancel a pending request.
 */
typedef int (*oshmem_request_cancel_fn_t)(struct oshmem_request_t* request,
                                          int flag);

/*
 * Optional function called when the request is completed from the SHMEM
 * library perspective. This function is not allowed to release any
 * ressources related to the request.
 */
typedef int (*oshmem_request_complete_fn_t)(struct oshmem_request_t* request);

/* TODO: decide whether to remove comm */
/**
 * Forward declaration
 */
struct oshmem_group_t;

/**
 * Forward declaration
 */
/*struct oshmem_file_t;*/

/**
 * Union for holding several different SHMEM pointer types on the request
 */
typedef union oshmem_shmem_object_t {
    struct oshmem_group_t *comm;
/*    struct oshmem_file_t *file;*/
} oshmem_shmem_object_t;

/**
 * Main top-level request struct definition
 */
struct oshmem_request_t {
    opal_free_list_item_t super; /**< Base type *//*TODO: Implement in shmem */
    oshmem_request_type_t req_type; /**< Enum indicating the type of the request */
    oshmem_status_public_t req_status; /**< Completion status */
    volatile bool req_complete; /**< Flag indicating completion on a request */
    volatile oshmem_request_state_t req_state; /**< enum indicate the state of the request */
    bool req_persistent; /* TODO: NOT Required */
    /**< flag indicating if this is a persistent request */
    int req_f_to_c_index; /* TODO: NOT Required */
    /**< Index in Fortran <-> C translation array */
    oshmem_request_free_fn_t req_free; /**< Called by free */
    oshmem_request_cancel_fn_t req_cancel; /* TODO: Not Required */
    /**< Optional function to cancel the request */
    oshmem_request_complete_fn_t req_complete_cb; /**< Called when the request is SHMEM completed */
    void *req_complete_cb_data;
    oshmem_shmem_object_t req_shmem_object; /**< Pointer to SHMEM object that created this request */
};

/**
 * Convenience typedef
 */
typedef struct oshmem_request_t oshmem_request_t;

/**
 * Padded struct to maintain back compatibiltiy.
 * See oshmem/communicator/communicator.h comments with struct oshmem_group_t
 * for full explanation why we chose the following padding construct for predefines.
 */
#define PREDEFINED_REQUEST_PAD 256

struct oshmem_predefined_request_t {
    struct oshmem_request_t request;
    char padding[PREDEFINED_REQUEST_PAD - sizeof(oshmem_request_t)];
};

typedef struct oshmem_predefined_request_t oshmem_predefined_request_t;

/**
 * Initialize a request.  This is a macro to avoid function call
 * overhead, since this is typically invoked in the critical
 * performance path (since requests may be re-used, it is possible
 * that we will have to initialize a request multiple times).
 */
#define OSHMEM_REQUEST_INIT(request, persistent)        \
    do {                                              \
        (request)->req_complete = false;              \
        (request)->req_state = OSHMEM_REQUEST_INACTIVE; \
        (request)->req_persistent = (persistent);     \
    } while (0);

/**
 * Finalize a request.  This is a macro to avoid function call
 * overhead, since this is typically invoked in the critical
 * performance path (since requests may be re-used, it is possible
 * that we will have to finalize a request multiple times).
 *
 * When finalizing a request, if MPI_Request_f2c() was previously
 * invoked on that request, then this request was added to the f2c
 * table, and we need to remove it
 *
 * This function should be called only from the SHMEM layer. It should
 * never be called from the SPML. It take care of the upper level clean-up.
 * When the user call MPI_Request_free we should release all SHMEM level
 * ressources, so we have to call this function too.
 */
#define OSHMEM_REQUEST_FINI(request)                                      \
do {                                                                    \
    (request)->req_state = OSHMEM_REQUEST_INVALID;                        \
    if (SHMEM_UNDEFINED != (request)->req_f_to_c_index) {                 \
        opal_pointer_array_set_item(&oshmem_request_f_to_c_table,         \
                                    (request)->req_f_to_c_index, NULL); \
        (request)->req_f_to_c_index = SHMEM_UNDEFINED;                    \
    }                                                                   \
} while (0);

/**
 * Non-blocking test for request completion.
 *
 * @param request (IN)   Array of requests
 * @param complete (OUT) Flag indicating if index is valid (a request completed).
 * @param status (OUT)   Status of completed request.
 * @return               OSHMEM_SUCCESS or failure status.
 *
 * Note that upon completion, the request is freed, and the
 * request handle at index set to NULL.
 */
typedef int (*oshmem_request_test_fn_t)(oshmem_request_t ** rptr,
                                        int *completed,
                                        oshmem_status_public_t * status);
/**
 * Non-blocking test for request completion.
 *
 * @param count (IN)     Number of requests
 * @param request (IN)   Array of requests
 * @param index (OUT)    Index of first completed request.
 * @param complete (OUT) Flag indicating if index is valid (a request completed).
 * @param status (OUT)   Status of completed request.
 * @return               OSHMEM_SUCCESS or failure status.
 *
 * Note that upon completion, the request is freed, and the
 * request handle at index set to NULL.
 */
typedef int (*oshmem_request_test_any_fn_t)(size_t count,
                                            oshmem_request_t ** requests,
                                            int *index,
                                            int *completed,
                                            oshmem_status_public_t * status);
/**
 * Non-blocking test for request completion.
 *
 * @param count (IN)      Number of requests
 * @param requests (IN)   Array of requests
 * @param completed (OUT) Flag indicating wether all requests completed.
 * @param statuses (OUT)  Array of completion statuses.
 * @return                OSHMEM_SUCCESS or failure status.
 *
 * This routine returns completed==true if all requests have completed.
 * The statuses parameter is only updated if all requests completed. Likewise,
 * the requests array is not modified (no requests freed), unless all requests
 * have completed.
 */
typedef int (*oshmem_request_test_all_fn_t)(size_t count,
                                            oshmem_request_t ** requests,
                                            int *completed,
                                            oshmem_status_public_t * statuses);
/**
 * Non-blocking test for some of N requests to complete.
 *
 * @param count (IN)        Number of requests
 * @param requests (INOUT)  Array of requests
 * @param outcount (OUT)    Number of finished requests
 * @param indices (OUT)     Indices of the finished requests
 * @param statuses (OUT)    Array of completion statuses.
 * @return                  OSHMEM_SUCCESS, OSHMEM_ERR_IN_STATUS or failure status.
 *
 */
typedef int (*oshmem_request_test_some_fn_t)(size_t count,
                                             oshmem_request_t ** requests,
                                             int * outcount,
                                             int * indices,
                                             oshmem_status_public_t * statuses);
/**
 * Wait (blocking-mode) for one requests to complete.
 *
 * @param request (IN)    Pointer to request.
 * @param status (OUT)    Status of completed request.
 * @return                OSHMEM_SUCCESS or failure status.
 *
 */
typedef int (*oshmem_request_wait_fn_t)(oshmem_request_t ** req_ptr,
                                        oshmem_status_public_t * status);
/**
 * Wait (blocking-mode) for one of N requests to complete.
 *
 * @param count (IN)      Number of requests
 * @param requests (IN)   Array of requests
 * @param index (OUT)     Index into request array of completed request.
 * @param status (OUT)    Status of completed request.
 * @return                OSHMEM_SUCCESS or failure status.
 *
 */
typedef int (*oshmem_request_wait_any_fn_t)(size_t count,
                                            oshmem_request_t ** requests,
                                            int *index,
                                            oshmem_status_public_t * status);
/**
 * Wait (blocking-mode) for all of N requests to complete.
 *
 * @param count (IN)      Number of requests
 * @param requests (IN)   Array of requests
 * @param statuses (OUT)  Array of completion statuses.
 * @return                OSHMEM_SUCCESS or failure status.
 *
 */
typedef int (*oshmem_request_wait_all_fn_t)(size_t count,
                                            oshmem_request_t ** requests,
                                            oshmem_status_public_t * statuses);
/**
 * Wait (blocking-mode) for some of N requests to complete.
 *
 * @param count (IN)        Number of requests
 * @param requests (INOUT)  Array of requests
 * @param outcount (OUT)    Number of finished requests
 * @param indices (OUT)     Indices of the finished requests
 * @param statuses (OUT)    Array of completion statuses.
 * @return                  OSHMEM_SUCCESS, OSHMEM_ERR_IN_STATUS or failure status.
 *
 */
typedef int (*oshmem_request_wait_some_fn_t)(size_t count,
                                             oshmem_request_t ** requests,
                                             int * outcount,
                                             int * indices,
                                             oshmem_status_public_t * statuses);

/**
 * Replaceable request functions
 */
typedef struct oshmem_request_fns_t {
    oshmem_request_test_fn_t req_test;
    oshmem_request_test_any_fn_t req_test_any;
    oshmem_request_test_all_fn_t req_test_all;
    oshmem_request_test_some_fn_t req_test_some;
    oshmem_request_wait_fn_t req_wait;
    oshmem_request_wait_any_fn_t req_wait_any;
    oshmem_request_wait_all_fn_t req_wait_all;
    oshmem_request_wait_some_fn_t req_wait_some;
} oshmem_request_fns_t;

/**
 * Globals used for tracking requests and request completion.
 */
OSHMEM_DECLSPEC extern opal_pointer_array_t oshmem_request_f_to_c_table;
OSHMEM_DECLSPEC extern size_t oshmem_request_waiting;
OSHMEM_DECLSPEC extern size_t oshmem_request_completed;
OSHMEM_DECLSPEC extern int32_t oshmem_request_poll;
OSHMEM_DECLSPEC extern opal_mutex_t oshmem_request_lock;
OSHMEM_DECLSPEC extern opal_condition_t oshmem_request_cond;
OSHMEM_DECLSPEC extern oshmem_predefined_request_t oshmem_request_null;
OSHMEM_DECLSPEC extern oshmem_request_t oshmem_request_empty;
OSHMEM_DECLSPEC extern oshmem_status_public_t oshmem_status_empty;
OSHMEM_DECLSPEC extern oshmem_request_fns_t oshmem_request_functions;

/**
 * Initialize the OSHMEM_Request subsystem; invoked during SHMEM_INIT.
 */
int oshmem_request_init(void);

/**
 * Free a persistent request to a MPI_PROC_NULL peer (there's no
 * freelist to put it back to, so we have to actually OBJ_RELEASE it).
 */
OSHMEM_DECLSPEC int oshmem_request_persistent_proc_null_free(oshmem_request_t **request);

/**
 * Shut down the SHMEM_Request subsystem; invoked during SHMEM_FINALIZE.
 */
int oshmem_request_finalize(void);

/**
 * Cancel a pending request.
 */
static inline int oshmem_request_cancel(oshmem_request_t* request)
{
    if (request->req_cancel != NULL ) {
        return request->req_cancel(request, true);
    }
    return OSHMEM_SUCCESS;
}

/**
 * Free a request.
 *
 * @param request (INOUT)   Pointer to request.
 */
static inline int oshmem_request_free(oshmem_request_t** request)
{
    return (*request)->req_free(request);
}

#define oshmem_request_test       (oshmem_request_functions.req_test)
#define oshmem_request_test_any   (oshmem_request_functions.req_test_any)
#define oshmem_request_test_all   (oshmem_request_functions.req_test_all)
#define oshmem_request_test_some  (oshmem_request_functions.req_test_some)
#define oshmem_request_wait       (oshmem_request_functions.req_wait)
#define oshmem_request_wait_any   (oshmem_request_functions.req_wait_any)
#define oshmem_request_wait_all   (oshmem_request_functions.req_wait_all)
#define oshmem_request_wait_some  (oshmem_request_functions.req_wait_some)

/**
 * Wait for any completion. It is a caller responsibility to check for
 * condition and call us again if needed.
 */
static inline void oshmem_request_wait_any_completion(void)
{
    OPAL_THREAD_LOCK(&oshmem_request_lock);
    oshmem_request_waiting++;
    opal_condition_wait(&oshmem_request_cond, &oshmem_request_lock);
    oshmem_request_waiting--;
    OPAL_THREAD_UNLOCK(&oshmem_request_lock);
}

/**
 * Wait a particular request for completion
 */
static inline void oshmem_request_wait_completion(oshmem_request_t *req)
{
    if (false == req->req_complete) {
#if OPAL_ENABLE_PROGRESS_THREADS
        if(opal_progress_spin(&req->req_complete)) {
            return;
        }
#endif
        OPAL_THREAD_LOCK(&oshmem_request_lock);
        oshmem_request_waiting++;
        while (false == req->req_complete) {
            opal_condition_wait(&oshmem_request_cond, &oshmem_request_lock);
        }
        oshmem_request_waiting--;
        OPAL_THREAD_UNLOCK(&oshmem_request_lock);
    }
}

/**
 *  Signal or mark a request as complete. If with_signal is true this will
 *  wake any thread pending on the request and oshmem_request_lock should be
 *  held while calling this function. If with_signal is false, there will
 *  signal generated, and no lock required. This is a special case when
 *  the function is called from the critical path for small messages, where
 *  we know the current execution flow created the request, and is still
 *  in the _START macro.
 */
static inline int oshmem_request_complete(oshmem_request_t* request,
                                          bool with_signal)
{
    if (NULL != request->req_complete_cb) {
        request->req_complete_cb(request);
    }
    oshmem_request_completed++;
    request->req_complete = true;
    if (with_signal && oshmem_request_waiting) {
        /* Broadcast the condition, otherwise if there is already a thread
         * waiting on another request it can use all signals.
         */
        opal_condition_broadcast(&oshmem_request_cond);
    }
    return OSHMEM_SUCCESS;
}

END_C_DECLS

#endif
