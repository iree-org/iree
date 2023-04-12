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
 *
 * Interface into the SHMEM portion of the Open SHMEM Run Time Environment
 */

#ifndef OSHMEM_SHMEM_RUNTIME_H
#define OSHMEM_SHMEM_RUNTIME_H

#include "oshmem_config.h"
#include "shmem.h"

#include "opal/class/opal_list.h"
#include "opal/class/opal_hash_table.h"

#include "orte/runtime/orte_globals.h"
#include "ompi/include/mpi.h"
#include <pthread.h>

BEGIN_C_DECLS

/* Global variables and symbols for the SHMEM layer */

/** Is oshmem initialized? */
OSHMEM_DECLSPEC extern bool oshmem_shmem_initialized;
/** Has oshmem been aborted **/
OSHMEM_DECLSPEC extern bool oshmem_shmem_aborted;

/** Do we have multiple threads? */
OSHMEM_DECLSPEC extern bool oshmem_mpi_thread_multiple;
/** Thread level requested to \c MPI_Init_thread() */
OSHMEM_DECLSPEC extern int oshmem_mpi_thread_requested;
/** Thread level provided by Open MPI */
OSHMEM_DECLSPEC extern int oshmem_mpi_thread_provided;
/** Identifier of the main thread */
OSHMEM_DECLSPEC extern struct opal_thread_t *oshmem_mpi_main_thread;

OSHMEM_DECLSPEC extern MPI_Comm oshmem_comm_world;

typedef pthread_mutex_t shmem_internal_mutex_t;
OSHMEM_DECLSPEC extern shmem_internal_mutex_t shmem_internal_mutex_alloc;

OSHMEM_DECLSPEC extern shmem_ctx_t oshmem_ctx_default;

#   define SHMEM_MUTEX_INIT(_mutex)                                     \
    do {                                                                \
        if (oshmem_mpi_thread_provided == SHMEM_THREAD_MULTIPLE)        \
            pthread_mutex_init(&_mutex, NULL);                          \
    } while (0)
#   define SHMEM_MUTEX_DESTROY(_mutex)                                  \
    do {                                                                \
        if (oshmem_mpi_thread_provided == SHMEM_THREAD_MULTIPLE)        \
            pthread_mutex_destroy(&_mutex);                             \
    } while (0)
#   define SHMEM_MUTEX_LOCK(_mutex)                                     \
    do {                                                                \
        if (oshmem_mpi_thread_provided == SHMEM_THREAD_MULTIPLE)        \
            pthread_mutex_lock(&_mutex);                                \
    } while (0)
#   define SHMEM_MUTEX_UNLOCK(_mutex)                                   \
    do {                                                                \
        if (oshmem_mpi_thread_provided == SHMEM_THREAD_MULTIPLE)        \
            pthread_mutex_unlock(&_mutex);                              \
    } while (0)


/** Bitflags to be used for the modex exchange for the various thread
 *  levels. Required to support heterogeneous environments */
#define OSHMEM_THREADLEVEL_SINGLE_BF     0x00000001
#define OSHMEM_THREADLEVEL_FUNNELED_BF   0x00000002
#define OSHMEM_THREADLEVEL_SERIALIZED_BF 0x00000004
#define OSHMEM_THREADLEVEL_MULTIPLE_BF   0x00000008

/** In ompi_mpi_init: the lists of Fortran 90 mathing datatypes.
 * We need these lists and hashtables in order to satisfy the new
 * requirements introduced in MPI 2-1 Sect. 10.2.5,
 * MPI_TYPE_CREATE_F90_xxxx, page 295, line 47.
 */
extern opal_hash_table_t ompi_mpi_f90_integer_hashtable;
extern opal_hash_table_t ompi_mpi_f90_real_hashtable;
extern opal_hash_table_t ompi_mpi_f90_complex_hashtable;

/** version string of ompi */
OSHMEM_DECLSPEC extern const char oshmem_version_string[];

/**
 * Initialize the Open SHMEM environment
 *
 * @param argc argc, typically from main() (IN)
 * @param argv argv, typically from main() (IN)
 * @param requested Thread support that is requested (IN)
 * @param provided Thread support that is provided (OUT)
 *
 * @returns OSHMEM_SUCCESS if successful
 * @returns Error code if unsuccessful
 *
 * Intialize all support code needed for SHMEM applications.  This
 * function should only be called by SHMEM applications (including
 * singletons).  If this function is called, ompi_init() and
 * ompi_rte_init() should *not* be called.
 *
 * It is permissable to pass in (0, NULL) for (argc, argv).
 */
int oshmem_shmem_init(int argc, char **argv, int requested, int *provided);

/**
 * Finalize the Open SHMEM environment
 *
 * @returns OSHMEM_SUCCESS if successful
 * @returns Error code if unsuccessful
 *
 */
int oshmem_shmem_finalize(void);

/**
 * Abort SHMEM processes
 */
OSHMEM_DECLSPEC int oshmem_shmem_abort(int errcode);

/**
 * Allgather between all PEs
 */
OSHMEM_DECLSPEC int oshmem_shmem_allgather(void *send_buf, void *rcv_buf, int elem_size);

/**
 * Allgatherv between all PEs
 */
OSHMEM_DECLSPEC int oshmem_shmem_allgatherv(void *send_buf, void* rcv_buf, int send_count,
                            int *rcv_size, int* displs);

/**
 * Barrier between all PEs
 */
OSHMEM_DECLSPEC void oshmem_shmem_barrier(void);

/**
 * Register OSHMEM specific runtime parameters
 */
OSHMEM_DECLSPEC int oshmem_shmem_register_params(void);

#if OSHMEM_PARAM_CHECK == 1

#define RUNTIME_CHECK_ERROR(...)                                    \
    do {                                                            \
        fprintf(stderr, "[%s]%s[%s:%d:%s] ",                        \
                orte_process_info.nodename,                         \
                ORTE_NAME_PRINT(ORTE_PROC_MY_NAME),                 \
                __FILE__, __LINE__, __func__);                      \
        fprintf(stderr, __VA_ARGS__);                               \
    } while(0);

/**
 * Check if SHMEM API generates internal error return code
 * Note: most API does not return error code
 */
#define RUNTIME_CHECK_RC(x)    \
    if (OPAL_UNLIKELY(OSHMEM_SUCCESS != (x)))                                           \
    {                                                                                   \
        RUNTIME_CHECK_ERROR("Internal error is appeared rc = %d\n", (x));               \
    }

/**
 * Check if we called start_pes() and passed initialization phase
 */
#define RUNTIME_CHECK_INIT()    \
    if (OPAL_UNLIKELY(!oshmem_shmem_initialized))                                       \
    {                                                                                   \
        RUNTIME_CHECK_ERROR("SHMEM is not initialized\n");                              \
        oshmem_shmem_abort(-1);                                                         \
    }

/**
 * Check if we target PE is valid
 */
#define RUNTIME_CHECK_PE(x)    \
    if (OPAL_UNLIKELY(((x) < 0) ||                                                      \
                      ((int)(x) > (int)(orte_process_info.num_procs - 1))))             \
    {                                                                                   \
        RUNTIME_CHECK_ERROR("Target PE #%d is not in valid range\n", (x));              \
        oshmem_shmem_abort(-1);                                                         \
    }

/**
 * Check if required address is in symmetric address space
 */
#include "oshmem/mca/memheap/memheap.h"
#define RUNTIME_CHECK_ADDR(x)    \
    if (OPAL_UNLIKELY(!MCA_MEMHEAP_CALL(is_symmetric_addr((x)))))        \
    {                                                                                   \
        RUNTIME_CHECK_ERROR("Required address %p is not in symmetric space\n", ((void*)x));    \
        oshmem_shmem_abort(-1);                                                         \
    }
/* Check if address is in symmetric space or size is zero */
#define RUNTIME_CHECK_ADDR_SIZE(x,s)    \
    if (OPAL_UNLIKELY((s) && !MCA_MEMHEAP_CALL(is_symmetric_addr((x)))))        \
    {                                                                                   \
        RUNTIME_CHECK_ERROR("Required address %p is not in symmetric space\n", ((void*)x));    \
        oshmem_shmem_abort(-1);                                                         \
    }
#define RUNTIME_CHECK_WITH_MEMHEAP_SIZE(x)    \
    if (OPAL_UNLIKELY((long)(x) > MCA_MEMHEAP_CALL(size)))        \
    {                                                                                   \
        RUNTIME_CHECK_ERROR("Requested (%ld)bytes and it exceeds symmetric space size (%ld)bytes\n", (long)(x), MCA_MEMHEAP_CALL(size));    \
    }

#else

#define RUNTIME_CHECK_RC(x)     (x = x)
#define RUNTIME_CHECK_INIT()
#define RUNTIME_CHECK_PE(x)
#define RUNTIME_CHECK_ADDR(x)
#define RUNTIME_CHECK_ADDR_SIZE(x,s)
#define RUNTIME_CHECK_WITH_MEMHEAP_SIZE(x)

#endif  /* OSHMEM_PARAM_CHECK */

END_C_DECLS

#endif /* OSHMEM_SHMEM_RUNTIME_H */
