/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2016      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_SPML_H
#define MCA_SPML_H

#include "oshmem_config.h"
#include "oshmem/types.h"
#include "oshmem/constants.h"

#include "opal_stdint.h"
#include "oshmem/mca/mca.h"
#include "opal/mca/btl/btl.h"
#include "oshmem/proc/proc.h"

#include "oshmem/mca/sshmem/sshmem.h"

BEGIN_C_DECLS

/*
 * SPML component types
 */

/**
 * MCA->PML Called by MCA framework to initialize the component.
 *
 * @param priority (OUT) Relative priority or ranking used by MCA to
 * selected a component.
 *
 * @param enable_progress_threads (IN) Whether this component is
 * allowed to run a hidden/progress thread or not.
 *
 * @param enable_mpi_threads (IN) Whether support for multiple MPI
 * threads is enabled or not (i.e., MPI_THREAD_MULTIPLE), which
 * indicates whether multiple threads may invoke this component
 * simultaneously or not.
 */
typedef enum {
    MCA_SPML_BASE_PUT_SYNCHRONOUS,
    MCA_SPML_BASE_PUT_COMPLETE,
    MCA_SPML_BASE_PUT_BUFFERED,
    MCA_SPML_BASE_PUT_READY,
    MCA_SPML_BASE_PUT_STANDARD,
    MCA_SPML_BASE_PUT_SIZE
} mca_spml_base_put_mode_t;

typedef struct mca_spml_base_module_1_0_0_t * (*mca_spml_base_component_init_fn_t)(int *priority,
                                                                                   bool enable_progress_threads,
                                                                                   bool enable_mpi_threads);

typedef int (*mca_spml_base_component_finalize_fn_t)(void);

/**
 * SPML component version and interface functions.
 */
struct mca_spml_base_component_2_0_0_t {
    mca_base_component_t spmlm_version;
    mca_base_component_data_t spmlm_data;
    mca_spml_base_component_init_fn_t spmlm_init;
    mca_spml_base_component_finalize_fn_t spmlm_finalize;
};
typedef struct mca_spml_base_component_2_0_0_t mca_spml_base_component_2_0_0_t;
typedef mca_spml_base_component_2_0_0_t mca_spml_base_component_t;

/**
 * MCA management functions.
 */

static inline char *mca_spml_base_mkey2str(sshmem_mkey_t *mkey)
{
    static char buf[64];

    if (mkey->len == 0) {
        snprintf(buf, sizeof(buf), "mkey: base=%p len=%d key=%" PRIu64, mkey->va_base, mkey->len, mkey->u.key);
    } else {
        snprintf(buf, sizeof(buf), "mkey: base=%p len=%d data=0x%p", mkey->va_base, mkey->len, mkey->u.data);
    }

    return buf;
}

/**
 * Downcall from MCA layer to enable the PML/BTLs.
 *
 * @param   enable  Enable/Disable SPML forwarding
 * @return          OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_enable_fn_t)(bool enable);

/**
 * Waits for an int variable to change on the local PE.
 * Blocked until the variable is not equal to value.
 *
 * @param addr   Address of the variable to pool on.
 * @param value  The value to pool on. Pool until the value held in addr is different than value.
 * @return       OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_wait_fn_t)(void* addr,
                                              int cmp,
                                              void* value,
                                              int datatype);

/**
 * Test for an int variable to change on the local PE.
 *
 * @param addr   Address of the variable to pool on.
 * @param value  The value to pool on. Pool until the value held in addr is different than value.
 * @param out_value Return value to indicated if variable is equal to given cmp value.
 * @return       OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_test_fn_t)(void* addr,
                                              int cmp,
                                              void* value,
                                              int datatype,
                                              int *out_value);

/**
 * deserialize remote mkey
 *
 * @param mkey remote mkey
 */
typedef void (*mca_spml_base_module_mkey_unpack_fn_t)(shmem_ctx_t ctx, sshmem_mkey_t *, uint32_t segno, int remote_pe, int tr_id);

/**
 * If possible, get a pointer to the remote memory described by the mkey
 *
 * @param dst_addr  address of the symmetric variable
 * @param mkey      remote memory key
 * @param pe        remote PE
 *
 * @return pointer to remote memory or NULL
 */
typedef void * (*mca_spml_base_module_mkey_ptr_fn_t)(const void *dst_addr, sshmem_mkey_t *mkey, int pe);

/**
 * free resources used by deserialized remote mkey
 *
 * @param mkey remote mkey
 */
typedef void (*mca_spml_base_module_mkey_free_fn_t)(sshmem_mkey_t *, int pe);

/**
 * Register (Pinn) a buffer of 'size' bits starting in address addr
 *
 * @param addr   base address of the registered buffer.
 * @param size   the size of the buffer to be registered.
 * @param seg_id sysv segment id
 * @param count  number of internal transports (btls) that registered memory
 * @return       array of mkeys (one mkey per "btl") or NULL on failure
 *
 */
typedef sshmem_mkey_t * (*mca_spml_base_module_register_fn_t)(void *addr,
                                                                size_t size,
                                                                uint64_t shmid,
                                                                int *count);

/**
 * deregister memory pinned by register()
 */
typedef int (*mca_spml_base_module_deregister_fn_t)(sshmem_mkey_t *mkeys);

/**
 * try to fill up mkeys that can be used to reach remote pe.
 * @param pe         remote pe
 * @param seg 0 - symmetric heap, 1 - static data, everything else are static data in .so
 * @param mkeys      mkeys array
 *
 * @return OSHMEM_SUCCSESS if keys are found
 */
typedef int (*mca_spml_base_module_oob_get_mkeys_fn_t)(shmem_ctx_t ctx, int pe,
                                                       uint32_t seg,
                                                       sshmem_mkey_t *mkeys);

/**
 * For each proc setup a datastructure that indicates the BTLs
 * that can be used to reach the destination.
 *
 * @param procs  A list of all procs participating in the parallel application.
 * @param nprocs The number of procs in the parallel application.
 * @return OSHMEM_SUCCESS or failure status.
 *
 */
typedef int (*mca_spml_base_module_add_procs_fn_t)(struct oshmem_group_t* group,
                                                   size_t nprocs);
typedef int (*mca_spml_base_module_del_procs_fn_t)(struct oshmem_group_t* group,
                                                   size_t nprocs);


/**
 * Create a communication context.
 *
 * @param options  The set of options requested for the given context.
 * @param ctx      A handle to the newly created context.
 * @return         OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_ctx_create_fn_t)(long options, shmem_ctx_t *ctx);


/**
 * Destroy a communication context.
 *
 * @param ctx      Handle to the context that will be destroyed.
 */
typedef void (*mca_spml_base_module_ctx_destroy_fn_t)(shmem_ctx_t ctx);

/**
 * Transfer data to a remote pe.
 *
 * @param ctx      The context object this routine is working on.
 * @param dst_addr The address in the remote PE of the object being written.
 * @param size     The number of bytes to be written.
 * @param src_addr An address on the local PE holdng the value to be written.
 * @param dst      The remote PE to be written to.
 * @return         OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_put_fn_t)(shmem_ctx_t ctx,
                                             void *dst_addr,
                                             size_t size,
                                             void *src_addr,
                                             int dst);

/**
 * These routines provide the means for copying contiguous data to another PE without
 * blocking the caller. These routines return before the data has been delivered to the
 * remote PE.
 *
 * @param ctx      The context object this routine is working on.
 * @param dst_addr The address in the remote PE of the object being written.
 * @param size     The number of bytes to be written.
 * @param src_addr An address on the local PE holdng the value to be written.
 * @param dst      The remote PE to be written to.
 * @param handle   The address of a handle to be passed to shmem_wait_nb() or
 *                 shmem_test_nb() to wait or poll for the completion of the transfer.
 * @return         OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_put_nb_fn_t)(shmem_ctx_t ctx,
                                                void *dst_addr,
                                                size_t size,
                                                void *src_addr,
                                                int dst,
                                                void **handle);

/**
 * Blocking data transfer from remote PE.
 * Read data from remote PE.
 *
 * @param ctx      The context object this routine is working on.
 * @param dst_addr The address on the local PE, to write the result of the get operation to.
 * @param size     The number of bytes to be read.
 * @param src_addr The address on the remote PE, to read from.
 * @param src      The ID of the remote PE.
 * @return         OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_get_fn_t)(shmem_ctx_t ctx,
                                             void *dst_addr,
                                             size_t size,
                                             void *src_addr,
                                             int src);

/**
 * Non-blocking data transfer from remote PE.
 * Read data from remote PE.
 *
 * @param ctx      The context object this routine is working on.
 * @param dst_addr The address on the local PE, to write the result of the get operation to.
 * @param size     The number of bytes to be read.
 * @param src_addr The address on the remote PE, to read from.
 * @param src      The ID of the remote PE.
 * @param handle   The address of a handle to be passed to shmem_wait_nb() or
 *                 shmem_test_nb() to wait or poll for the completion of the transfer.
 * @return         - OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_get_nb_fn_t)(shmem_ctx_t ctx,
                                               void *dst_addr,
                                               size_t size,
                                               void *src_addr,
                                               int src,
                                               void **handle);

/**
 *  Post a receive and wait for completion.
 *
 *  @param buf (IN)         User buffer.
 *  @param count (IN)       The number of bytes to be sent.
 *  @param src (IN)         The ID of the remote PE.
 *  @return                 OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_recv_fn_t)(void *buf, size_t count, int src);

/**
 *  Post a send request and wait for completion.
 *
 *  @param buf (IN)         User buffer.
 *  @param count (IN)       The number of bytes to be sent.
 *  @param dst (IN)         The ID of the remote PE.
 *  @param mode (IN)        Send mode (STANDARD,BUFFERED,SYNCHRONOUS,READY)
 *  @return                 OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_send_fn_t)(void *buf,
                                              size_t count,
                                              int dst,
                                              mca_spml_base_put_mode_t mode);

/**
 *  The routine transfers the data asynchronously from the source PE to all
 *  PEs in the OpenSHMEM job. The routine returns immediately. The source and
 *  target buffers are reusable only after the completion of the routine.
 *  After the data is transferred to the target buffers, the counter object
 *  is updated atomically. The counter object can be read either using atomic
 *  operations such as shmem_atomic_fetch or can use point-to-point synchronization
 *  routines such as shmem_wait_until and shmem_test.
 *
 *  Shmem_quiet may be used for completing the operation, but not required for
 *  progress or completion. In a multithreaded OpenSHMEM program, the user
 *  (the OpenSHMEM program) should ensure the correct ordering of
 *  shmemx_alltoall_global calls.
 *
 *  @param dest        A symmetric data object that is large enough to receive
 *                     “size” bytes of data from each PE in the OpenSHMEM job.
 *  @param source      A symmetric data object that contains “size” bytes of data
 *                     for each PE in the OpenSHMEM job.
 *  @param size        The number of bytes to be sent to each PE in the job.
 *  @param counter     A symmetric data object to be atomically incremented after
 *                     the target buffer is updated.
 *
 *  @return            OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_put_all_nb_fn_t)(void *dest,
                                                    const void *source,
                                                    size_t size,
                                                    long *counter);

/**
 * Assures ordering of delivery of put() requests
 *
 * @param ctx      - The context object this routine is working on.
 * @return         - OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_fence_fn_t)(shmem_ctx_t ctx);

/**
 * Wait for completion of all outstanding put() requests
 *
 * @param ctx      - The context object this routine is working on.
 * @return         - OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_quiet_fn_t)(shmem_ctx_t ctx);

/**
 * Waits for completion of a non-blocking put or get issued by the calling PE.
 *
 * @return         - OSHMEM_SUCCESS or failure status.
 */
typedef int (*mca_spml_base_module_wait_nb_fn_t)(void *);

/**
 * Called by memheap when memory is allocated by shmalloc(),
 * shcalloc(), shmemalign() or shrealloc()
 *
 * @param addr   base address of the registered buffer.
 * @param size   the size of the buffer to be registered.
 */
typedef void (*mca_spml_base_module_memuse_hook_fn_t)(void *, size_t);

/**
 *  SPML instance.
 */
struct mca_spml_base_module_1_0_0_t {

    mca_spml_base_module_add_procs_fn_t spml_add_procs;
    mca_spml_base_module_del_procs_fn_t spml_del_procs;

    mca_spml_base_module_enable_fn_t spml_enable;
    mca_spml_base_module_register_fn_t spml_register;
    mca_spml_base_module_deregister_fn_t spml_deregister;
    mca_spml_base_module_oob_get_mkeys_fn_t spml_oob_get_mkeys;

    mca_spml_base_module_ctx_create_fn_t spml_ctx_create;
    mca_spml_base_module_ctx_destroy_fn_t spml_ctx_destroy;

    mca_spml_base_module_put_fn_t spml_put;
    mca_spml_base_module_put_nb_fn_t spml_put_nb;
    mca_spml_base_module_get_fn_t spml_get;
    mca_spml_base_module_get_nb_fn_t spml_get_nb;

    mca_spml_base_module_recv_fn_t spml_recv;
    mca_spml_base_module_send_fn_t spml_send;

    mca_spml_base_module_wait_fn_t spml_wait;
    mca_spml_base_module_wait_nb_fn_t spml_wait_nb;
    mca_spml_base_module_test_fn_t spml_test;
    mca_spml_base_module_fence_fn_t spml_fence;
    mca_spml_base_module_quiet_fn_t spml_quiet;

    mca_spml_base_module_mkey_unpack_fn_t spml_rmkey_unpack;
    mca_spml_base_module_mkey_free_fn_t   spml_rmkey_free;
    mca_spml_base_module_mkey_ptr_fn_t    spml_rmkey_ptr;

    mca_spml_base_module_memuse_hook_fn_t spml_memuse_hook;
    mca_spml_base_module_put_all_nb_fn_t  spml_put_all_nb;
    void *self;
};

typedef struct mca_spml_base_module_1_0_0_t mca_spml_base_module_1_0_0_t;
typedef mca_spml_base_module_1_0_0_t mca_spml_base_module_t;

/*
 * Macro for use in components that are of type spml
 */
#define MCA_SPML_BASE_VERSION_2_0_0 \
    OSHMEM_MCA_BASE_VERSION_2_1_0("spml", 2, 0, 0)

/*
 * macro for doing direct call / call through struct
 */
#if MCA_oshmem_spml_DIRECT_CALL

#include MCA_oshmem_spml_DIRECT_CALL_HEADER

#define MCA_SPML_CALL_STAMP(a, b) mca_spml_ ## a ## _ ## b
#define MCA_SPML_CALL_EXPANDER(a, b) MCA_SPML_CALL_STAMP(a,b)
#define MCA_SPML_CALL(a) MCA_SPML_CALL_EXPANDER(MCA_oshmem_spml_DIRECT_CALL_COMPONENT, a)

#else
#define MCA_SPML_CALL(a) mca_spml.spml_ ## a
#endif

OSHMEM_DECLSPEC extern mca_spml_base_module_t mca_spml;

END_C_DECLS
#endif /* MCA_SPML_H */
