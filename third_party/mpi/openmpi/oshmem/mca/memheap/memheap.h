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

#ifndef MCA_MEMHEAP_H
#define MCA_MEMHEAP_H
#include "oshmem/mca/mca.h"
#include "oshmem/constants.h"
#include "oshmem/proc/proc.h"

#include "oshmem/mca/sshmem/sshmem.h"
#include "oshmem/mca/spml/spml.h"


BEGIN_C_DECLS
struct mca_memheap_base_module_t;

typedef struct memheap_context
{
    void*   user_base_addr;
    void*   private_base_addr;
    size_t  user_size;
    size_t  private_size;
} memheap_context_t;

/**
 * Component initialize
 */
typedef int (*mca_memheap_base_component_init_fn_t)(memheap_context_t *);

/*
 * Symmetric heap allocation. Malloc like interface
 */
typedef int (*mca_memheap_base_module_alloc_fn_t)(size_t, void**);

typedef int (*mca_memheap_base_module_memalign_fn_t)(size_t align,
                                                     size_t size,
                                                     void**);

typedef int (*mca_memheap_base_module_realloc_fn_t)(size_t newsize,
                                                    void *,
                                                    void **);

/*
 * Symmetric heap free.
 */
typedef int (*mca_memheap_base_module_free_fn_t)(void*);

/**
 * Service functions
 */

typedef sshmem_mkey_t * (*mca_memheap_base_module_get_local_mkey_fn_t)(void* va,
                                                                         int transport_id);

/*
 * Symmetric heap destructor.
 */
typedef int (*mca_memheap_base_module_finalize_fn_t)(void);

typedef int (*mca_memheap_base_is_memheap_addr_fn_t)(const void* va);

/* get mkeys from all ranks */
typedef void (*mca_memheap_base_mkey_exchange_fn_t)(void);

/*
 * memheap component descriptor. Contains component version, information and
 * init functions
 */
struct mca_memheap_base_component_2_0_0_t {
    mca_base_component_t memheap_version; /**< version */
    mca_base_component_data_t memheap_data; /**< metadata */
    mca_memheap_base_component_init_fn_t memheap_init; /**<init function */
};
typedef struct mca_memheap_base_component_2_0_0_t mca_memheap_base_component_2_0_0_t;
typedef struct mca_memheap_base_component_2_0_0_t mca_memheap_base_component_t;

/**
 * memheap module descriptor
 */
struct mca_memheap_base_module_t {
    mca_memheap_base_component_t                   *memheap_component;  /** Memory Heap Management Componenet */
    mca_memheap_base_module_finalize_fn_t           memheap_finalize;
    mca_memheap_base_module_alloc_fn_t              memheap_alloc;
    mca_memheap_base_module_memalign_fn_t           memheap_memalign;
    mca_memheap_base_module_realloc_fn_t            memheap_realloc;
    mca_memheap_base_module_free_fn_t               memheap_free;

    /*
     * alloc/free that should be used for internal allocation.
     * Internal memory does not count towards
     *  symmetric heap memory
     */
    mca_memheap_base_module_alloc_fn_t              memheap_private_alloc;
    mca_memheap_base_module_free_fn_t               memheap_private_free;

    mca_memheap_base_module_get_local_mkey_fn_t     memheap_get_local_mkey;
    mca_memheap_base_is_memheap_addr_fn_t           memheap_is_symmetric_addr;
    mca_memheap_base_mkey_exchange_fn_t             memheap_get_all_mkeys;

    /*
     * Total size of user available memheap
     */
    long                                            memheap_size;
};

typedef struct mca_memheap_base_module_t mca_memheap_base_module_t;

/*
 * Macro for use in components that are of type rcache
 */
#define MCA_MEMHEAP_BASE_VERSION_2_0_0 \
    OSHMEM_MCA_BASE_VERSION_2_1_0("memheap", 2, 0, 0)

/*
 * macro for doing direct call / call through struct
 */
#if MCA_oshmem_memheap_DIRECT_CALL

#include MCA_oshmem_memheap_DIRECT_CALL_HEADER

#define MCA_MEMHEAP_CALL_STAMP(a, b) mca_memheap_ ## a ## _ ## b
#define MCA_MEMHEAP_CALL_EXPANDER(a, b) MCA_MEMHEAP_CALL_STAMP(a,b)
#define MCA_MEMHEAP_CALL(a) MCA_MEMHEAP_CALL_EXPANDER(MCA_oshmem_memheap_DIRECT_CALL_COMPONENT, a)

#else
#define MCA_MEMHEAP_CALL(a) mca_memheap.memheap_ ## a
#endif

OSHMEM_DECLSPEC extern mca_memheap_base_module_t mca_memheap;

int mca_memheap_alloc_with_hint(size_t size, long hint, void**);

static inline int mca_memheap_base_mkey_is_shm(sshmem_mkey_t *mkey)
{
    return (0 == mkey->len) && (MAP_SEGMENT_SHM_INVALID != (int)mkey->u.key);
}

/**
 * check if memcpy() can be used to copy data to dst_addr
 * must be memheap address and segment must be mapped
 */
static inline int mca_memheap_base_can_local_copy(sshmem_mkey_t *mkey, void *dst_addr) {
    return mca_memheap.memheap_is_symmetric_addr(dst_addr) &&
        mca_memheap_base_mkey_is_shm(mkey);
}


END_C_DECLS

#endif /* MCA_MEMHEAP_H */
